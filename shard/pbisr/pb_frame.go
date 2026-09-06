// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"sync"
)

// pbReadBufPool recycles inbound frame payload buffers for readers that OPT IN
// (pbFrameReader.pooled) — today only the backup's serveConn, whose payloads
// are provably dead once the receiver returns: Receive applies Data
// synchronously (the store copies what it keeps) and ReceiveGroup's decode
// copies per record. The primary link's readLoop does NOT opt in: roundTrip
// hands its payload to a waiting goroutine that outlives deliver.
var pbReadBufPool = sync.Pool{
	New: func() any {
		b := make([]byte, 0, 512)
		return &b
	},
}

// pbGetReadBuf returns an n-byte buffer, pool-backed when capacity allows.
func pbGetReadBuf(n int) []byte {
	bp := pbReadBufPool.Get().(*[]byte)
	if cap(*bp) >= n {
		return (*bp)[:n]
	}
	return make([]byte, n) // undersized slot is dropped; the release refills the pool
}

// pbPutReadBuf recycles a buffer obtained (directly or by growth) for a pooled
// reader. Oversized buffers are dropped to bound retained memory.
func pbPutReadBuf(b []byte) {
	if cap(b) > int(pbPayloadMaxCap) {
		return
	}
	b = b[:0]
	pbReadBufPool.Put(&b)
}

// Batched-transport wire frame, little-endian header, length-delimited:
//
//	[magic u8][ver u8][kind u8][shard u32][reqID u64][payloadLen u32][payload…]
//
// This is the same shape as raft/fabric's mux frame (see raft/fabric/frame.go),
// ported here so pbisr's batched transport does not depend on raft/fabric. kind's
// high bit marks a response (pbKindResponse); the low bits are reserved for
// future message-type discrimination (only one message shape exists today, so
// pbKindReplicate is the sole request kind). shard routes a REQUEST to that
// shard's receiver; on a response it is informational since correlation is by
// reqID (shards interleave on the shared conn, so responses can arrive out of
// shard-order).
const (
	pbFrameMagic uint8 = 0x50 // 'P' for pbisr

	// pbFrameVersion 2 added PrevEpoch to the replicate/group payloads
	// and gave the catch-up handshake its own response payload (CatchupInfoMsg)
	// instead of reusing the AckMsg codec. Both are INCOMPATIBLE payload-layout
	// changes, so the version is bumped: a v1 peer's frames are rejected outright
	// (errPBBadVersion) rather than misparsed into a plausible-looking write. PB is
	// cluster-homogeneous (whole cluster runs one -replication-mode), so no
	// negotiation/downgrade path is offered — a mixed-version cluster fails loudly at the frame boundary,
	// which is the only safe answer for a protocol whose whole job is deciding
	// which writes are real.
	pbFrameVersion uint8 = 2

	// pbFrameHeaderSize is magic+ver+kind (3) + shard(4) + reqID(8) + payloadLen(4).
	pbFrameHeaderSize = 1 + 1 + 1 + 4 + 8 + 4

	// pbKindReplicate marks a single-write replicate request frame;
	// pbKindReplicateGroup marks a grouped request carrying k>=1 uniform-epoch,
	// seq-dense writes answered by ONE cumulative ack. pbKindResponse
	// is the high bit of kind, set on ack responses.
	//
	// pbKindCatchupReq marks a learner-catch-up HANDSHAKE request (ISR grow):
	// the growing primary asks a lagging survivor for its applied high-water so it
	// can compute the ring delta to backfill. Its request AND response payloads
	// both reuse the AckMsg codec — the request carries the primary's growing epoch
	// E (informational); the response is the survivor's CatchupInfo (Epoch =
	// backup watermark, Seq = backup lastApplied). The subsequent delta ships over
	// the existing pbKindReplicateGroup frames, NOT a new kind.
	//
	// pbKindSnapshotChunk carries ONE chunk of a full state transfer —
	// the repair for a ring-evicted or diverged target. It is a separate kind, and
	// a separate SYNCHRONOUS request/response, precisely so it does NOT ride the
	// peer sender path: submitLearnerLocked is non-blocking and abandons the grow
	// on a full channel, which a snapshot would trip on its first frame. Its
	// response is a plain AckMsg (OK == chunk accepted / install succeeded).
	pbKindReplicate      uint8 = 0
	pbKindReplicateGroup uint8 = 1
	pbKindCatchupReq     uint8 = 2
	pbKindSnapshotChunk  uint8 = 3
	pbKindResponse       uint8 = 0x80

	// pbGroupCountMax bounds the record count a group frame may declare, so a
	// corrupt/hostile count cannot drive an unbounded allocation on decode. The
	// sender's own grouping cap (pbGroupBatchMax) is far below this.
	pbGroupCountMax uint32 = 4096

	// pbMaxPayload bounds a single framed payload so a corrupt/hostile length
	// cannot drive an unbounded allocation.
	pbMaxPayload uint32 = 64 << 20
)

var (
	errPBBadMagic   = errors.New("pbisr: bad frame magic")
	errPBBadVersion = errors.New("pbisr: unsupported frame version")
	errPBOversize   = errors.New("pbisr: frame payload exceeds max")
)

// pbFrame is one message on the batched-transport wire: either a replicate
// request or an ack response, correlated by reqID.
//
// payload aliases a per-read buffer on the receive path, so consumers that
// outlive the read must copy. pooled marks an OUTBOUND payload borrowed from a
// buffer pool that must be returned once the writer has committed its bytes to
// the wire; it is never set on a read frame.
type pbFrame struct {
	kind    uint8
	shard   uint32
	reqID   uint64
	payload []byte
	pooled  bool
}

// writePBFrameHdr serializes f's header (little-endian) into dst, which must
// be at least pbFrameHeaderSize bytes. It does not write the payload; callers
// write dst followed by f.payload.
//
// Unlike the read path, this does NOT enforce pbMaxPayload — it trusts the
// caller's payload to already be bounded (op entries from Engine.Propose are
// bounded in practice). Only pbFrameReader.read enforces the cap, since that
// is the side that must defend against a corrupt/hostile length.
func writePBFrameHdr(dst []byte, f *pbFrame) {
	dst[0] = pbFrameMagic
	dst[1] = pbFrameVersion
	dst[2] = f.kind
	binary.LittleEndian.PutUint32(dst[3:], f.shard)
	binary.LittleEndian.PutUint64(dst[7:], f.reqID)
	binary.LittleEndian.PutUint32(dst[15:], uint32(len(f.payload))) //nolint:gosec // bounded by pbMaxPayload upstream
}

// pbFrameReader reassembles frames from a stream, tolerating partial TCP
// reads (io.ReadFull loops until the full header/payload arrives) and
// validating the magic, version, and payload-length bound before allocating.
type pbFrameReader struct {
	r *bufio.Reader
	// pooled draws payload buffers from pbReadBufPool; the CONSUMER of each
	// frame then owns returning the payload via pbPutReadBuf once it is dead
	// (see pbReadBufPool's safety note for who may opt in).
	pooled bool
	hdr    [pbFrameHeaderSize]byte
}

// read returns the next frame, or an error (io.EOF at a clean stream end,
// errPBBadMagic/errPBBadVersion/errPBOversize on a malformed stream, or the
// underlying read error). The returned payload is freshly allocated per call.
func (fr *pbFrameReader) read() (pbFrame, error) {
	if _, err := io.ReadFull(fr.r, fr.hdr[:]); err != nil {
		return pbFrame{}, err
	}
	if fr.hdr[0] != pbFrameMagic {
		return pbFrame{}, errPBBadMagic
	}
	if fr.hdr[1] != pbFrameVersion {
		return pbFrame{}, errPBBadVersion
	}
	f := pbFrame{
		kind:  fr.hdr[2],
		shard: binary.LittleEndian.Uint32(fr.hdr[3:]),
		reqID: binary.LittleEndian.Uint64(fr.hdr[7:]),
	}
	plen := binary.LittleEndian.Uint32(fr.hdr[15:])
	if plen > pbMaxPayload {
		return pbFrame{}, errPBOversize
	}
	if plen > 0 {
		if fr.pooled {
			f.payload = pbGetReadBuf(int(plen))
		} else {
			f.payload = make([]byte, plen)
		}
		if _, err := io.ReadFull(fr.r, f.payload); err != nil {
			return pbFrame{}, err
		}
	}
	return f, nil
}

// Slim payload codecs for the batched transport: unlike net_codec.go's
// encodeReplicateReq/decodeReplicateReq (which prefix the shard onto the
// payload for the old unbatched transport), these omit the shard since it now
// lives in the frame header. Layout (v2):
// epoch(8) seq(8) prevSeq(8) prevEpoch(8) dataLen(4) data.

// pbReplicateHdrSize is the fixed header of a single-write replicate payload.
const pbReplicateHdrSize = 8 + 8 + 8 + 8 + 4

// encodeReplicateMsg appends m's wire encoding to dst and returns the result,
// so callers can build into a reused/pooled buffer.
func encodeReplicateMsg(dst []byte, m ReplicateMsg) []byte {
	var hdr [pbReplicateHdrSize]byte
	binary.BigEndian.PutUint64(hdr[0:8], m.Epoch)
	binary.BigEndian.PutUint64(hdr[8:16], m.Seq)
	binary.BigEndian.PutUint64(hdr[16:24], m.PrevSeq)
	binary.BigEndian.PutUint64(hdr[24:32], m.PrevEpoch)
	binary.BigEndian.PutUint32(hdr[32:36], uint32(len(m.Data))) //nolint:gosec // bounded by pbMaxPayload upstream
	dst = append(dst, hdr[:]...)
	dst = append(dst, m.Data...)
	return dst
}

// decodeReplicateMsg is the inverse of encodeReplicateMsg. Data ALIASES b —
// ownership of b transfers to the returned message. That is safe for the one
// production caller (serveConn: pbFrameReader.read allocates a fresh payload
// per frame, and nothing else touches it after decode), and it removes a
// per-message copy from the backup hot path (2026-07-22 alloc profile). A
// caller decoding from a REUSED buffer must copy Data itself.
func decodeReplicateMsg(b []byte) (ReplicateMsg, error) {
	if len(b) < pbReplicateHdrSize {
		return ReplicateMsg{}, fmt.Errorf("pbisr: short replicate msg (%d bytes)", len(b))
	}
	var m ReplicateMsg
	m.Epoch = binary.BigEndian.Uint64(b[0:8])
	m.Seq = binary.BigEndian.Uint64(b[8:16])
	m.PrevSeq = binary.BigEndian.Uint64(b[16:24])
	m.PrevEpoch = binary.BigEndian.Uint64(b[24:32])
	dataLen := binary.BigEndian.Uint32(b[32:36])
	if int(dataLen) != len(b)-pbReplicateHdrSize {
		return ReplicateMsg{}, fmt.Errorf("pbisr: replicate msg data length mismatch (hdr %d, have %d)", dataLen, len(b)-pbReplicateHdrSize)
	}
	if dataLen > 0 {
		m.Data = b[pbReplicateHdrSize:]
	}
	return m, nil
}

// encodeReplicateGroup appends the group encoding of msgs to dst and returns
// the result. msgs MUST be non-empty, uniform-epoch, and seq-dense (the
// sender's grouping guarantees this: runSender breaks a batch on an epoch change
// or a seq discontinuity, and ringDeltaLocked truncates a replayed delta at an
// epoch boundary for exactly this reason). Per-record epoch/seq/prevSeq/prevEpoch
// are therefore implied and not repeated on the wire:
//
//	epoch(8) firstSeq(8) prevSeq(8) prevEpoch(8) count(4)  then count × [dataLen(4) data]
//
// Only the FIRST record's predecessor is carried explicitly: every later record's
// predecessor is the record before it in this same group, whose epoch IS the
// group's uniform epoch. That implication is precisely why the uniform-epoch
// precondition is load-bearing rather than an optimization — a group spanning an
// epoch boundary would rebuild a chain that never existed.
//
// The response to a group frame is a plain AckMsg payload with CUMULATIVE
// semantics — see Engine.ReceiveGroup.
func encodeReplicateGroup(dst []byte, msgs []ReplicateMsg) []byte {
	var hdr [pbGroupHdrSize]byte
	binary.BigEndian.PutUint64(hdr[0:8], msgs[0].Epoch)
	binary.BigEndian.PutUint64(hdr[8:16], msgs[0].Seq)
	binary.BigEndian.PutUint64(hdr[16:24], msgs[0].PrevSeq)
	binary.BigEndian.PutUint64(hdr[24:32], msgs[0].PrevEpoch)
	binary.BigEndian.PutUint32(hdr[32:36], uint32(len(msgs))) //nolint:gosec // bounded by pbGroupBatchMax upstream
	dst = append(dst, hdr[:]...)
	var dlen [4]byte
	for i := range msgs {
		binary.BigEndian.PutUint32(dlen[:], uint32(len(msgs[i].Data))) //nolint:gosec // bounded by pbMaxPayload upstream
		dst = append(dst, dlen[:]...)
		dst = append(dst, msgs[i].Data...)
	}
	return dst
}

// decodeReplicateGroup is the inverse of encodeReplicateGroup. It rebuilds the
// implied per-record (Epoch, Seq, PrevSeq, PrevEpoch) chain from the group header:
// record i>0 succeeds record i-1, whose (seq, epoch) are (firstSeq+i-1, epoch) —
// so its predecessor's epoch is the group's uniform epoch. Only record 0's
// predecessor comes off the wire. Record data is copied out of b (b aliases a
// per-read buffer).
func decodeReplicateGroup(b []byte) ([]ReplicateMsg, error) {
	if len(b) < pbGroupHdrSize {
		return nil, fmt.Errorf("pbisr: short replicate group (%d bytes)", len(b))
	}
	epoch := binary.BigEndian.Uint64(b[0:8])
	firstSeq := binary.BigEndian.Uint64(b[8:16])
	prevSeq := binary.BigEndian.Uint64(b[16:24])
	prevEpoch := binary.BigEndian.Uint64(b[24:32])
	count := binary.BigEndian.Uint32(b[32:36])
	if count == 0 || count > pbGroupCountMax {
		return nil, fmt.Errorf("pbisr: replicate group count %d out of range", count)
	}
	b = b[pbGroupHdrSize:]
	msgs := make([]ReplicateMsg, count)
	for i := range msgs {
		if len(b) < 4 {
			return nil, fmt.Errorf("pbisr: truncated replicate group at record %d", i)
		}
		dataLen := binary.BigEndian.Uint32(b[0:4])
		b = b[4:]
		if uint32(len(b)) < dataLen {
			return nil, fmt.Errorf("pbisr: truncated replicate group data at record %d", i)
		}
		seq := firstSeq + uint64(i)
		msgs[i] = ReplicateMsg{Epoch: epoch, Seq: seq, PrevSeq: seq - 1, PrevEpoch: epoch}
		if dataLen > 0 {
			msgs[i].Data = append([]byte(nil), b[:dataLen]...)
			b = b[dataLen:]
		}
	}
	msgs[0].PrevSeq = prevSeq
	msgs[0].PrevEpoch = prevEpoch
	if len(b) != 0 {
		return nil, fmt.Errorf("pbisr: %d trailing bytes after replicate group", len(b))
	}
	return msgs, nil
}

// pbGroupHdrSize is the fixed header of a group replicate payload:
// epoch(8) firstSeq(8) prevSeq(8) prevEpoch(8) count(4).
const pbGroupHdrSize = 8 + 8 + 8 + 8 + 4

// pbCatchupInfoSize is the fixed size of a catch-up handshake RESPONSE payload:
// epoch(8) appliedSeq(8) frontierSeq(8) frontierEpoch(8) ok(1). The catch-up
// REQUEST payload still reuses the AckMsg codec (it carries only the growing
// primary's epoch, informationally).
const pbCatchupInfoSize = 8 + 8 + 8 + 8 + 1

// encodeCatchupInfo appends c's wire encoding to dst and returns the result.
func encodeCatchupInfo(dst []byte, c CatchupInfoMsg) []byte {
	var hdr [pbCatchupInfoSize]byte
	binary.BigEndian.PutUint64(hdr[0:8], c.Epoch)
	binary.BigEndian.PutUint64(hdr[8:16], c.AppliedSeq)
	binary.BigEndian.PutUint64(hdr[16:24], c.FrontierSeq)
	binary.BigEndian.PutUint64(hdr[24:32], c.FrontierEpoch)
	if c.OK {
		hdr[32] = 1
	}
	return append(dst, hdr[:]...)
}

// decodeCatchupInfo is the inverse of encodeCatchupInfo.
func decodeCatchupInfo(b []byte) (CatchupInfoMsg, error) {
	if len(b) != pbCatchupInfoSize {
		return CatchupInfoMsg{}, fmt.Errorf("pbisr: bad catchup info length %d", len(b))
	}
	return CatchupInfoMsg{
		Epoch:         binary.BigEndian.Uint64(b[0:8]),
		AppliedSeq:    binary.BigEndian.Uint64(b[8:16]),
		FrontierSeq:   binary.BigEndian.Uint64(b[16:24]),
		FrontierEpoch: binary.BigEndian.Uint64(b[24:32]),
		OK:            b[32] == 1,
	}, nil
}

// pbSnapshotChunkHdrSize is the fixed header of a snapshot-chunk payload (Stage
// 4.3): epoch(8) frontierSeq(8) frontierEpoch(8) offset(8) total(8) final(1)
// dataLen(4). The two epochs are distinct and both load-bearing — see
// SnapshotChunk.
const pbSnapshotChunkHdrSize = 8 + 8 + 8 + 8 + 8 + 1 + 4

// encodeSnapshotChunk appends c's wire encoding to dst and returns the result.
func encodeSnapshotChunk(dst []byte, c SnapshotChunk) []byte {
	var hdr [pbSnapshotChunkHdrSize]byte
	binary.BigEndian.PutUint64(hdr[0:8], c.Epoch)
	binary.BigEndian.PutUint64(hdr[8:16], c.FrontierSeq)
	binary.BigEndian.PutUint64(hdr[16:24], c.FrontierEpoch)
	binary.BigEndian.PutUint64(hdr[24:32], c.Offset)
	binary.BigEndian.PutUint64(hdr[32:40], c.Total)
	if c.Final {
		hdr[40] = 1
	}
	binary.BigEndian.PutUint32(hdr[41:45], uint32(len(c.Data))) //nolint:gosec // bounded by pbSnapshotChunkBytes upstream
	dst = append(dst, hdr[:]...)
	return append(dst, c.Data...)
}

// decodeSnapshotChunk is the inverse of encodeSnapshotChunk. Data ALIASES b, so
// the receiver must copy anything it retains — Engine.ReceiveSnapshotChunk
// appends into its own staging buffer, which copies.
//
// The declared dataLen is cross-checked against the ACTUAL remaining bytes rather
// than used to slice, and Total is bounded by the engine (pbSnapshotMaxBytes)
// before it is allowed to influence anything — a wire-declared count never sizes
// a reservation here.
func decodeSnapshotChunk(b []byte) (SnapshotChunk, error) {
	if len(b) < pbSnapshotChunkHdrSize {
		return SnapshotChunk{}, fmt.Errorf("pbisr: short snapshot chunk (%d bytes)", len(b))
	}
	c := SnapshotChunk{
		Epoch:         binary.BigEndian.Uint64(b[0:8]),
		FrontierSeq:   binary.BigEndian.Uint64(b[8:16]),
		FrontierEpoch: binary.BigEndian.Uint64(b[16:24]),
		Offset:        binary.BigEndian.Uint64(b[24:32]),
		Total:         binary.BigEndian.Uint64(b[32:40]),
		Final:         b[40] == 1,
	}
	dataLen := binary.BigEndian.Uint32(b[41:45])
	if int(dataLen) != len(b)-pbSnapshotChunkHdrSize {
		return SnapshotChunk{}, fmt.Errorf("pbisr: snapshot chunk data length mismatch (hdr %d, have %d)", dataLen, len(b)-pbSnapshotChunkHdrSize)
	}
	if dataLen > 0 {
		c.Data = b[pbSnapshotChunkHdrSize:]
	}
	return c, nil
}

// encodeAckMsg appends a's wire encoding to dst and returns the result.
func encodeAckMsg(dst []byte, a AckMsg) []byte {
	var hdr [17]byte
	binary.BigEndian.PutUint64(hdr[0:8], a.Epoch)
	binary.BigEndian.PutUint64(hdr[8:16], a.Seq)
	if a.OK {
		hdr[16] = 1
	}
	return append(dst, hdr[:]...)
}

// decodeAckMsg is the inverse of encodeAckMsg.
func decodeAckMsg(b []byte) (AckMsg, error) {
	if len(b) != 17 {
		return AckMsg{}, fmt.Errorf("pbisr: bad ack msg length %d", len(b))
	}
	return AckMsg{
		Epoch: binary.BigEndian.Uint64(b[0:8]),
		Seq:   binary.BigEndian.Uint64(b[8:16]),
		OK:    b[16] == 1,
	}, nil
}
