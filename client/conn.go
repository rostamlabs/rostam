// SPDX-License-Identifier: Apache-2.0

package client

import (
	"bufio"
	"context"
	"crypto/tls"
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"time"
)

const connBufSize = 8192

// Conn is the per-connection handle exposed to lifecycle hooks. Users do
// not construct or call methods on Conn directly — Client.Call manages it.
// The fields are unexported to discourage poking; hooks receive the
// pointer to enable cfg-level identity checks (e.g. tagging).
type Conn struct {
	addr       string
	tcp        net.Conn
	r          *bufio.Reader
	w          *bufio.Writer
	createdAt  time.Time
	maxAgeTime time.Time

	// authToken, when non-empty, triggers protocol-v2 framing on every RPC.
	// Captured from the Client's Config at dial time so doCall doesn't have
	// to chase the config pointer per request.
	authToken string

	// Last call's success state. A poisoned conn (broken framing / EOF mid-frame)
	// is destroyed on Release instead of returned to the pool.
	poisoned bool

	// Reusable scratch buffers for one in-flight call. hdrBuf is 9 bytes so
	// doCall can reuse one Conn-resident array for both the request prefix
	// ([frameLen:4][opNameLen:1]) and the argsLen field — escape analysis
	// otherwise forces `var ... [N]byte` locals to the heap when their
	// slices flow into bufio.Writer.Write (whose interface call to the
	// underlying net.Conn is opaque). Keeping the array on Conn means
	// "allocated once per connection" not "once per RPC".
	hdrBuf   [9]byte
	respBody []byte
}

// Addr returns the server address this conn is dialed to.
func (c *Conn) Addr() string { return c.addr }

// dial opens a new TCP connection to addr with the given dial timeout and
// stamps lifetime metadata. Hooks (BeforeConnect/AfterConnect) are invoked
// by the pool around this call.
func dial(ctx context.Context, addr string, cfg *Config) (*Conn, error) {
	dialer := net.Dialer{Timeout: cfg.DialTimeout}
	tcp, err := dialer.DialContext(ctx, "tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("client: dial %s: %w", addr, err)
	}
	if t, ok := tcp.(*net.TCPConn); ok {
		_ = t.SetNoDelay(true)
		_ = t.SetKeepAlive(true)
		_ = t.SetKeepAlivePeriod(30 * time.Second)
	}
	// When TLS is configured, wrap the raw TCP conn and complete the handshake at
	// dial time so a TLS failure (wrong CA, mTLS no-cert rejection, ServerName
	// mismatch) surfaces HERE as a dial error rather than mid-RPC. The v1/v2
	// framing then rides unchanged over conn (a *tls.Conn satisfies net.Conn). nil
	// TLSConfig ⇒ the plaintext path is byte-identical to before.
	conn := tcp
	if cfg.TLSConfig != nil {
		tconn := tls.Client(tcp, cfg.TLSConfig)
		hctx := ctx
		if cfg.DialTimeout > 0 {
			var cancel context.CancelFunc
			hctx, cancel = context.WithTimeout(ctx, cfg.DialTimeout)
			defer cancel()
		}
		if herr := tconn.HandshakeContext(hctx); herr != nil {
			_ = tcp.Close()
			return nil, fmt.Errorf("client: TLS handshake %s: %w", addr, herr)
		}
		conn = tconn
	}
	c := &Conn{
		addr:      addr,
		tcp:       conn,
		r:         bufio.NewReaderSize(conn, connBufSize),
		w:         bufio.NewWriterSize(conn, connBufSize),
		createdAt: time.Now(),
		authToken: cfg.AuthToken,
	}
	c.maxAgeTime = c.createdAt.Add(cfg.MaxConnLifetime)
	if cfg.MaxConnLifetimeJitter > 0 {
		c.maxAgeTime = c.maxAgeTime.Add(jitterDuration(cfg.MaxConnLifetimeJitter))
	}
	return c, nil
}

// doCall sends one request frame and reads one response frame.
// Returns (status, payload, error). Payload aliases into c.respBody; the
// caller must consume it before the next doCall.
func (c *Conn) doCall(ctx context.Context, op string, args []byte, callTimeout time.Duration) (uint8, []byte, error) {
	deadline := time.Now().Add(callTimeout)
	if d, ok := ctx.Deadline(); ok && d.Before(deadline) {
		deadline = d
	}
	if err := c.tcp.SetDeadline(deadline); err != nil {
		c.poisoned = true
		return 0, nil, err
	}
	defer func() { _ = c.tcp.SetDeadline(time.Time{}) }()

	// Frame the request inline rather than building a heap []byte via
	// encodeRequest — bufio.Writer coalesces these small Writes into one
	// syscall on Flush.
	//
	// v1 wire (no auth):   [frameLen:4][opNameLen:1][opName][argsLen:4][args].
	// v2 wire (auth set):  [frameLen:4][version=2:1][tokenLen:1][token][opNameLen:1][opName][argsLen:4][args].
	//
	// The v1 hot path is preserved: when authToken is empty, the write
	// sequence is identical to the pre-v2 implementation.
	if len(op) > maxOpNameLen {
		c.poisoned = true
		return 0, nil, fmt.Errorf("client: opName length %d exceeds %d", len(op), maxOpNameLen)
	}
	v2Prefix := 0
	if c.authToken != "" {
		if len(c.authToken) > 255 {
			c.poisoned = true
			return 0, nil, fmt.Errorf("client: authToken length %d exceeds 255", len(c.authToken))
		}
		v2Prefix = 1 + 1 + len(c.authToken) // [version:1][tokenLen:1][token]
	}
	bodyLen := v2Prefix + 1 + len(op) + 4 + len(args)
	if bodyLen > MaxFrameSize {
		c.poisoned = true
		return 0, nil, ErrFrameTooLarge
	}
	// Use c.hdrBuf[0:5] for [frameLen:4][firstByte:1] and c.hdrBuf[5:9] for
	// [argsLen:4]; the response path will reuse hdrBuf[0:4] below.
	binary.BigEndian.PutUint32(c.hdrBuf[0:4], uint32(bodyLen)) //nolint:gosec // bounded above
	if c.authToken != "" {
		// v2: emit [version=2][tokenLen][token], then [opNameLen][opName].
		c.hdrBuf[4] = 0x02 // ProtocolV2
		if _, err := c.w.Write(c.hdrBuf[0:5]); err != nil {
			c.poisoned = true
			return 0, nil, err
		}
		// tokenLen + token piggyback on hdrBuf[4:5] reuse for the single byte.
		c.hdrBuf[4] = byte(len(c.authToken)) //nolint:gosec // bounded above
		if _, err := c.w.Write(c.hdrBuf[4:5]); err != nil {
			c.poisoned = true
			return 0, nil, err
		}
		if _, err := c.w.WriteString(c.authToken); err != nil {
			c.poisoned = true
			return 0, nil, err
		}
		c.hdrBuf[4] = byte(len(op)) //nolint:gosec // bounded by maxOpNameLen
		if _, err := c.w.Write(c.hdrBuf[4:5]); err != nil {
			c.poisoned = true
			return 0, nil, err
		}
	} else {
		// v1: [frameLen:4][opNameLen:1] coalesced into one Write.
		c.hdrBuf[4] = byte(len(op)) //nolint:gosec // bounded by maxOpNameLen
		if _, err := c.w.Write(c.hdrBuf[0:5]); err != nil {
			c.poisoned = true
			return 0, nil, err
		}
	}
	if _, err := c.w.WriteString(op); err != nil {
		c.poisoned = true
		return 0, nil, err
	}
	binary.BigEndian.PutUint32(c.hdrBuf[5:9], uint32(len(args))) //nolint:gosec // bounded above
	if _, err := c.w.Write(c.hdrBuf[5:9]); err != nil {
		c.poisoned = true
		return 0, nil, err
	}
	if len(args) > 0 {
		if _, err := c.w.Write(args); err != nil {
			c.poisoned = true
			return 0, nil, err
		}
	}
	if err := c.w.Flush(); err != nil {
		c.poisoned = true
		return 0, nil, err
	}

	if _, err := io.ReadFull(c.r, c.hdrBuf[0:4]); err != nil {
		c.poisoned = true
		return 0, nil, err
	}
	n := binary.BigEndian.Uint32(c.hdrBuf[0:4])
	if n == 0 || n > MaxFrameSize {
		c.poisoned = true
		return 0, nil, fmt.Errorf("client: invalid response frame length %d", n)
	}
	if cap(c.respBody) < int(n) {
		c.respBody = make([]byte, n)
	} else {
		c.respBody = c.respBody[:n]
	}
	if _, err := io.ReadFull(c.r, c.respBody); err != nil {
		c.poisoned = true
		return 0, nil, err
	}
	status, payload, err := decodeResponse(c.respBody)
	if err != nil {
		c.poisoned = true
		return 0, nil, err
	}
	return status, payload, nil
}

// ping sends the no-op __ping__ op. Returns nil on healthy round-trip.
func (c *Conn) ping(ctx context.Context, callTimeout time.Duration) error {
	status, _, err := c.doCall(ctx, "__ping__", nil, callTimeout)
	if err != nil {
		return err
	}
	if status != StatusOK {
		return fmt.Errorf("client: __ping__ returned status %d", status)
	}
	return nil
}

func (c *Conn) close() error {
	if c.tcp == nil {
		return nil
	}
	err := c.tcp.Close()
	c.tcp = nil
	return err
}

// isExpired reports whether the conn is past its lifetime.
func (c *Conn) isExpired() bool {
	return time.Now().After(c.maxAgeTime)
}

// jitterDuration returns a value in [0, d). Used only for connection-lifetime
// jitter — not security-critical. We use time.Now().UnixNano() % d for cheapness;
// the bias is irrelevant at this scale. time.Now is goroutine-safe so no lock
// is needed.
func jitterDuration(d time.Duration) time.Duration {
	if d <= 0 {
		return 0
	}
	return time.Duration(time.Now().UnixNano() % int64(d)) //nolint:gosec // jitter, not security-critical
}
