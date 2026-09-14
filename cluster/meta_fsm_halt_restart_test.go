// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/raft/mux"
	"github.com/rostamlabs/rostam/shard"
)

// metaHaltChildEnv selects the child phase of TestMetaUnknownOpHaltsAgainAfterRestart
// ("first" or "restart"); metaHaltDirEnv names the data directory both lives share.
const (
	metaHaltChildEnv = "ROSTAM_META_HALT_CHILD"
	metaHaltDirEnv   = "ROSTAM_META_HALT_DIR"
)

// metaTestUnknownOp is an op number no binary recognises. Kept well above the
// registry so a future op cannot quietly make this test meaningless.
const metaTestUnknownOp Op = 200

var (
	metaHaltIndexRe   = regexp.MustCompile(`at log index (\d+)`)
	metaHaltAppliedRe = regexp.MustCompile(`applied_index=(\d+)`)
	metaKnownRe       = regexp.MustCompile(`KNOWN_FRONTIER=(\d+)`)
)

// TestMetaUnknownOpHaltsAgainAfterRestart drives the PRODUCTION halt — the
// default one, not a test hook — through a real meta-Raft on a real data
// directory, across a real process restart.
//
// The first life commits a known entry, snapshots, then commits an entry with an
// op no binary recognises. The node must exit on it, and report that its applied
// frontier is still the known one. The second life reopens the same directory:
// Raft restores the snapshot and replays the log, and the node must meet the
// SAME entry and exit on it AGAIN. If the first life had recorded the entry as
// applied anywhere durable — a snapshot covering it — the second life would
// never see it, which is the silent skip this test exists to catch.
//
// It has to be a child process: the production halt is os.Exit, and os.Exit is
// the whole point — it is what keeps a deferred frontier advance from running.
func TestMetaUnknownOpHaltsAgainAfterRestart(t *testing.T) {
	if phase := os.Getenv(metaHaltChildEnv); phase != "" {
		runMetaHaltChild(t, phase, os.Getenv(metaHaltDirEnv))
		return
	}
	dir := t.TempDir()

	out1, exit1 := runMetaHaltPhase(t, "first", dir)
	if exit1 != 1 || !metaHaltIndexRe.MatchString(out1) {
		t.Fatalf("first life: exit=%d, want a halt (exit 1) on the unknown op; output:\n%s", exit1, out1)
	}
	known := mustMatchUint(t, metaKnownRe, out1, "first life known frontier")
	poison := mustMatchUint(t, metaHaltIndexRe, out1, "first life halt index")
	applied1 := mustMatchUint(t, metaHaltAppliedRe, out1, "first life applied frontier")
	if poison <= known {
		t.Fatalf("halt index %d must be past the known frontier %d", poison, known)
	}
	if applied1 != known {
		t.Fatalf("first life halted with applied frontier %d, want %d: the unknown entry must not be recorded as applied", applied1, known)
	}

	out2, exit2 := runMetaHaltPhase(t, "restart", dir)
	if exit2 != 1 || !metaHaltIndexRe.MatchString(out2) {
		t.Fatalf("restart: exit=%d, want the node to stop on the same entry again (exit 1), not skip it; output:\n%s", exit2, out2)
	}
	if got := mustMatchUint(t, metaHaltIndexRe, out2, "restart halt index"); got != poison {
		t.Fatalf("restart halted at log index %d, want the same entry %d", got, poison)
	}
	if got := mustMatchUint(t, metaHaltAppliedRe, out2, "restart applied frontier"); got != known {
		t.Fatalf("restart halted with applied frontier %d, want %d", got, known)
	}
}

func runMetaHaltPhase(t *testing.T, phase, dir string) (string, int) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestMetaUnknownOpHaltsAgainAfterRestart$", "-test.count=1", "-test.timeout=60s") //nolint:gosec // re-executing this test binary
	cmd.Env = append(os.Environ(), metaHaltChildEnv+"="+phase, metaHaltDirEnv+"="+dir)
	out, err := cmd.CombinedOutput()
	var ee *exec.ExitError
	switch {
	case err == nil:
		return string(out), 0
	case errors.As(err, &ee):
		return string(out), ee.ExitCode()
	default:
		t.Fatalf("%s: run child: %v", phase, err)
		return "", -1
	}
}

func mustMatchUint(t *testing.T, re *regexp.Regexp, s, what string) uint64 {
	t.Helper()
	m := re.FindStringSubmatch(s)
	if m == nil {
		t.Fatalf("%s: no match for %q in output:\n%s", what, re, s)
	}
	v, err := strconv.ParseUint(m[1], 10, 64)
	if err != nil {
		t.Fatalf("%s: %v", what, err)
	}
	return v
}

// runMetaHaltChild is one life of the node. Reaching the end of either phase
// means the node did NOT halt; the parent reads that from the exit code.
func runMetaHaltChild(t *testing.T, phase, dir string) {
	if dir == "" {
		t.Fatal("child: no data dir")
	}
	sl, err := mux.New("127.0.0.1:0", []uint32{metaGroupID}, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	self := Peer{NodeID: "node1", RaftAddr: sl.Addr().String(), ServerAddr: "127.0.0.1:0"}
	cfg := Config{
		NodeID:    "node1",
		DataDir:   dir,
		NumShards: 4,
		Bootstrap: true,
		ShardCfg:  shard.Config{RaftHeartbeatMs: 50, RaftElectionMs: 100, NoSync: true},
		Ops:       reg,
		Peers:     []Peer{self},
		RaftAddr:  sl.Addr().String(),
	}
	mr, err := startMetaRaft(cfg, metaTransport(sl))
	if err != nil {
		t.Fatal(err)
	}
	if err := waitForAnyLeader(mr.Raft, 10*time.Second); err != nil {
		t.Fatal(err)
	}

	switch phase {
	case "first":
		if err := mr.ApplySetCatalogEntry("default/docs", 8, 0, 5*time.Second); err != nil {
			t.Fatal(err)
		}
		// A snapshot is what would make a skip DURABLE: raft restores it on
		// restart and replays only what follows it.
		if err := mr.Raft.Snapshot().Error(); err != nil {
			t.Fatal(err)
		}
		fmt.Printf("KNOWN_FRONTIER=%d\n", mr.FSM.AppliedIndex())
		data, err := encodeLogEntry(LogEntry{Op: metaTestUnknownOp})
		if err != nil {
			t.Fatal(err)
		}
		f := mr.Raft.Apply(data, 5*time.Second)
		_ = f.Error()
		// Still alive: the entry was skipped. Snapshot past it so the restart
		// inherits the skip, exactly as a live node's periodic snapshot would.
		fmt.Printf("NOT_HALTED index=%d frontier=%d\n", f.Index(), mr.FSM.AppliedIndex())
		_ = mr.Raft.Snapshot().Error()
	case "restart":
		// Raft replays committed entries once it is leader again; give it ample
		// time to reach the unknown one.
		time.Sleep(3 * time.Second)
		fmt.Printf("NOT_HALTED frontier=%d\n", mr.FSM.AppliedIndex())
	default:
		t.Fatalf("child: unknown phase %q", phase)
	}
	_ = mr.Close()
	_ = sl.Close()
}
