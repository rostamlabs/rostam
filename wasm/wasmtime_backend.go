//go:build cgo

// SPDX-License-Identifier: Apache-2.0

// Package wasm provides a WebAssembly runtime for Rostam user-defined
// update functions. Modules are compiled at registration time via
// wasmtime-go (Cranelift JIT under cgo) and instantiated per-shard so
// host calls route through the right ops.TxContext.
//
// Determinism gate: at compile time, the module's imports are walked.
// Any import outside the small allowed set (the rostam host functions)
// rejects the module. This keeps Raft Apply deterministic across nodes
// — a precondition for replication.
package wasm

import (
	"errors"
	"fmt"
	"sync"
	"time"

	wasmtime "github.com/bytecodealliance/wasmtime-go/v45"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
)

// ErrBannedImport is returned by Compile when the module imports a
// non-deterministic or otherwise disallowed function. Mirrors the
// wazero backend's sentinel of the same name.
var ErrBannedImport = errors.New("wasm: module imports a banned function")

// Module is a validated WASM blob, backend-agnostic in terms of its
// public surface. Holds the source bytes for re-compile on a Runtime's
// engine; the engine-specific compiled artifact is built per-Runtime
// in AddModule.
type Module struct {
	source []byte
	// writesState is true when the module imports a state-mutating host
	// function (cache_put/cache_del/cache_expire). Computed once during the
	// determinism-gate import walk in Compile and carried to AddModule so
	// RegisterModule can reject an OpReadOnly module that would mutate state.
	writesState bool
}

// Bytes returns the original WASM source bytes.
func (m *Module) Bytes() []byte { return m.source }

// WritesState reports whether the module imports a state-mutating host function.
// It is derived from the module's IMPORTS at Compile time, so it is available on
// a compiled-but-not-yet-installed module — which is what lets a caller run the
// OpReadOnly guard (ValidateModuleKind) BEFORE AddModule swaps the runtime slot.
func (m *Module) WritesState() bool { return m.writesState }

// Close is a no-op for the wasmtime backend — the compiled module
// lives on each Runtime, not on the Module struct.
func (m *Module) Close() error { return nil }

// allowedImports is the determinism whitelist: only the rostam host
// functions are permitted. WASI is intentionally excluded — no WASI
// implementation is linked (see defineHostFunctions), and WASI calls
// such as fd_write have host-dependent, non-deterministic effects that
// would break the Raft Apply determinism guarantee. Whitelisting an
// import the linker cannot satisfy would also surface as a confusing
// unresolved-import error at instantiation time rather than at compile.
var allowedImports = map[[2]string]struct{}{
	{"rostam", "cache_get"}:    {},
	{"rostam", "cache_put"}:    {},
	{"rostam", "cache_del"}:    {},
	{"rostam", "cache_expire"}: {},
	{"rostam", "set_result"}:   {},
}

// writeHostFuncs is the subset of allowedImports that MUTATE cache state. A
// module can only call a host function it imports, so importing one of these is
// a sufficient (and necessary) signal that the module may write. RegisterModule
// uses this to reject an OpReadOnly module that imports any of them, enforcing
// the ops-registry invariant that OpReadOnly ops must not mutate state (they
// bypass Raft, so a write from one would silently diverge replicas).
var writeHostFuncs = map[[2]string]struct{}{
	{"rostam", "cache_put"}:    {},
	{"rostam", "cache_del"}:    {},
	{"rostam", "cache_expire"}: {},
}

// validateEngine is a shared engine used only by Compile to parse and
// validate module bytes for the determinism gate. A wasmtime.Engine is
// safe to reuse across module compilations, so sharing one avoids
// allocating and abandoning a cgo-backed engine on every registration.
var validateEngine = wasmtime.NewEngine()

// Compile parses and validates the WASM bytes via wasmtime, runs the
// determinism gate, and returns a *Module holding the source bytes for
// later runtime-specific re-compilation.
func Compile(bytes []byte) (*Module, error) {
	if len(bytes) == 0 {
		return nil, errors.New("wasm: empty module bytes")
	}
	// Parse + validate on the shared validation engine; the source bytes
	// are what we keep. Runtime instantiation builds the engine-specific
	// artifact on the Runtime's own engine.
	mod, err := wasmtime.NewModule(validateEngine, bytes)
	if err != nil {
		return nil, fmt.Errorf("wasm: compile: %w", err)
	}
	writesState := false
	for _, imp := range mod.Imports() {
		modName := imp.Module()
		name := ""
		if imp.Name() != nil {
			name = *imp.Name()
		}
		key := [2]string{modName, name}
		if _, ok := allowedImports[key]; !ok {
			return nil, fmt.Errorf("%w: %s.%s", ErrBannedImport, modName, name)
		}
		if _, ok := writeHostFuncs[key]; ok {
			writesState = true
		}
	}
	return &Module{source: bytes, writesState: writesState}, nil
}

// hostStateHolder carries the per-call mutable state for host
// functions. Allocated once per registered module at AddModule time
// and reused across every Invoke — the per-module mu in runtimeModule
// serializes access. tx is updated per Invoke; out is reset (truncated)
// at the top of Invoke and re-used so common-size results never
// reallocate.
type hostStateHolder struct {
	tx       *ops.TxContext
	out      []byte
	mem      *wasmtime.Memory
	store    *wasmtime.Store
	active   bool // true between Invoke entry and return; defensive
	tooLarge bool // set if set_result would exceed maxResultBytes

	// hostErr is the FIRST error a state-mutating host function got back from
	// the TxContext during this Invoke. Without it the error never crossed the
	// host boundary: the host func returned -1 to the guest, the guest returned
	// whatever it liked, and Invoke handed the caller a nil error.
	//
	// That is a silent-divergence hole on the REPLICATED path. WASM ops register
	// as ops.OpReadWrite, so they are serialised into the Raft log and applied by
	// the FSM on every replica. A cache.ErrFull raised by ONE replica's
	// cache_put/cache_del/cache_expire (page occupancy is per-node runtime state —
	// see shard/apply_class.go) would be swallowed, ApplyResponse.Err would be nil,
	// classifyApplyErr would never see it, and that replica would advance its
	// applied index while holding state its peers do not — permanently and
	// undetectably. Surfacing the error from Invoke is what lets
	// errors.Is(err, cache.ErrFull) hold at the FSM, classify as classFatal, and
	// halt the node instead.
	//
	// FIRST error wins: a guest that ignores -1 and keeps calling must not be able
	// to overwrite the failure that already compromised this apply. Reset at the
	// top of every Invoke (the holder is reused for the module's whole lifetime),
	// so it can never leak into a later call.
	hostErr error
}

// recordHostErr latches the first non-nil error seen during an Invoke. Callers
// still return the guest's -1 signal afterwards: the guest contract is unchanged,
// the error is what the HOST needs.
func (h *hostStateHolder) recordHostErr(err error) {
	if err != nil && h.hostErr == nil {
		h.hostErr = err
	}
}

// readBufOffset is the fixed offset in module memory where Rostam
// writes cache_get results. Modules read from here after cache_get
// returns. Matches the wazero backend.
const readBufOffset = 65536

// Resource limits applied to every untrusted module. These bound a
// single registration's blast radius so a malicious or buggy module
// cannot wedge a CPU core, exhaust host RAM, or blow the host stack.
const (
	// maxModuleMemBytes caps each module's linear memory growth so a
	// memory.grow loop cannot exhaust host RAM.
	maxModuleMemBytes int64 = 64 << 20 // 64 MiB

	// maxWasmStackBytes caps the guest call stack to bound deep/infinite
	// recursion.
	maxWasmStackBytes = 1 << 20 // 1 MiB

	// invokeWallClock is the wall-clock deadline for a single Invoke.
	// Fuel alone does not bound time spent inside host calls, so an
	// epoch-based deadline backstops host-call-bound loops too.
	invokeWallClock = 5 * time.Second

	// epochTickInterval is the period of the per-Runtime background epoch
	// ticker. Each tick bumps the engine epoch once; an Invoke's deadline is
	// expressed as a whole number of ticks (invokeWallClock / interval), so the
	// wall-clock deadline is quantised to this granularity. Smaller = tighter
	// deadline precision at the cost of more wakeups; 100ms is the usual
	// wasmtime-recommended value and is far finer than the 5s budget.
	epochTickInterval = 100 * time.Millisecond

	// maxResultBytes caps the total bytes a module may accumulate via
	// set_result across one Invoke, preventing guest-driven host heap
	// exhaustion independently of the fuel budget.
	maxResultBytes = 16 << 20 // 16 MiB
)

// Runtime owns a wasmtime Engine, a Linker pre-populated with the
// rostam host module, and a map of compiled modules keyed by op name.
// Concurrent Invoke on different ops is safe (per-module mu inside
// runtimeModule serializes individual modules).
//
// Wall-clock deadlines use a SINGLE per-Runtime background ticker that bumps
// the shared engine epoch at epochTickInterval. Each Invoke arms an epoch
// deadline of epochTicks ticks relative to the current epoch, so a tick only
// expires stores whose OWN budget has elapsed — one op timing out can never
// trap a fresh concurrent invoke of another module (they share the engine but
// hold independent, later, absolute deadlines).
//
// MODULES ARE KEYED BY CONTENT, NOT BY OP NAME. A ModuleID is a pure function of
// the (bytes, export name, fuel budget) triple a slot is instantiated from, so
// installing a module can never DISPLACE another one: two versions of one op, or
// two ops sharing one module, simply occupy the slots their content names. That
// is what retires the "the runtime holds the new bytes while the ops.Registry
// entry still describes the old ones" hazard — there is no longer a per-name slot
// to overwrite, and therefore nothing to roll back on a failed registry install.
// It is also the groundwork for per-group version binding (see the deferred
// design note in cluster/wasm_gate.go): several versions of one op can coexist
// here, and an apply can name the one it needs.
type Runtime struct {
	// groupBindingState is the published per-group version table
	// (GroupBindings). It is deliberately NOT guarded by mu: resolveModuleForInvoke
	// reads it on every WASM apply, from a different goroutine per shard group, and
	// a lock there would serialize invocations across groups against each other and
	// against every AddModule. It is a copy-on-write atomic pointer instead.
	groupBindingState

	mu      sync.RWMutex
	engine  *wasmtime.Engine
	modules map[ModuleID]*runtimeModule
	closed  bool

	// epochTicks is the per-Invoke epoch-deadline budget, in ticks of the
	// background ticker (>= 1). Set once at construction; read-only thereafter.
	epochTicks uint64
	// stopEpoch is closed by Close to stop the background ticker goroutine;
	// epochWG waits for it to exit so no epoch bump outlives the Runtime.
	stopEpoch chan struct{}
	epochWG   sync.WaitGroup
}

type runtimeModule struct {
	mu       sync.Mutex // serializes Invoke per module
	store    *wasmtime.Store
	instance *wasmtime.Instance
	apply    *wasmtime.Func
	memory   *wasmtime.Memory
	holder   *hostStateHolder
	maxFuel  uint64 // per-Invoke fuel budget (always > 0)
	// writesState is true when the module imports a state-mutating host
	// function. Carried from Module so RegisterModule can reject an OpReadOnly
	// registration whose module would mutate state.
	writesState bool
}

// HasModule reports whether id is instantiated on this Runtime. Callers use it
// to skip the (expensive) wasm.Compile that would otherwise precede a redundant
// AddModule.
func (r *Runtime) HasModule(id ModuleID) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.modules[id]
	return ok
}

// HoldModuleTableForTest acquires the module-table WRITE lock and returns the
// function that releases it. It is a test hook, and it is exported for one
// reason: the lock is unexported and the property it lets a test assert lives in
// another package.
//
// THE PROPERTY. AddModule takes this lock to install a module, and it is called
// from inside cluster's applyWASMRegistration, which holds cluster's own
// wasmApplyMu across it. So "an apply in flight" means BOTH locks held, and any
// code path that must not be able to deadlock against an apply must not touch
// either. cluster.TestWASMBlobGetTouchesNoApplyLock holds both and requires
// __wasm_blob_get__ to complete anyway; without this hook it could hold only
// wasmApplyMu, and the most plausible regression it exists to catch — serving a
// blob get by short-circuiting on HasModule — passed it.
//
// It grants nothing a caller could want in production: it hands back a blocked
// Runtime and every AddModule, HasModule, Invoke and Close on it until released.
func (r *Runtime) HoldModuleTableForTest() (release func()) {
	r.mu.Lock()
	return r.mu.Unlock
}

// NewRuntime constructs a Runtime with a fuel- and epoch-metered engine
// and an empty module map. SetConsumeFuel and SetEpochInterruption must
// be enabled on the engine config for the per-Invoke fuel and wall-clock
// deadlines applied in Invoke to take effect. It starts the background
// epoch ticker; Close stops it.
func NewRuntime() (*Runtime, error) {
	return newRuntimeWithTiming(invokeWallClock, epochTickInterval)
}

// newRuntimeWithTiming is the timing-parameterised constructor behind
// NewRuntime. wallClock is the per-Invoke deadline; tickInterval is the epoch
// ticker's period. The per-Invoke budget is ceil(wallClock/tickInterval) with a
// +1 guard so a tick landing immediately after SetEpochDeadline cannot shorten
// the budget below wallClock; it is always >= 2 (a single foreign epoch bump
// never trips a fresh invoke). Tests use short values; production uses the
// package constants.
func newRuntimeWithTiming(wallClock, tickInterval time.Duration) (*Runtime, error) {
	if tickInterval <= 0 {
		tickInterval = epochTickInterval
	}
	ticks := uint64(wallClock / tickInterval) //nolint:gosec // non-negative durations
	// +1 guards the boundary where the first tick fires just after an Invoke
	// arms its deadline; this keeps the effective budget >= wallClock.
	ticks++
	// Floor at 2 so the isolation guarantee holds even for a pathological
	// wallClock < tickInterval config: a single foreign epoch bump must never
	// trap a fresh invoke.
	if ticks < 2 {
		ticks = 2
	}

	cfg := wasmtime.NewConfig()
	cfg.SetConsumeFuel(true)
	cfg.SetEpochInterruption(true)
	cfg.SetMaxWasmStack(maxWasmStackBytes)
	r := &Runtime{
		engine:     wasmtime.NewEngineWithConfig(cfg),
		modules:    make(map[ModuleID]*runtimeModule),
		epochTicks: ticks,
		stopEpoch:  make(chan struct{}),
	}
	r.epochWG.Add(1)
	go r.runEpochTicker(tickInterval)
	return r, nil
}

// runEpochTicker bumps the engine epoch once per interval until Close signals
// stop. A single shared ticker (rather than a per-Invoke timer that increments
// the global epoch) is what keeps one invoke's timeout from trapping another:
// each Invoke's absolute deadline is relative to the epoch at its own start.
func (r *Runtime) runEpochTicker(interval time.Duration) {
	defer r.epochWG.Done()
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-r.stopEpoch:
			return
		case <-t.C:
			r.engine.IncrementEpoch()
		}
	}
}

// AddModule compiles m.Bytes() on this Runtime's engine, instantiates the module
// against a fresh linker with the rostam host functions, and stores the result
// under its content address. It returns that ModuleID, which is what Invoke and
// the registry handler closure name.
//
// IT TAKES NO OP NAME, and that is the point of this signature. A slot is
// identified by what it IS, so installing a module NEVER displaces another one:
// registering v2 of an op leaves v1 instantiated and invocable, two ops that
// share bytes/export/fuel share one slot, and a failed registry install
// afterwards has nothing to undo (the caller's old ModuleID still resolves to
// the old slot). The previous name-keyed map made every install a swap, which is
// what the now-deleted rollbackWASMRuntimeModule existed to repair.
//
// IDEMPOTENT: re-adding an identical module is a no-op. A WASM registration is
// replicated to every shard Raft group (see cluster.Node.Call), so the FSM hook
// that calls AddModule runs once per hosted group on every node — without this
// check each of those would spend a Cranelift compile and a fresh store, and
// abandon the previous pair.
//
// NOTHING IS EVER EVICTED. Superseded versions accumulate for the Runtime's
// lifetime, bounded by the number of DISTINCT registrations the node has
// installed since it started; retiring one is only safe once every group this
// node hosts has committed past it, which is the version-visibility problem (see the deferred
// design note in cluster/wasm_gate.go). Today in-place updates are refused, so
// the set is one slot per op name.
//
// maxFuel is the per-Invoke instruction budget for this module. The wire format
// documents 0 as "unspecified"; rather than run untrusted code unbounded, a 0
// budget is clamped to defaultMaxFuel. The store is also given a hard memory
// limit so a memory.grow loop cannot exhaust host RAM.
func (r *Runtime) AddModule(m *Module, exportName string, maxFuel uint64) (ModuleID, error) {
	id := ModuleIDFor(m.Bytes(), exportName, maxFuel)
	if exportName == "" {
		exportName = "apply"
	}
	if maxFuel == 0 {
		maxFuel = defaultMaxFuel
	}

	r.mu.RLock()
	closed := r.closed
	_, haveExisting := r.modules[id]
	r.mu.RUnlock()
	if closed {
		return ModuleID{}, errors.New("wasm: runtime closed")
	}
	if haveExisting {
		return id, nil // identical module already instantiated — nothing to do
	}

	compiled, err := wasmtime.NewModule(r.engine, m.Bytes())
	if err != nil {
		return ModuleID{}, fmt.Errorf("wasm: compile module %s on engine: %w", id, err)
	}

	store := wasmtime.NewStore(r.engine)
	// Cap linear-memory growth (negative = keep wasmtime default for the
	// unspecified limits). Bounds a runaway memory.grow loop.
	store.Limiter(maxModuleMemBytes, -1, -1, -1, -1)
	holder := &hostStateHolder{}
	linker := wasmtime.NewLinker(r.engine)

	if err := defineHostFunctions(linker, store, holder); err != nil {
		return ModuleID{}, fmt.Errorf("wasm: define hosts for module %s: %w", id, err)
	}

	instance, err := linker.Instantiate(store, compiled)
	if err != nil {
		return ModuleID{}, fmt.Errorf("wasm: instantiate module %s: %w", id, err)
	}
	applyFn := instance.GetFunc(store, exportName)
	if applyFn == nil {
		return ModuleID{}, fmt.Errorf("wasm: export %q not found in module %s", exportName, id)
	}
	mem := instance.GetExport(store, "memory")
	if mem == nil || mem.Memory() == nil {
		return ModuleID{}, fmt.Errorf("wasm: module %s has no exported memory", id)
	}

	// Wire the holder so host closures can resolve memory + store at call time.
	holder.mem = mem.Memory()
	holder.store = store
	// Pre-grow the result buffer to a reasonable size so set_result on
	// small results never reallocates. Small ops fit in 256 bytes; larger
	// ones grow on demand.
	holder.out = make([]byte, 0, 256)

	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return ModuleID{}, errors.New("wasm: runtime closed")
	}
	// Last-writer-wins on a key whose value is a pure function of the key: a
	// concurrent AddModule of the same content raced us and produced an
	// equivalent slot, so either is correct.
	r.modules[id] = &runtimeModule{
		store:       store,
		instance:    instance,
		apply:       applyFn,
		memory:      mem.Memory(),
		holder:      holder,
		maxFuel:     maxFuel,
		writesState: m.writesState,
	}
	r.mu.Unlock()
	return id, nil
}

// Invoke runs the module instantiated under id against tx. The args are written
// into module memory at offset 0 and (args_offset=0, args_len) are passed to
// apply(). Whatever apply called set_result() with is returned. Serialized
// per-module via runtimeModule.mu.
func (r *Runtime) Invoke(id ModuleID, tx *ops.TxContext, args []byte) ([]byte, error) {
	r.mu.RLock()
	closed := r.closed
	rm, ok := r.modules[id]
	r.mu.RUnlock()
	if closed {
		return nil, errors.New("wasm: runtime closed")
	}
	if !ok {
		return nil, fmt.Errorf("wasm: module %s not instantiated on this runtime", id)
	}
	rm.mu.Lock()
	defer rm.mu.Unlock()

	if len(args) > 0 {
		data := rm.memory.UnsafeData(rm.store)
		if len(data) < len(args) {
			return nil, errors.New("wasm: module memory too small for args")
		}
		copy(data[0:len(args)], args)
	}

	// Reuse the pre-allocated holder. tx is the only field that changes
	// per Invoke; out is truncated so set_result can append without
	// allocating in the common case.
	h := rm.holder
	h.tx = tx
	h.out = h.out[:0]
	h.tooLarge = false
	// Clear the latched host error BEFORE the guest can run, so a failure recorded
	// by a previous Invoke on this (reused, per-module) holder can never be
	// attributed to this one.
	h.hostErr = nil
	h.active = true
	defer func() {
		h.tx = nil
		h.active = false
	}()

	// Refuel and arm the wall-clock deadline for this Invoke. Fuel bounds
	// pure-compute loops; the epoch deadline backstops host-call-bound loops
	// that consume little fuel. The deadline is epochTicks ticks of the shared
	// per-Runtime ticker relative to the CURRENT epoch, so it expires only once
	// THIS invoke has run for ~invokeWallClock — a slow invoke of another module
	// cannot trap this one (its ticks are counted from its own, later, start).
	if err := rm.store.SetFuel(rm.maxFuel); err != nil {
		return nil, fmt.Errorf("wasm: set fuel: %w", err)
	}
	rm.store.SetEpochDeadline(r.epochTicks)

	res, err := rm.apply.Call(rm.store, int32(0), int32(len(args))) //nolint:gosec // bounded by args size
	// A failed host mutation OUTRANKS everything the guest reported — including a
	// trap. The guest only ever sees -1 and is free to ignore it, so its status code
	// (or a trap it took afterwards) says nothing about whether this node's state
	// still matches its peers'. The wrapped host error does, and it is the only form
	// classifyApplyErr can act on (errors.Is through the %w). Checked before the
	// call error for exactly that reason.
	if h.hostErr != nil {
		return nil, fmt.Errorf("wasm: host call failed: %w", h.hostErr)
	}
	if err != nil {
		return nil, fmt.Errorf("wasm: call: %w", err)
	}
	if h.tooLarge {
		return nil, fmt.Errorf("wasm: result exceeds %d bytes", maxResultBytes)
	}
	if code, ok := res.(int32); ok && code != 0 {
		return nil, fmt.Errorf("wasm: op returned status %d", code)
	}
	// Return a copy so the caller's slice survives the next Invoke
	// truncating h.out. (h.out itself is reused; callers must not
	// retain the returned slice's backing across calls.)
	if len(h.out) == 0 {
		return nil, nil
	}
	out := make([]byte, len(h.out))
	copy(out, h.out)
	return out, nil
}

// Close marks the Runtime closed and drops all per-module references.
// Subsequent AddModule/Invoke calls return "runtime closed" rather than
// writing into the cleared map (which would panic) or silently
// resurrecting modules on a Runtime the caller believes is closed. The
// underlying wasmtime objects (stores, instances, engine) are reclaimed
// by GC/finalizers once unreferenced.
func (r *Runtime) Close() error {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return nil
	}
	r.closed = true
	r.modules = nil
	close(r.stopEpoch)
	r.mu.Unlock()
	// Wait outside the lock so the ticker goroutine (which touches only the
	// engine, not r.mu) can observe stopEpoch and exit without deadlocking.
	r.epochWG.Wait()
	return nil
}

// moduleWritesState reports whether the module instantiated under id imports a
// state-mutating host function. ok is false when id is not instantiated. Used by
// RegisterModule to enforce the OpReadOnly no-write invariant.
func (r *Runtime) moduleWritesState(id ModuleID) (writes, ok bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	rm, found := r.modules[id]
	if !found {
		return false, false
	}
	return rm.writesState, true
}

// residentModuleState is the ONE lookup resolveModuleForInvoke makes per
// invocation: is this version instantiated here, and does it mutate state?
//
// It answers both from a single RLock because both are needed on every resolve
// and they must describe the SAME slot. Since thin markers, a binding can name a
// version whose bytes have not arrived yet (resident=false), and a version that
// HAS arrived was compile-verified by whoever pushed it but never kind-checked
// against THIS op's declared Kind — nothing about a blob carries an op name. So
// the resolver has to ask both questions together, on every path.
//
// The stub build answers differently on purpose. See wasm_stub.go.
func (r *Runtime) residentModuleState(id ModuleID) (writes, resident bool) {
	return r.moduleWritesState(id)
}

// defineHostFunctions registers the rostam.* imports on the linker.
// All closures read per-call state directly from holder (tx, out) and
// holder-attached module handles (mem, store). holder.active is the
// guard that says "we are inside an Invoke" — host calls that hit the
// closures outside an Invoke (shouldn't happen, but defensive) return
// an error.
func defineHostFunctions(linker *wasmtime.Linker, store *wasmtime.Store, holder *hostStateHolder) error {
	// cache_get(keyPtr, keyLen) -> i64 (high32=ptr, low32=len, or -1 on miss).
	if err := linker.DefineFunc(store, "rostam", "cache_get",
		func(keyPtr, keyLen int32) int64 {
			if !holder.active {
				return -1
			}
			if keyPtr < 0 || keyLen < 0 {
				return -1
			}
			data := holder.mem.UnsafeData(holder.store)
			if int64(keyPtr)+int64(keyLen) > int64(len(data)) {
				return -1
			}
			key := data[keyPtr : keyPtr+keyLen]
			v, err := holder.tx.Get(key)
			if err != nil {
				if errors.Is(err, cache.ErrNotFound) {
					return -1
				}
				return -1
			}
			if readBufOffset+len(v) > len(data) {
				return -1
			}
			copy(data[readBufOffset:readBufOffset+len(v)], v)
			return (int64(readBufOffset) << 32) | int64(len(v))
		}); err != nil {
		return err
	}

	// cache_put(keyPtr, keyLen, valPtr, valLen, ttlMs) -> i32 (0 = ok).
	if err := linker.DefineFunc(store, "rostam", "cache_put",
		func(keyPtr, keyLen, valPtr, valLen int32, ttlMs int64) int32 {
			if !holder.active {
				return -1
			}
			if keyPtr < 0 || keyLen < 0 || valPtr < 0 || valLen < 0 {
				return -1
			}
			data := holder.mem.UnsafeData(holder.store)
			if int64(keyPtr)+int64(keyLen) > int64(len(data)) || int64(valPtr)+int64(valLen) > int64(len(data)) {
				return -1
			}
			key := data[keyPtr : keyPtr+keyLen]
			val := data[valPtr : valPtr+valLen]
			var ttl time.Duration
			if ttlMs > 0 {
				ttl = time.Duration(ttlMs) * time.Millisecond
			}
			// PutIndexed, not Put: a guest write is a KV write like any other, so
			// it owes the record index the same maintenance a builtin handler
			// does — store first, post second. A bare Put would leave the posting
			// describing whatever the key held BEFORE the guest wrote it, and a
			// query for the new value would miss a live key. cache_del below
			// needs no counterpart (it removes a live slot, so the cache's
			// onRemove hook unposts), and cache_expire routes through
			// TxContext.Expire, which re-posts for the same reason.
			if err := holder.tx.PutIndexed(key, val, ttl); err != nil {
				// -1 is the guest's signal; recordHostErr is what carries the reason
				// out through Invoke so the replicated apply path can classify it.
				holder.recordHostErr(err)
				return -1
			}
			return 0
		}); err != nil {
		return err
	}

	// cache_del(keyPtr, keyLen) -> i32 (1 if existed, 0 if absent, -1 on err).
	if err := linker.DefineFunc(store, "rostam", "cache_del",
		func(keyPtr, keyLen int32) int32 {
			if !holder.active {
				return -1
			}
			if keyPtr < 0 || keyLen < 0 {
				return -1
			}
			data := holder.mem.UnsafeData(holder.store)
			if int64(keyPtr)+int64(keyLen) > int64(len(data)) {
				return -1
			}
			key := data[keyPtr : keyPtr+keyLen]
			existed, err := holder.tx.Del(key)
			if err != nil {
				// Same contract as the other guest-facing failures here: -1 is the
				// guest's "the host refused" signal. A persistent shard with no room
				// for the delete's tombstone record surfaces as cache.ErrFull — which
				// recordHostErr carries out of Invoke, because -1 alone would leave the
				// replicated apply path believing the delete succeeded.
				holder.recordHostErr(err)
				return -1
			}
			if existed {
				return 1
			}
			return 0
		}); err != nil {
		return err
	}

	// cache_expire(keyPtr, keyLen, ttlMs) -> i32.
	if err := linker.DefineFunc(store, "rostam", "cache_expire",
		func(keyPtr, keyLen int32, ttlMs int64) int32 {
			if !holder.active || ttlMs <= 0 {
				return -1
			}
			if keyPtr < 0 || keyLen < 0 {
				return -1
			}
			data := holder.mem.UnsafeData(holder.store)
			if int64(keyPtr)+int64(keyLen) > int64(len(data)) {
				return -1
			}
			key := data[keyPtr : keyPtr+keyLen]
			if err := holder.tx.Expire(key, time.Duration(ttlMs)*time.Millisecond); err != nil {
				holder.recordHostErr(err)
				return -1
			}
			return 0
		}); err != nil {
		return err
	}

	// set_result(ptr, len) - appends bytes from module memory into the
	// holder's pre-allocated result buffer (zero allocations in the
	// common case where len(out) <= cap(out)).
	if err := linker.DefineFunc(store, "rostam", "set_result",
		func(ptr, length int32) {
			if !holder.active {
				return
			}
			if ptr < 0 || length < 0 {
				return
			}
			data := holder.mem.UnsafeData(holder.store)
			if int64(ptr)+int64(length) > int64(len(data)) {
				return
			}
			if len(holder.out)+int(length) > maxResultBytes {
				// Signal Invoke to fail rather than growing the host heap
				// without bound. Stop accumulating further bytes.
				holder.tooLarge = true
				return
			}
			holder.out = append(holder.out, data[ptr:ptr+length]...)
		}); err != nil {
		return err
	}
	return nil
}
