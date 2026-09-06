// SPDX-License-Identifier: Apache-2.0

// Command rostam-server runs a Rostam store reachable over one or more network
// transports — REST/JSON, gRPC, and the binary TCP protocol — all over a single
// store. It is the companion server for the client SDKs (see clients/python).
//
//	rostam-server -http :8080 -data ./data                 # single-node REST
//	rostam-server -http :8080 -grpc :9090 -tcp :7000       # all three, one store
//	rostam-server -http :8080 -api-key "$ROSTAM_KEY"       # require a bearer token
//
// Production secure mode — server TLS (HTTP/gRPC/TCP) plus per-collection RBAC.
// TLS + -keys-file is "full secure mode"; -tls-require-client-cert turns on mTLS
// (a verified client-cert CN authenticates when its CN is a -keys-file CertCN
// entry). No -tls-* flags ⇒ plaintext (the dev default):
//
//	rostam-server -http :8443 -tls-cert server.pem -tls-key server.key \
//	    -tls-ca ca.pem -tls-require-client-cert -keys-file keys.json
//
// File-based key administration (offline; operates on a -keys-file JSON):
//
//	rostam-server keys add -file keys.json -token T -tenant acme \
//	    -scopes read:default/docs,write:default/* [-cert-cn client-cn]
//	rostam-server keys revoke -file keys.json -token T
//	rostam-server keys list   -file keys.json          # tokens masked
//
// Replicated (Raft) mode distributes/replicates data across a cluster; vectors
// are partitioned across shards by collection name:
//
//	rostam-server -cluster -bootstrap -node-id n1 -raft-addr 127.0.0.1:7400 \
//	    -tcp :7000 -http :8080 -data ./n1               # bootstrap a 1-node cluster
//	rostam-server -cluster -node-id n2 -raft-addr 10.0.0.2:7400 \
//	    -peers "n1@10.0.0.1:7400@10.0.0.1:7000,n2@10.0.0.2:7400@10.0.0.2:7000" ...
//
// Online rebalancing — trigger a redistribution to a new membership / RF and
// exit (no server is started); the running cluster moves shards live:
//
//	rostam-server -reconfigure -replication-factor 2 \
//	    -peers "n1@10.0.0.1:7400@10.0.0.1:7000,n2@...,n3@..."   # grow/rebalance
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"crypto/tls"

	"net"
	"net/http"

	// G108 is about pprof being auto-registered on net/http's DEFAULT mux and
	// thus exposed by any server using it. That is not what happens here: the
	// handlers are only reachable through the private listener started under
	// ROSTAM_PPROF (see main), which is opt-in, unset by default, and documented
	// to be bound to loopback. No production transport serves DefaultServeMux.
	_ "net/http/pprof" //nolint:gosec // G108: reachable only via the opt-in ROSTAM_PPROF listener
	"runtime"

	rostam "github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/backup"
	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/objstore"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/rlog"
	"github.com/rostamlabs/rostam/tlsutil"
	"github.com/rostamlabs/rostam/vector"
)

// fatal logs a structured error through the process slog logger and exits
// non-zero — the slog replacement for log.Fatalf, so a startup/shutdown fatal
// prints in the SAME format (text or json) as every other server log rather than
// the stdlib log's date-time line. Like log.Fatalf it calls os.Exit(1), so no
// deferred functions run (the process is aborting a bad configuration).
func fatal(msg string, args ...any) {
	slog.Error(msg, args...)
	os.Exit(1)
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "keys" {
		runKeysCmd(os.Args[2:])
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "mcp" {
		runMcpCmd(os.Args[2:])
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "llm-proxy" {
		runLlmProxyCmd(os.Args[2:])
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "rag" {
		runRagCmd(os.Args[2:])
		return
	}
	// EXPERIMENT: debug pprof endpoint when ROSTAM_PPROF=host:port is set. Also
	// enables block + mutex profiling (off by default) so /debug/pprof/block and
	// /debug/pprof/mutex expose goroutine-hop waits and lock contention — the
	// scheduler-churn ceiling this profiling targets.
	if addr := os.Getenv("ROSTAM_PPROF"); addr != "" {
		runtime.SetBlockProfileRate(10000)                 // sample a block event ~every 10µs of blocking
		runtime.SetMutexProfileFraction(100)               // sample ~1/100 mutex contention events
		go func() { _ = http.ListenAndServe(addr, nil) }() //nolint:gosec,errcheck
	}
	showVersion := flag.Bool("version", false, "print the version and exit")
	helpAll := flag.Bool("help-all", false, "print every flag with its full description (-h prints a grouped summary)")
	httpAddr := flag.String("http", ":8080", "REST/JSON listen address (\"\" = disabled)")
	grpcAddr := flag.String("grpc", "", "gRPC listen address (\"\" = disabled)")
	tcpAddr := flag.String("tcp", "", "binary TCP listen address (\"\" = disabled)")
	epollTCP := flag.Bool("epoll", true, "serve -tcp via the epoll/event-loop transport. Default ON for SINGLE-NODE (it beats goroutine-per-connection under core pressure — up to ~1.4x at low concurrency on an 8-core co-located box — and is within noise on dedicated cores). In -cluster mode the effective default is OFF unless -epoll is set explicitly: the event loop executes dispatch INLINE, so a replicated write blocks the whole loop for a full replication round trip, capping write throughput at ~(loops/RTT) regardless of connections — measured 2.1x slower than the goroutine server on a real-network 3-node PB RF=2 cluster (61k vs 127k ops/s @128 conns; see shard/pbisr/BENCHMARK.md). Plaintext only: with TLS configured the TCP transport silently falls back to the goroutine server")
	epollLoops := flag.Int("epoll-loops", 0, "epoll event-loop count (0 = GOMAXPROCS); ignored under -epoll=false or TLS")
	configFile := flag.String("config", "", "path to a JSON config file. Currently carries the cache stanza: {\"cache\":{\"max_memory\":\"8GiB\"}} bounds TOTAL cache memory for this node (default: a fraction of host RAM). Sized knobs live here rather than as flags because they belong together and this command line is already large")
	data := flag.String("data", "", "data directory (empty = in-memory, no persistence)")
	shards := flag.Int("shards", 0, "shards (cache shards in single-node mode; Raft shards in -cluster mode; 0 = default). TUNING (-cluster): each Raft shard runs its own goroutine set, and the replicated-write path is scheduler-bound, so over-sharding (many groups per core) burns CPU on goroutine churn. Rule of thumb: set Raft shards ≈ this node's core count (GOMAXPROCS), not far above it. Also set GOMAXPROCS to the cores this node actually owns — when co-locating multiple nodes on one box, GOMAXPROCS ≈ cores/nodes measurably cuts scheduler churn")
	apiKey := flag.String("api-key", "", "if set, require this bearer token on every request (granted *:* superuser scope). PREFER the ROSTAM_API_KEY env var: a flag-passed secret is visible to other local users via /proc and lands in shell history")
	keysFile := flag.String("keys-file", "", "path to a KeyRegistry JSON file enabling granular per-collection RBAC scopes (takes precedence over -api-key)")
	internalToken := flag.String("internal-token", "", "inter-node service token; forwarded ops carry it and the authorizer treats it as superuser (set this for clusters running with -keys-file/-api-key auth). PREFER the ROSTAM_INTERNAL_TOKEN env var: a flag-passed secret leaks via /proc and shell history")
	auditLog := flag.Bool("audit-log", false, "emit a structured JSON audit record (to stderr) for every RBAC authorization decision (principal redacted, raw token NEVER logged — only a non-reversible fingerprint); off by default (zero added cost). Applies only with -keys-file RBAC")
	logFormat := flag.String("log-format", "text", "server log output format: \"text\" (default; reads cleanly like the historical stderr logs — timestamp, level, message, key=value fields) or \"json\" (one JSON object per line for production log aggregation)")
	logLevel := flag.String("log-level", "info", "minimum server log level: debug|info|warn|error (default info)")
	accessLogFlag := flag.Bool("access-log", false, "emit one structured access-log line (to stderr) per client request on EVERY transport (HTTP/gRPC/TCP): request-id, transport, op, status, latency, REDACTED principal (token fingerprint or cert CN — raw token NEVER logged), and bytes. Reuses an inbound X-Request-Id header / gRPC metadata when present, else generates one. Off by default (zero added hot-path cost), mirroring -audit-log")
	tenantIsolation := flag.Bool("tenant-isolation", false, "enforce APIKey.Tenant as an authoritative resource boundary: after a key's scopes grant a request, ALSO require the resource's tenant == the key's Tenant (defense-in-depth so a mis-scoped key cannot cross tenants). Off by default = byte/behaviour-identical (scope-only). A key with Tenant=\"*\" is the cross-tenant/admin marker (exempt); the internal-service token stays superuser. Applies only with -keys-file RBAC")

	// Stateless JWT bearer acceptance (opt-in; HTTP/gRPC only — a JWT cannot fit
	// the binary TCP transport's 255B token cap, so it never arrives over TCP).
	// Setting -jwt-public-key is the trigger: a bearer that is NOT in -keys-file
	// AND looks like a JWT is verified against this public key (alg PINNED to the
	// key type: RSA->RS256, ECDSA-P256->ES256), with exp/nbf and the optional
	// -jwt-issuer/-jwt-audience checked, and a REQUIRED tenant + scopes claim. A
	// verified JWT yields a synthetic principal (its tenant + scopes claims) that
	// runs through the SAME per-collection RBAC scope-match + -tenant-isolation
	// guard + -audit-log as an API key. A verify FAILURE is fail-closed (deny +
	// audit; never a fallthrough). Unset = JWT-off (a JWT-looking token just fails
	// the registry lookup -> deny, byte-identical to before). Applies only with
	// -keys-file RBAC.
	jwtPublicKey := flag.String("jwt-public-key", "", "path to a PEM public key (RSA or ECDSA-P256) enabling stateless JWT bearer acceptance on HTTP/gRPC; alg is pinned to the key type (RS256/ES256). A verified JWT's tenant+scopes claims drive the SAME RBAC as an API key. Unset = JWT-off. Applies only with -keys-file RBAC")
	jwtIssuer := flag.String("jwt-issuer", "", "if set, a JWT's \"iss\" claim MUST equal this value (else deny); requires -jwt-public-key")
	jwtAudience := flag.String("jwt-audience", "", "if set, a JWT's \"aud\" claim MUST contain this value (else deny); requires -jwt-public-key")

	// TLS on the client-facing transports (HTTP/gRPC/TCP). Setting -tls-cert (with
	// -tls-key) enables server TLS on all three; -tls-ca + -tls-require-client-cert
	// turns it into mTLS where a verified client-cert CN authenticates via the
	// registry (its CN must be a CertCN entry). No -tls-* flags ⇒ plaintext, the
	// DEV default; PRODUCTION should set TLS (and pair it with -keys-file for full
	// secure mode: TLS + per-collection RBAC).
	tlsCert := flag.String("tls-cert", "", "server TLS certificate PEM (enables TLS on HTTP/gRPC/TCP; requires -tls-key)")
	tlsKey := flag.String("tls-key", "", "server TLS private key PEM (requires -tls-cert)")
	tlsCA := flag.String("tls-ca", "", "CA bundle PEM to verify client certs (mTLS); required with -tls-require-client-cert")
	tlsRequireClientCert := flag.Bool("tls-require-client-cert", false, "require+verify a client cert against -tls-ca (strict mTLS); a CN entry in -keys-file then authenticates cert-only clients")
	insecure := flag.Bool("insecure", false, "acknowledge running with NO authentication (-keys-file/-api-key both unset) on a non-loopback bind. Without this the server refuses to expose an open, unauthenticated datastore to the network; loopback-only binds are unaffected. For dev/trusted-network use only")
	// Dedicated node client cert for the inter-node dial (-cluster). For STRICT
	// inter-node mTLS (peers running -tls-require-client-cert), the cert this node
	// presents to peers MUST carry the clientAuth extended key usage. A typical
	// server cert (-tls-cert) is serverAuth-only and will be REJECTED by a peer's
	// RequireAndVerifyClientCert handshake. Provide a node cert with clientAuth EKU
	// here; if unset, the inter-node dial falls back to -tls-cert/-tls-key (fine for
	// server-TLS-only inter-node, or when -tls-cert already carries clientAuth EKU).
	tlsNodeCert := flag.String("tls-node-cert", "", "client cert PEM this node presents to peers on the inter-node dial (-cluster; needs clientAuth EKU for strict peer mTLS; defaults to -tls-cert)")
	tlsNodeKey := flag.String("tls-node-key", "", "private key PEM for -tls-node-cert (defaults to -tls-key)")
	// OPT-IN per-node mTLS identity. A CSV of trusted peer cert CommonNames. When
	// set, the inter-node CLIENT pins each dialed peer's verified cert CN to this
	// set (a CA-signed peer whose CN is absent fails the handshake), and the
	// authorizer additionally requires the internal-token caller's verified
	// ClientCN to be allowlisted (a leaked token + a non-allowlisted cert is
	// denied). Empty (default) = OFF = byte-identical to the shared-token/shared-CA
	// path. REQUIRES inter-node client-cert TLS (a -tls-cert cluster node, which
	// builds InterNodeTLS presenting this node's client cert) — set-without-it is
	// FATAL (it would reject all peers / be a no-op). Add every member's CN to
	// EVERY node's allowlist BEFORE starting a new node (the static-allowlist
	// join-window ordering).
	nodeCNAllowlist := flag.String("node-cn-allowlist", "", "OPT-IN per-node mTLS identity: CSV of trusted peer cert CNs (-cluster). Empty = OFF (byte-identical). Requires inter-node client-cert TLS (-tls-cert); the client verifies each peer's CN and the authorizer requires the internal-token caller's ClientCN to be allowlisted")

	cluster := flag.Bool("cluster", false, "run a replicated Raft-backed node (writes replicate; vectors partitioned by collection)")
	nodeID := flag.String("node-id", "node1", "cluster node id (-cluster)")
	raftAddr := flag.String("raft-addr", "127.0.0.1:7400", "Raft transport address (-cluster)")
	raftTransport := flag.String("raft-transport", "mux", "inter-node Raft transport (-cluster): \"mux\" (default) is the per-group NetworkTransport over one shared TCP listener; \"fabric\" is the multiplexed batching transport that carries every Raft group's traffic to a peer over a single connection (fewer syscalls, zero-reflection codec). EXPERIMENTAL: flag-gated; \"mux\" stays the default path")
	bootstrap := flag.Bool("bootstrap", false, "bootstrap a fresh cluster from -peers (first start only)")
	peers := flag.String("peers", "", "cluster peers as id@raftAddr@serverAddr, comma-separated (empty = self only)")
	replicationFactor := flag.Int("replication-factor", 0, "nodes per shard (-cluster; 0/>=peers = full replication, smaller = partition shards across nodes)")
	noSync := flag.Bool("nosync", false, "-cluster: skip fsync on Raft log writes. Trades crash-durability of the last few ms of writes for much higher throughput; replication still holds (a majority has every acked write in memory). Use when durability comes from replication rather than local disk — the same posture as Redis/Valkey with appendonly off or an in-memory Aerospike namespace")
	volatileLog := flag.Bool("volatile-log", false, "-cluster: put DATA shards' Raft logs fully in memory (no write() syscall on the replication hot path); durability comes only from replication. Goes further than -nosync (which still writes to the page cache). The meta group stays durable. SAFETY: a node that crashes MUST rejoin as a FRESH member (catch up from a leader snapshot), never resume in place, or its lost vote state can break Raft safety")
	persistentVectors := flag.Bool("persistent-vectors", false, "mmap-back vector collections off-heap (-cluster; Raft stays the durability authority)")
	pbFrontierStampInterval := flag.Duration("pb-frontier-stamp-interval", 0, "how often a PB shard persists its applied frontier into the cache header (-replication-mode pb; 0 = default 100ms). Each stamp is a full-region msync PER SHARD, so this trades stall FREQUENCY against stall SIZE and the metrics move in OPPOSITE directions — measured 100ms vs 1s on a 3-node 8-shard cluster, 6/6 pairs each way: 1s gave +4.1% throughput and -6.7% p99 but 2.7x WORSE p999 (8306us vs 3123us), because a rarer msync covers 10x more dirty pages. Raise it only if you are throughput-bound and do not care about the tail. It also bounds how much frontier advance a crash loses, which a restarted node's catch-up must cover, but it cannot affect correctness — a staler watermark only under-reports, and log matching turns that into a true-prefix catch-up or a clean divergence reject")
	disableColdCompaction := flag.Bool("disable-cold-compaction", false, "turn OFF the live-only rewrite of each persistent shard's pages file at open. Default (false) leaves it ON: it is the only thing that reclaims the page bytes left behind by overwritten and expired keys, and without it a persistent shard under TTL churn climbs to ErrFull and fails closed. This is the escape hatch if that rewrite ever misbehaves in the field — setting it costs reclamation only (the pages file is left exactly as found)")
	reconfigure := flag.Bool("reconfigure", false, "trigger an online rebalance to -peers / -replication-factor against a running cluster, then exit (does not start a server)")
	replicationMode := flag.String("replication-mode", "raft", "-cluster data-plane replication engine: \"raft\" (default) uses per-shard Raft groups, byte-identical to today. \"pb\" selects primary-backup/ISR replication (shard.ReplicationModePB) for every shard, with automatic failover on by default (see -pb-auto-failover). Recommended at replication-factor 2, where it beats raft on throughput and median (p50) latency; at RF=3 its full-ISR commit is slower than raft's majority. No acked-write loss across failover, under a bounded cross-node clock-rate assumption. Requires -min-isr and every node's PB listen address set via -pb-addr (or the peer's 4th @-field in -peers). See shard/pbisr/DESIGN.md for the replication model and its guarantees")
	minISR := flag.Int("min-isr", 0, "-cluster -replication-mode=pb: minimum in-sync-replica count required per shard (must be >= 1 in pb mode; unused in raft mode). "+
		"DURABILITY WARNING: -min-isr=1 provides NO no-acked-loss guarantee across failover, whatever the replication factor. "+
		"Every promotion resets the shard's in-sync set to the new primary alone, and with a floor of 1 that primary is permitted to "+
		"acknowledge writes held on no other node until the grow driver re-admits the backups (seconds). Those writes are durable only "+
		"on that one node: if it then fails, the shard has no in-sync survivor to promote and stays DOWN until it returns. Set -min-isr=2 "+
		"or higher (requires at least that many replicas) to keep every acknowledged write on a second node at all times")
	pbCommitPrimary := flag.Bool("pb-commit-primary", false, "-cluster -replication-mode=pb: commit a write on LOCAL primary apply and replicate to backups asynchronously (Aerospike commit-master posture). DURABILITY DOWNGRADE: an acked write can be lost if the primary dies before a backup received it. Default (false) waits for the full ISR, so no acked write is lost while any ISR member survives. Lower per-write latency, no throughput change on a pipelined path. Unused in raft mode")
	pbAutoFailover := flag.Bool("pb-auto-failover", true, "-cluster -replication-mode=pb: automatic primary failover. Each primary commits a periodic liveness beacon; when one goes silent past the failover timeout the meta leader promotes an ISR survivor (only from the ISR, and only one whose applied high-water is verified — so no acked write is lost), and the ISR shrink/grow drivers un-wedge a shard stalled on a dead backup and re-open it once a survivor catches up. Default ON: without it a PB shard whose primary dies stays DOWN until an operator intervenes. Set -pb-auto-failover=false for a STATIC cluster — byte-identical to the pre-Plan-4 behaviour (no beacon reaches the meta-Raft log, no epoch is ever bumped automatically). Unused in raft mode")
	wasmBlobRetention := flag.Duration("wasm-blob-retention", 0, "-cluster: how long a WASM module blob that nothing on this node references — a SUPERSEDED version, or a __wasm_blob_put__ orphan — is kept before its file is deleted. 0 (default) DISABLES retirement entirely: no sweeper runs and every version any shard group ever bound stays on disk forever, which is the safe answer and byte-identical to the behaviour before retirement existed. "+
		"SETTING IT IS AN ASSERTION, NOT A TUNING KNOB: no local rule can be safe against a lagging replica elsewhere (whether some other replica will still apply an entry under a superseded version is not a function of anything this node holds, and deciding it cluster-wide is cluster-wide GC, which is deliberately not built), so the value you set is your claim that no replica of any group this node hosts will be further behind the supersession of a module version than this. Err high — hours or days, not minutes. "+
		"If the claim is wrong, the lagging replica PARKS on the missing version: it is named with its exact fingerprint in Stats().WASMBlock, logged with escalating severity, and unblocked by one `rostam call __wasm_blob_put__ <fingerprint-hex><module-bytes>` against that node — no restart, no failover, no data movement. Only the FILE is removed; a module already instantiated stays in the runtime, so an executing invocation is unaffected")
	pbAddr := flag.String("pb-addr", "", "-cluster -replication-mode=pb: this node's pbisr.NetTransport listen endpoint, e.g. \"10.0.0.1:7200\". Required in pb mode. Sets this node's own Peer.PBAddr; other peers' PBAddr come from the 4th @-field of their -peers entry (id@raftAddr@serverAddr@pbAddr). Unused in raft mode")

	backupDir := flag.String("backup-dir", "", "filesystem directory to stream periodic collection snapshots into (empty = backup OFF). Future cloud targets (S3/GCS) implement the same objstore.ObjectStore interface")
	backupInterval := flag.Duration("backup-interval", 0, "how often to run a backup (e.g. 30m); 0 = OFF. Drives the -backup-dir FS backup and the -backup-bucket S3 backup")
	ttlSweepInterval := flag.Duration("ttl-sweep-interval", 30*time.Second, "how often each shard actively reaps expired TTL keys to reclaim memory (e.g. 30s, 5s); 0 DISABLES active reaping, leaving only lazy-on-read expiry (an expired key is still never returned) plus, on persistent shards, cold compaction at the next restart. This is a memory-reclaim-latency vs CPU-churn tradeoff, NOT a correctness knob: a slower sweep lets expired bytes linger longer, which on a write-heavy replicated shard raises the chance of hitting the cache cap between sweeps, so lower it if such a node is climbing toward ErrFull")
	onlineCompaction := flag.Bool("online-compaction", false, "opt every REPLICATED mmap reject-writes shard into ONLINE relocating compaction: while the process runs, dead overwritten ('ghost') bytes are reclaimed in place instead of only at restart, so a write/overwrite-heavy replicated shard keeps write capacity without a restart. OFF by default because of its safety contract: recycling overwrites retired page bytes, memory-safe ONLY when every read is released within the alias fence (AliasQuarantine = 2*-write-timeout) — i.e. all reads flow through the server transport, whose write deadline bounds the zero-copy response. Do NOT enable it if an in-process embedder retains a raw Store.Get result past the fence (that is not this standalone server, so it is safe here). No-op on heap/single-node/ring-buffer shards")
	backupPrefix := flag.String("backup-prefix", "default", "tenant/bucket key prefix for FS backup objects: keys are <prefix>/<collection>/<timestamp>.snap")
	backupRetention := flag.Int("backup-retention", 24, "keep only the newest N snapshots per collection (0 = keep all)")
	restore := flag.Bool("restore", false, "one-shot DISASTER RECOVERY (-cluster): after bring-up, restore this node's owned shards (cache + vectors) AND the MetaRaft catalog from the backup at -backup-dir (or -backup-bucket), then continue serving. Requires the SAME topology (shard count + node IDs) as the backup — a differing topology fails loud (placement remap is deferred). By default it ALSO fails loud if any shard has no backup artifact (a missing blob would bring that shard up empty, silently losing its keys); see -allow-missing-shards. Run once, on every node of a fresh cluster")
	allowMissingShards := flag.Bool("allow-missing-shards", false, "-restore: proceed even when one or more shards have NO backup artifact — those shards come up EMPTY (logged loudly per shard). Off by default so an incomplete backup fails loud rather than silently losing a shard's keys behind a clean 'restore complete'. An explicit operator override for a known-partial restore")

	// OPT-IN S3 cold-tiering / backup tier. Nothing set (-backup-bucket empty AND
	// -cold-tier-after 0) ⇒ ZERO behavior change: no objstore is constructed and no
	// goroutine is started. A bucket or cold-tier IS set but region/creds are
	// missing ⇒ FATAL at startup (mirrors the TLS/RBAC fail-loud validation). Creds
	// come from the AWS_* env (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
	// AWS_SESSION_TOKEN); they are NEVER read from a flag and NEVER logged. The one
	// S3 client is shared by the backup cron, the cold-tier sweeper, and the admin
	// REST surface.
	backupBucket := flag.String("backup-bucket", "", "S3 (compatible) bucket for backups / cold tiering (empty = S3 tier OFF). Creds from AWS_* env; pair with -backup-region")
	backupEndpoint := flag.String("backup-endpoint", "", "S3 endpoint override for MinIO/R2/GCS-S3 (e.g. http://127.0.0.1:9000); empty = AWS https://s3.<region>.amazonaws.com")
	backupRegion := flag.String("backup-region", "", "S3 region used for SigV4 request signing (required when -backup-bucket or -cold-tier-after is set)")
	backupTenant := flag.String("backup-tenant", "default", "top-level S3 key prefix (backup namespace) for S3 backup + cold-tier objects: <tenant>/<collection>/<ts>.snap")
	coldTierAfter := flag.Duration("cold-tier-after", 0, "idle-evict threshold: evict a collection to S3 after it is untouched this long (e.g. 1h); 0 = cold tiering OFF. Requires -backup-bucket")
	s3PathStyle := flag.Bool("s3-path-style", true, "use path-style S3 addressing (<endpoint>/<bucket>/<key>); true for MinIO/R2/localstack, false for AWS virtual-host")
	// -h prints the grouped summary. The stock alphabetical dump of 59 flags,
	// several with paragraph-length descriptions, is the first thing a new
	// operator meets and it is unreadable.
	flag.Usage = func() { printGroupedUsage(flag.CommandLine.Output(), flag.CommandLine, false) }
	flag.Parse()

	// -version answers and exits before anything else is set up: it must work on
	// a machine where the config, data dir or listen address would be rejected,
	// and it prints to stdout unshaped by -log-format.
	if *showVersion {
		fmt.Println(versionString())
		return
	}
	if *helpAll {
		printGroupedUsage(os.Stdout, flag.CommandLine, true)
		return
	}

	// Fill anything not given on the command line from ROSTAM_*. This runs
	// before the logger is built so ROSTAM_LOG_FORMAT/ROSTAM_LOG_LEVEL take
	// effect on the very first line, and before every fatal() below so a
	// container configured entirely by environment fails on the same rules a
	// flag-configured one does.
	if err := applyEnvDefaults(flag.CommandLine, os.LookupEnv); err != nil {
		fatal("invalid environment configuration", "err", err)
	}

	// Install the process-wide structured logger FIRST, so every subsequent line
	// (including the startup fatals below) renders in the configured format/level.
	// A bad -log-format/-log-level is itself fatal — but slog.Default is still the
	// stdlib text handler at this point, so the message is not lost.
	if _, err := rlog.Setup(*logFormat, *logLevel); err != nil {
		fatal("invalid logging configuration", "err", err)
	}
	// OPT-IN access log (mirrors -audit-log): nil when off, so no middleware/
	// interceptor is installed and every transport's hot path stays byte-identical.
	var accessLog *rlog.AccessLog
	if *accessLogFlag {
		al, err := rlog.New(*logFormat)
		if err != nil {
			fatal("invalid access-log configuration", "err", err)
		}
		accessLog = al
	}

	// Resolve the cache memory bound BEFORE any server is built so a bad
	// -config fails at startup rather than after the listeners are up. Zero =
	// unset, which the engine turns into a host-derived budget.
	var cacheMaxMemory int64
	if *configFile != "" {
		fc, err := loadFileConfig(*configFile)
		if err != nil {
			fatal("invalid -config file", "err", err)
		}
		if cacheMaxMemory, err = fc.cacheMaxMemoryBytes(); err != nil {
			fatal("invalid -config cache stanza", "err", err)
		}
	}

	// Resolve the TTL sweeper cadence into the sentinel the public CacheConfig
	// expects: negative disables active reaping, positive is the interval in ms.
	// (0 there would mean "library default"; the server always sets an explicit
	// value so its default is the flag's 30s, not the library's 1s.)
	ttlSweepIntervalMs, err := resolveTTLSweepMs(*ttlSweepInterval)
	if err != nil {
		fatal(err.Error())
	}

	// Operator action: trigger an online rebalance and exit. -peers is the target
	// membership (grow: include new nodes; decommission: omit departing nodes);
	// -replication-factor is the target RF. Connects to the peers' server addrs.
	if *reconfigure {
		runReconfigure(*peers, *nodeID, *raftAddr, *tcpAddr, *httpAddr, *replicationFactor)
		return
	}

	// Secret resolution (env preferred over flag). A secret passed on the command
	// line is world-readable via /proc/<pid>/cmdline and lands in shell history, so
	// the env var is the recommended channel; the flag is kept only for backward
	// compatibility and emits a one-line warning when used. Env wins when both are
	// set (an operator migrating to the env var should not be silently overridden by
	// a stale flag in a unit file).
	*apiKey = resolveSecret("-api-key", "ROSTAM_API_KEY", *apiKey)
	*internalToken = resolveSecret("-internal-token", "ROSTAM_INTERNAL_TOKEN", *internalToken)

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		fatal("register ops failed", "err", err)
	}

	// OPT-IN per-node mTLS identity allowlist (CSV -> set). Empty/unset = nil = OFF
	// (byte-identical). Threaded into BOTH the authorizer (the server-side
	// internal-token CN gate) and the cluster config (the inter-node client-side
	// peer-CN verify). Startup validation (set-without-inter-node-client-cert-TLS
	// -> fatal) runs in the -cluster branch where the inter-node TLS is built.
	nodeAllowlist := parseCNAllowlist(*nodeCNAllowlist)

	// Warn about the TLS fallback ONLY when -epoll was passed explicitly. Now that
	// it defaults on, warning unconditionally would nag every TLS operator about a
	// flag they never set, for a fallback that is automatic and correct.
	epollExplicit := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "epoll" {
			epollExplicit = true
		}
	})
	if *epollTCP && *tlsCert != "" && epollExplicit {
		slog.Warn("-epoll is plaintext-only; TLS is configured, so the TCP transport uses the goroutine server (epoll disabled for this run)")
	}
	// Cluster mode: the epoll transport executes dispatch INLINE on the event
	// loop, so a replicated write (PB full-ISR ack wait, Raft apply wait) parks
	// the whole loop for a replication round trip — every other connection on
	// that loop stalls, capping write throughput at ~(loops / RTT) no matter how
	// many clients connect. Measured on a real-network 3-node PB RF=2 cluster:
	// 61k ops/s with epoll vs 127k with the goroutine server @128 conns (see
	// shard/pbisr/BENCHMARK.md, "epoll inline-dispatch ceiling"). So in -cluster
	// mode the effective default is the goroutine server; an explicit -epoll
	// still forces the event-loop transport for read-dominated cluster workloads.
	if *cluster && !epollExplicit {
		*epollTCP = false
	}
	cfg := rostam.ServerConfig{HTTPAddr: *httpAddr, GRPCAddr: *grpcAddr, TCPAddr: *tcpAddr, EpollTCP: *epollTCP, EpollLoops: *epollLoops, AccessLog: accessLog, EnableOnlineCompaction: *onlineCompaction}
	// Authenticator precedence (fail-closed): -keys-file > -api-key > nil (open).
	//   -keys-file: granular per-collection RBAC. A keys file that does not exist
	//               or fails to parse is FATAL — we never silently start open.
	//   -api-key:   single static superuser key. Adapted to the new RBAC signature
	//               by granting the matching token the "*:*" (superuser) scope, so
	//               it behaves exactly like the old token==apiKey gate.
	//   neither:    nil authenticator = open/dev mode (unchanged).
	// For clusters running with auth, set -internal-token so inter-node forwards
	// carry a trusted superuser identity (peerClient presents it).
	switch {
	case *keysFile != "":
		keyReg, err := vector.OpenKeyRegistry(*keysFile)
		if err != nil {
			fatal("opening -keys-file failed", "keys_file", *keysFile, "err", err)
		}
		// Opt-in RBAC knobs (both default OFF = byte/behaviour-identical to the
		// scope-only engine):
		//   -audit-log:        a non-nil AuditLogger makes the authorizer emit one
		//                      redacted JSON audit record per decision.
		//   -tenant-isolation: turns APIKey.Tenant into an authoritative resource
		//                      boundary (a scope cannot escape its tenant); a key
		//                      with Tenant="*" is the cross-tenant/admin exemption.
		opts := authz.RBACOptions{TenantIsolation: *tenantIsolation, NodeCNAllowlist: nodeAllowlist}
		// Tenant isolation is opt-in / default-OFF (byte-compatible default). With it
		// off, a key's APIKey.Tenant is documentation-only: a broad glob scope
		// (read:*, write:*, *:*) reads/writes EVERY tenant's collections, not just the
		// key's own. A key carrying a real (non-"*") Tenant while isolation is OFF is
		// almost certainly a misconfiguration that silently breaches tenant
		// boundaries, so REFUSE to start (fail-closed) rather than warn — the safe
		// production stance. A single-tenant deployment has no tenant-scoped keys and
		// never trips this; an operator who genuinely wants cross-tenant keys marks
		// them Tenant="*" (the explicit cross-tenant marker, exempt here).
		if !*tenantIsolation && keyRegistryHasTenantScopedKey(keyReg) {
			fatal("-keys-file has tenant-scoped keys but -tenant-isolation is OFF — a glob scope (read:*/write:*/*:*) would CROSS tenant boundaries and is NOT confined to the key's Tenant. Set -tenant-isolation to enforce APIKey.Tenant as a boundary, or mark intentionally cross-tenant keys Tenant=\"*\".")
		}
		if *auditLog {
			opts.Audit = authz.NewJSONStderrAuditLogger()
		}
		// Opt-in JWT acceptance (HTTP/gRPC only). -jwt-public-key is the trigger:
		// read the PEM, build the alg-pinned verifier (alg inferred from key type),
		// and wire it. A bad key config is FATAL — we fail startup loudly rather
		// than silently disable a requested security feature. Unset = nil verifier
		// = JWT-off.
		if *jwtPublicKey != "" {
			pemBytes, err := os.ReadFile(*jwtPublicKey)
			if err != nil {
				fatal("-jwt-public-key error", "path", *jwtPublicKey, "err", err)
			}
			v, err := authz.NewJWTVerifier(pemBytes, *jwtIssuer, *jwtAudience)
			if err != nil {
				fatal("-jwt-public-key error", "path", *jwtPublicKey, "err", err)
			}
			opts.JWTVerifier = v
		}
		cfg.Authenticator = authz.NewRBACAuthenticatorOpts(keyReg, reg, *internalToken, opts)
		// Wire the SAME registry into the dispatcher so the online key-admin ops
		// (__keys_add__/__keys_revoke__/__keys_list__) mutate/list the registry the
		// authenticator reads: an add/revoke takes effect immediately (no restart)
		// and persists via the registry's atomic keys-file flush. With -api-key or
		// open mode this stays nil and the keys ops fail loud (ErrKeyAdminUnavailable).
		cfg.KeyRegistry = keyReg
	case *apiKey != "":
		cfg.Authenticator = staticKeyAuthenticator(*apiKey)
	}

	// Fail-closed on an OPEN + EXPOSED bind. With no authenticator (-keys-file and
	// -api-key both unset) EVERY request is accepted, so binding a non-loopback
	// interface would silently expose an unauthenticated datastore to the whole
	// network — the single worst production footgun. Refuse unless the operator
	// explicitly passes -insecure. Loopback-only open binds (the dev default,
	// ":8080" excluded — see exposedBind) are the one thing we do NOT want to break,
	// so only a genuinely reachable address trips this.
	if cfg.Authenticator == nil && !*insecure {
		for _, a := range []string{*httpAddr, *grpcAddr, *tcpAddr} {
			if exposedBind(a) {
				fatal("refusing to bind non-loopback address with NO authentication — every request would be served unauthenticated. Set -keys-file, -api-key, or the ROSTAM_API_KEY environment variable to require auth, bind a loopback address (127.0.0.1/localhost), or pass -insecure to run open deliberately (dev/trusted-network only)", "addr", a)
			}
		}
	}

	// TLS wiring (fail-closed). When -tls-cert OR -tls-key is set, build the
	// server *tls.Config and apply it to all three client-facing transports.
	// tlsutil.ServerTLS is the single fail-closed gate: cert-without-key ⇒ error;
	// require-client-cert-without-ca ⇒ error; an unreadable/malformed cert/key/ca
	// ⇒ error. Any such error is FATAL — we never silently start plaintext when
	// TLS was requested. No -tls-* flags ⇒ TLSConfig stays nil ⇒ plaintext (the
	// dev default; production should set TLS, ideally with -keys-file for full
	// secure mode: TLS + per-collection RBAC).
	if *tlsCert != "" || *tlsKey != "" {
		tlsCfg, err := buildServerTLS(*tlsCert, *tlsKey, *tlsCA, *tlsRequireClientCert)
		if err != nil {
			fatal("TLS config error", "err", err)
		}
		cfg.TLSConfig = tlsCfg
	} else if *tlsCA != "" || *tlsRequireClientCert {
		// Fail closed: a TLS-related flag is present but the cert+key pair that
		// actually enables TLS is missing. Refusing to start prevents a silent
		// plaintext server when the operator believes TLS is active (e.g. a
		// fat-fingered -tls-cert that resolved to an empty env var).
		fatal("-tls-ca / -tls-require-client-cert set but -tls-cert/-tls-key missing; refusing to start plaintext with TLS flags present")
	}

	if *cluster {
		// Fail loud (matching the TLS/allowlist validations below): a cluster with an
		// authenticator configured (-keys-file or -api-key) but no -internal-token
		// would have every inter-node forwarded op carry no token and be denied
		// superuser by the destination authorizer — the cluster is then silently
		// non-functional for any op that must forward to the leader/owner. Refuse to
		// start rather than emit confusing runtime denials.
		if (*keysFile != "" || *apiKey != "") && *internalToken == "" {
			fatal("-cluster with authentication (-keys-file/-api-key) requires -internal-token (or the ROSTAM_INTERNAL_TOKEN env var) so inter-node forwarded ops carry a trusted superuser identity; without it every forwarded op is denied")
		}
		selfServer := *tcpAddr
		if selfServer == "" {
			selfServer = *httpAddr
		}
		cfg.Cluster = &rostam.EmbeddedConfig{
			NodeID:            *nodeID,
			DataDir:           *data,
			NumShards:         *shards,
			ReplicationFactor: *replicationFactor,
			PersistentVectors: *persistentVectors,

			PBFrontierStampInterval: *pbFrontierStampInterval,
			RaftAddr:                *raftAddr,
			RaftTransport:           *raftTransport,
			Bootstrap:               *bootstrap,
			Ops:                     reg,
			Peers:                   parsePeers(*peers, *nodeID, *raftAddr, selfServer, *pbAddr),
			InternalToken:           *internalToken,
			NodeCNAllowlist:         nodeAllowlist,
			Cache: rostam.CacheConfig{
				MaxMemoryBytes:        cacheMaxMemory,
				DisableColdCompaction: *disableColdCompaction,
				TTLSweepIntervalMs:    ttlSweepIntervalMs,
			},
			NoSync:            *noSync,
			VolatileLog:       *volatileLog,
			ReplicationMode:   *replicationMode,
			MinISR:            *minISR,
			PBCommitPrimary:   *pbCommitPrimary,
			PBAutoFailover:    *pbAutoFailover,
			WASMBlobRetention: *wasmBlobRetention,
		}
		// Inter-node TLS dial (fail-closed). When client TLS is enabled (-tls-cert)
		// AND this is a cluster node, the client-facing ports (incl. the TCP port
		// that inter-node forwarding dials) are TLS-wrapped, so the inter-node dial
		// must be TLS too or forwarded ops EOF at the peer's TLS handshake. Build a
		// client config that verifies each peer's server cert against -tls-ca and
		// presents this node's own cert/key as the client cert (for peer mTLS); the
		// per-peer ServerName is set later by peerClient, so pass "" here. AUTH stays
		// the internal token; this only provides the encrypted inter-node transport.
		// Any build error is FATAL — never silently dial inter-node plaintext when
		// the cluster's ports are TLS-wrapped. No -tls-cert ⇒ InterNodeTLS stays nil
		// ⇒ plaintext inter-node dial (unchanged).
		if *tlsCert != "" {
			// The cert this node presents to peers: a dedicated node cert (with
			// clientAuth EKU, required for strict peer mTLS) if given, else the server
			// cert. tlsutil.ClientTLS errors loudly on a bad cert/key/ca.
			nodeCert, nodeKey := *tlsNodeCert, *tlsNodeKey
			if nodeCert == "" {
				nodeCert, nodeKey = *tlsCert, *tlsKey
			}
			// -tls-ca is MANDATORY here. The inter-node REPLICATION plane (Raft
			// mux/fabric + PB) authenticates peers ONLY by CA-verified client certs —
			// it has no internal-token backstop like the client-facing forwarding
			// plane does. Without a cluster CA the replication listener would still
			// TLS-wrap but request NO client cert (ClientAuth stays NoClientCert), so
			// it would serve replicated writes — full AppendEntries/replicate frame
			// forgery for every tenant — to any peer that merely completes a TLS
			// handshake. That is an auth bypass, not a "degradation," so fail closed
			// rather than silently serve unauthenticated replication. (A missing CA
			// would ALSO leave the client dial verifying peers against the platform
			// root store instead of a pinned cluster CA — a second reason to require it.)
			if *tlsCA == "" {
				fatal("-cluster -tls-cert requires -tls-ca: the inter-node replication plane authenticates peers only by CA-verified client certs (no token fallback); without a cluster CA it would serve replicated writes to any unauthenticated TLS peer")
			}
			interNodeTLS, err := tlsutil.ClientTLS(*tlsCA, nodeCert, nodeKey, "")
			if err != nil {
				fatal("inter-node TLS config error", "err", err)
			}
			cfg.InterNodeTLS = interNodeTLS

			// SERVER side of the same trust boundary: wrap the inter-node REPLICATION
			// listeners (Raft mux/fabric + PB) with mTLS. Without this the replication
			// ports stay plaintext even with -tls-cert set, so anyone reaching them
			// reads/forges every tenant's replicated writes (the hole this closes).
			// requireClientCert is unconditionally true: -tls-ca is guaranteed present
			// (the fatal above), so we always demand and CA-verify a peer client cert
			// (strict mTLS, so the CN allowlist is enforceable server-side). Any build
			// error is FATAL: never serve replication plaintext when TLS was requested.
			interNodeServerTLS, err := tlsutil.ServerTLS(nodeCert, nodeKey, *tlsCA, true)
			if err != nil {
				fatal("inter-node server TLS config error", "err", err)
			}
			cfg.InterNodeServerTLS = interNodeServerTLS
		}
		// Startup validation (fail-loud): -node-cn-allowlist set WITHOUT inter-node
		// client-cert TLS means there is no peer cert CN to verify on the dial —
		// the client-side peer-CN verify would have no chain and the server-side
		// gate would deny every internal-token forward (no ClientCN). Refuse to
		// start a misconfig that would reject all peers. cfg.InterNodeTLS is nil
		// here exactly when -tls-cert was not set (the only path that builds it).
		if len(nodeAllowlist) > 0 && cfg.InterNodeTLS == nil {
			fatal("-node-cn-allowlist set but inter-node client-cert TLS is not configured (set -tls-cert/-tls-key, and -tls-node-cert with clientAuth EKU for strict peer mTLS); refusing to start a config that would reject all peers")
		}
	} else {
		// -node-cn-allowlist is a cluster-only inter-node identity gate; in
		// single-node mode there are no peers to verify. Fail loud rather than
		// silently ignore a requested security flag.
		if len(nodeAllowlist) > 0 {
			fatal("-node-cn-allowlist requires -cluster (it gates inter-node peer identity; single-node has no peers)")
		}
		cfg.DirectConfig = rostam.DirectConfig{
			DataDir: *data,
			Ops:     reg,
			Cache: rostam.CacheConfig{
				NumShardsPerNode:      *shards,
				MaxMemoryBytes:        cacheMaxMemory,
				DisableColdCompaction: *disableColdCompaction,
				TTLSweepIntervalMs:    ttlSweepIntervalMs,
			},
			// Preserve the authenticator chosen above: it is promoted from the embedded
			// DirectConfig, so replacing the struct wholesale would otherwise zero it and
			// run single-node UNAUTHENTICATED. (RHS reads the value set in the precedence
			// block before this assignment takes effect.)
			Authenticator: cfg.Authenticator,
		}

		// Register a single-node __topology__ so the discovery op resolves WITHOUT
		// -cluster: the dashboard's cluster view then sees one node owning its one
		// logical shard, instead of the op erroring as "not registered". In -cluster
		// mode cluster.Node registers the real, live topology on this same registry;
		// this branch never builds a cluster.Node, so there is no double-register.
		//
		// ServerAddr and Leaders are deliberately EMPTY. A smart client caches this
		// topology and, in pickInitialTarget, would DIAL a non-empty Leaders[shard]
		// / owner ServerAddr — but the server's bind address (e.g. ":7000",
		// "0.0.0.0:7000", or the httpAddr fallback) is not a dialable target and is
		// not a reliable advertised address. Leaving them empty makes OwnerAddr
		// return "" so the client falls back to its own configured server target —
		// byte-identical to the pre-registration behavior (where the op was absent
		// and the topology cache stayed nil) — while still giving the dashboard the
		// node id + placement it renders. (A multi-node deployment uses -cluster,
		// which advertises real peer addresses.)
		selfID := *nodeID
		if err := ops.RegisterTopology(reg, func() (ops.Topology, error) {
			return ops.Topology{
				NumShards: 1,
				Members:   []ops.TopologyMember{{NodeID: selfID, ServerAddr: ""}},
				Leaders:   []string{""},
				Placement: [][]string{{selfID}},
			}, nil
		}); err != nil {
			fatal("register single-node topology failed", "err", err)
		}
	}

	// OPT-IN S3 tier (backup cron + cold-tier sweeper + admin REST surface). Build
	// and VALIDATE the plan BEFORE starting the server so a misconfig fails loud at
	// startup (mirrors the TLS/RBAC validation above), never a silent degrade.
	// Nothing configured ⇒ tierPlan is nil ⇒ no objstore, no goroutine, no admin
	// backend (the admin routes then 412). The admin backend is attached to
	// cfg.Admin now (so NewServer wires it into the HTTP handler) but its store is
	// set after NewServer returns.
	tier, err := buildTierPlan(tierFlags{
		Bucket:        *backupBucket,
		Endpoint:      *backupEndpoint,
		Region:        *backupRegion,
		Interval:      *backupInterval,
		Retention:     *backupRetention,
		Tenant:        *backupTenant,
		ColdTierAfter: *coldTierAfter,
		PathStyle:     *s3PathStyle,
	})
	if err != nil {
		fatal("S3 tier config error", "err", err)
	}
	admin := newAdminBackend(tier)
	if admin != nil {
		cfg.Admin = admin
	}

	srv, err := rostam.NewServer(cfg)
	if err != nil {
		fatal("server start failed", "err", err)
	}
	defer func() { _ = srv.Close() }()

	// Attach the live store to the S3 tier (backup cron, cold-tier sweeper, admin
	// surface) and, when cold tiering is on, inject the wall clock so the engine
	// stamps per-collection last-access on resolve (the engine never calls
	// time.Now itself). The S3 tier requires a directly-reachable single-node store
	// — cluster vectors are partitioned across Raft shards (Raft is the durability
	// authority), so backing them up from here would silently snapshot nothing;
	// fail loud instead so a requested S3 backup/cold-tier is never quietly a no-op.
	var tierVS *vector.CollectionStore
	tierNode := srv.ClusterNode()
	if tier != nil {
		tierVS = srv.VectorStore()
		if tierVS == nil && tierNode == nil {
			fatal("-backup-bucket/-cold-tier-after set but this backend has neither a single-node vector store nor a cluster node")
		}
		if tierVS != nil {
			if admin != nil {
				admin.SetStore(tierVS)
			}
			if tier.ColdTierAfter > 0 {
				tierVS.SetClock(func() time.Time { return time.Now() })
			}
		} else if tier.ColdTierAfter > 0 {
			// Cluster mode: cold tiering is a single-node vector-store feature
			// (per-collection idle eviction). Cluster vectors are partitioned across
			// shards, so there is no single store to sweep — fail loud rather than
			// silently no-op. S3 BACKUP, by contrast, IS supported in cluster mode
			// (per-shard, below).
			fatal("-cold-tier-after is not supported in -cluster mode (vectors are partitioned across shards; use -backup-bucket for per-shard cluster backup)")
		}
	}
	mode := "single-node"
	if *cluster {
		mode = "cluster node " + *nodeID
	}
	for proto, addr := range map[string]string{"http": srv.HTTPAddr(), "grpc": srv.GRPCAddr(), "tcp": srv.TCPAddr()} {
		if addr != "" {
			slog.Info("serving", "proto", proto, "addr", addr, "mode", mode, "data", *data)
		}
	}

	// One-shot disaster-recovery restore (-restore). Runs AFTER bring-up (the
	// cluster has elected leaders) and BEFORE any backup loop starts, so a restore
	// is never raced by a fresh backup. Each node restores the shards it owns; the
	// leader of each shard group installs the snapshot and replicates it to
	// followers (Raft) or every owner installs directly (PB), and the MetaRaft
	// leader restores the catalog. Same-topology only.
	if *restore {
		node := srv.ClusterNode()
		if node == nil {
			fatal("-restore requires -cluster")
		}
		var robj objstore.ObjectStore
		var rtenant string
		switch {
		case *backupDir != "":
			o, err := backup.NewFSObjectStore(*backupDir)
			if err != nil {
				fatal("restore store error", "err", err)
			}
			robj, rtenant = o, *backupPrefix
		case tier != nil:
			robj, rtenant = tier.Store, tier.Tenant
		default:
			fatal("-restore requires -backup-dir or -backup-bucket")
		}
		rctx, cancelRestore := context.WithTimeout(context.Background(), 3*time.Minute)
		if err := node.RestoreFromBackup(rctx, robj, rtenant, *allowMissingShards); err != nil {
			cancelRestore()
			fatal("restore failed", "err", err)
		}
		cancelRestore()
		slog.Info("restore complete; continuing to serve", "tenant", rtenant)
	}

	// Flag-gated background backup driver. OFF by default (-backup-dir empty OR
	// -backup-interval 0) ⇒ byte-identical to the no-backup path. When ON it
	// requires a directly-reachable single-node store: cluster vectors are
	// partitioned across Raft shards (Raft is the durability authority), so
	// backing them up from here would silently snapshot nothing — fail loud
	// instead so a requested backup is never quietly a no-op.
	// Backup needs BOTH -backup-dir and -backup-interval; if exactly one is set
	// the operator likely meant to enable it — warn rather than silently disable.
	if (*backupDir != "") != (*backupInterval > 0) {
		slog.Warn("backup disabled — it requires BOTH -backup-dir and -backup-interval>0", "dir", *backupDir, "interval", *backupInterval)
	}
	backupCtx, stopBackup := context.WithCancel(context.Background())
	defer stopBackup()
	backupDone := make(chan struct{})
	close(backupDone) // default: nothing to wait for unless the loop starts below
	if *backupDir != "" && *backupInterval > 0 {
		obj, err := backup.NewFSObjectStore(*backupDir)
		if err != nil {
			fatal("backup store error", "err", err)
		}
		if node := srv.ClusterNode(); node != nil {
			// Cluster mode: per-node/per-shard point-in-time backup (cache + vectors
			// via the shard FSM snapshot) + the MetaRaft catalog. This REPLACES the
			// former fatal "cluster backup is a follow-up" halt — a cluster IS backed
			// up now, just per shard rather than through the single-node vector store.
			slog.Info("cluster backup ON", "dir", *backupDir, "interval", *backupInterval, "tenant", *backupPrefix, "retention", *backupRetention)
			backupDone = make(chan struct{})
			go runClusterBackupLoop(backupCtx, node, obj, *backupPrefix, *backupRetention, *backupInterval, backupDone)
		} else {
			vs := srv.VectorStore()
			if vs == nil {
				fatal("-backup-dir set but this backend has neither a single-node vector store nor a cluster node")
			}
			opts := backup.BackupOpts{Tenant: *backupPrefix, Retention: *backupRetention}
			slog.Info("backup ON", "dir", *backupDir, "interval", *backupInterval, "prefix", *backupPrefix, "retention", *backupRetention)
			backupDone = make(chan struct{})
			go runBackupLoop(backupCtx, vs, obj, opts, *backupInterval, backupDone)
		}
	}

	// OPT-IN S3 driver goroutines (share backupCtx so a single shutdown cancels
	// every loop). The S3 backup cron and the cold-tier sweeper each only start
	// when their flag is set; both supply the wall clock HERE (the ONLY place
	// time.Now is read for these paths). tierVS is the validated single-node store.
	s3BackupDone := make(chan struct{})
	close(s3BackupDone)
	coldSweepDone := make(chan struct{})
	close(coldSweepDone)
	if tier != nil {
		if tier.BackupInterval > 0 {
			s3BackupDone = make(chan struct{})
			if tierNode != nil {
				slog.Info("cluster S3 backup ON", "bucket", *backupBucket, "interval", tier.BackupInterval, "tenant", tier.Tenant, "retention", tier.Retention)
				go runClusterBackupLoop(backupCtx, tierNode, tier.Store, tier.Tenant, tier.Retention, tier.BackupInterval, s3BackupDone)
			} else {
				slog.Info("S3 backup ON", "bucket", *backupBucket, "interval", tier.BackupInterval, "tenant", tier.Tenant, "retention", tier.Retention)
				go runS3BackupLoop(backupCtx, tierVS, tier, tier.BackupInterval, s3BackupDone)
			}
		}
		if tier.ColdTierAfter > 0 {
			slog.Info("cold tiering ON", "bucket", *backupBucket, "evict_after", tier.ColdTierAfter, "tenant", tier.Tenant)
			coldSweepDone = make(chan struct{})
			go runColdTierSweeper(backupCtx, tierVS, tier, coldSweepDone)
		}
	}

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	slog.Info("shutting down")
	// Cancel every background loop and wait for each in-flight run to unwind.
	stopBackup()
	<-backupDone
	<-s3BackupDone
	<-coldSweepDone
}

// runBackupLoop ticks every interval and snapshots every collection to obj via
// backup.Backup. The per-run Timestamp is computed HERE with time.Now() (the
// backup package itself never calls time.Now, staying deterministic/testable) as
// a sortable UTC RFC3339 stamp so retention prunes oldest-first. A backup
// failure is LOGGED and the loop continues — a transient object-store error must
// never crash the serving process. The loop exits when ctx is cancelled (server
// shutdown), passing ctx into each run so an in-flight backup unwinds promptly,
// and closes done so main can wait for a clean stop.
func runBackupLoop(ctx context.Context, vs *vector.CollectionStore, obj objstore.ObjectStore, opts backup.BackupOpts, interval time.Duration, done chan<- struct{}) {
	defer close(done)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			runOpts := opts
			runOpts.Timestamp = time.Now().UTC()
			results, err := backup.Backup(ctx, vs, obj, runOpts)
			if err != nil {
				slog.Warn("backup run had failures (continuing)", "err", err)
			}
			ok := 0
			for _, r := range results {
				if r.Err == nil {
					ok++
				}
			}
			slog.Info("backup run", "ok", ok, "total", len(results), "ts", runOpts.Timestamp.Format(time.RFC3339))
		}
	}
}

// runClusterBackupLoop ticks every interval and drives a per-node/per-shard
// cluster backup (production-readiness #5): each owned shard this node LEADS is
// snapshotted (cache + vectors via the shard FSM snapshot, strictly more complete
// than the single-node vector-only path) under <tenant>/node-<id>/shard-NNNN/, and
// the MetaRaft leader also writes the catalog under <tenant>/meta/. It reuses the
// same objstore.ObjectStore (FS or S3) and a generalized retention/prune over the
// .shard/.meta key layout. Per-run failures are LOGGED and the loop continues — a
// transient object-store error must never crash the serving node. The loop exits
// when ctx is cancelled and closes done so main can wait for a clean stop.
func runClusterBackupLoop(ctx context.Context, node *cluster.Node, obj objstore.ObjectStore, tenant string, retention int, interval time.Duration, done chan<- struct{}) {
	defer close(done)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			results, err := node.BackupOwnedShards(ctx, obj, tenant, retention)
			if err != nil {
				slog.Warn("cluster backup run had failures (continuing)", "err", err)
			}
			s := cluster.SummarizeBackupResults(results)
			// A HOSTED shard with no leader/primary this cycle is UNCOVERED — no node
			// backs it up. Surface it loudly so a chronically un-backed shard is visible
			// BEFORE a restore needs it (completeness accounting, #5 M2).
			if s.Uncovered > 0 {
				slog.Warn("cluster backup: hosted shard(s) had NO leader/primary — NOT backed up this cycle", "uncovered", s.Uncovered, "hosted", s.Hosted)
			}
			metaKey, merr := node.BackupMetaCatalog(ctx, obj, tenant, retention)
			if merr != nil {
				slog.Warn("cluster meta backup failed (continuing)", "err", merr)
			}
			metaNote := "meta: not leader (skipped)"
			if metaKey != "" {
				metaNote = "meta: " + metaKey
			}
			slog.Info("cluster backup run",
				"backed", s.Backed, "hosted", s.Hosted, "empty", s.Empty, "no_leader", s.Uncovered, "failed", s.Failed, "meta", metaNote)
		}
	}
}

// staticKeyAuthenticator adapts the legacy single-static-key mode to the unified
// authz.Authenticator signature: the matching token is a superuser (granted
// every action on every resource — the old token==apiKey gate allowed any op),
// and any other (or empty) token is denied. An empty apiKey would match an empty
// token, so the caller must only build this when apiKey != "" (the -api-key flag
// branch guarantees that). No KeyRegistry is needed for a single static key.
// exposedBind reports whether a listen address is reachable beyond loopback, so
// an open (unauthenticated) server on it would face the network. "" (transport
// disabled) and loopback binds return false; a bare ":port" / "0.0.0.0" / "::"
// (all interfaces), any non-loopback IP, and any non-"localhost" hostname (which
// could resolve anywhere) return true. Unparseable addresses fail safe as exposed.
func exposedBind(addr string) bool {
	if addr == "" {
		return false // transport disabled
	}
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return true // unparseable: be conservative
	}
	if host == "" {
		return true // no host = bind all interfaces (0.0.0.0 / ::)
	}
	if host == "localhost" {
		return false
	}
	if ip := net.ParseIP(host); ip != nil {
		return !ip.IsLoopback() // 0.0.0.0, ::, and real IPs are exposed; 127.x/::1 are not
	}
	return true // a non-localhost hostname could resolve to a public address
}

func staticKeyAuthenticator(apiKey string) authz.Authenticator {
	return func(req authz.AuthRequest) bool {
		// Constant-time compare: a plain == is a byte-by-byte timing oracle on
		// the *:* superuser key (matching how the internal token is compared).
		return req.Token != "" && authz.SecureTokenEqual(req.Token, apiKey)
	}
}

// buildServerTLS is the testable flag→*tls.Config layer for the server's
// client-facing transports. It is a thin pass-through to tlsutil.ServerTLS (the
// single fail-closed assembler) so main() can stay free of crypto logic and the
// flag→config wiring can be unit-tested WITHOUT calling log.Fatalf. It returns
// (nil, err) on any misconfiguration — cert-without-key, require-client-cert
// without a CA, or an unreadable/malformed file — which main() surfaces as a
// FATAL error (never a silent plaintext fallback).
func buildServerTLS(certFile, keyFile, caFile string, requireClientCert bool) (*tls.Config, error) {
	return tlsutil.ServerTLS(certFile, keyFile, caFile, requireClientCert)
}

// runReconfigure triggers an online rebalance to the target membership (-peers)
// and replication factor against a running cluster, then exits. It connects to
// the target peers' server addresses and blocks until the rebalance completes.
func runReconfigure(peersSpec, selfID, selfRaft, tcpAddr, httpAddr string, rf int) {
	selfServer := tcpAddr
	if selfServer == "" {
		selfServer = httpAddr
	}
	target := parsePeers(peersSpec, selfID, selfRaft, selfServer, "")
	addrs := make([]string, 0, len(target))
	for _, p := range target {
		if p.ServerAddr != "" {
			addrs = append(addrs, p.ServerAddr)
		}
	}
	if len(addrs) == 0 {
		fatal("-reconfigure needs target peers with server addresses (see -peers)")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	res, err := rostam.Reconfigure(ctx, addrs, target, rf)
	if err != nil {
		fatal("reconfigure failed", "err", err)
	}
	slog.Info("reconfigure complete", "moves", res.Moves, "done", res.Done, "failed", res.Failed)
}

// keyRegistryHasTenantScopedKey reports whether the registry holds at least one
// key with a non-empty, non-"*" Tenant — i.e. a key whose operator evidently
// intended a tenant boundary. Used to decide whether the tenant-isolation-OFF
// startup warning is relevant (a registry of only cross-tenant "*" keys, or
// tenant-less keys, has nothing to confine, so no warning is emitted).
func keyRegistryHasTenantScopedKey(reg *vector.KeyRegistry) bool {
	if reg == nil {
		return false
	}
	for _, k := range reg.ListKeys() {
		if k.Tenant != "" && k.Tenant != "*" {
			return true
		}
	}
	return false
}

// resolveSecret returns the secret to use for a flag/env pair, preferring the
// environment variable over the command-line flag. A flag-passed secret is
// visible to other local users via /proc/<pid>/cmdline and lands in shell history
// and process-supervisor logs, so the env var is the recommended channel; when the
// flag is used (and the env var is not), a one-line warning is emitted. The env
// value WINS when both are set. The secret itself is NEVER logged.
func resolveSecret(flagName, envName, flagVal string) string {
	if envVal := os.Getenv(envName); envVal != "" {
		return envVal
	}
	if flagVal != "" {
		slog.Warn("secret passed on the command line is visible to other local users via /proc and shell history; prefer the environment variable", "flag", flagName, "env", envName)
	}
	return flagVal
}

// parseCNAllowlist parses the -node-cn-allowlist CSV into a set of trusted peer
// cert CommonNames. Whitespace around each CN is trimmed and empty entries are
// dropped. An empty/whitespace-only spec returns nil (OFF = byte-identical: the
// caller treats len==0 as "no allowlist"); a non-empty spec with at least one
// real CN returns a populated map. Returning nil (not an empty non-nil map) for
// the off case keeps the byte-identical contract crisp at every consumer.
func parseCNAllowlist(spec string) map[string]bool {
	if strings.TrimSpace(spec) == "" {
		return nil
	}
	out := make(map[string]bool)
	for _, raw := range strings.Split(spec, ",") {
		cn := strings.TrimSpace(raw)
		if cn != "" {
			out[cn] = true
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// parsePeers parses "id@raftAddr@serverAddr" (optionally "id@raftAddr@serverAddr@pbAddr"
// — the 4th field is this peer's pbisr.NetTransport listen address, needed only for
// -replication-mode=pb) comma-separated peer specs. An empty spec yields a single
// self-peer (single-node cluster), using selfPBAddr (from -pb-addr) as that peer's
// PBAddr (empty in raft mode, byte-identical to before this field existed).
func parsePeers(spec, selfID, selfRaft, selfServer, selfPBAddr string) []rostam.Peer {
	if strings.TrimSpace(spec) == "" {
		return []rostam.Peer{{NodeID: selfID, RaftAddr: selfRaft, ServerAddr: selfServer, PBAddr: selfPBAddr}}
	}
	var peers []rostam.Peer
	for _, p := range strings.Split(spec, ",") {
		parts := strings.Split(strings.TrimSpace(p), "@")
		if len(parts) != 3 && len(parts) != 4 {
			fatal("bad -peers entry (want id@raftAddr@serverAddr or id@raftAddr@serverAddr@pbAddr)", "entry", p)
		}
		peer := rostam.Peer{NodeID: parts[0], RaftAddr: parts[1], ServerAddr: parts[2]}
		if len(parts) == 4 {
			peer.PBAddr = parts[3]
		}
		peers = append(peers, peer)
	}
	return peers
}
