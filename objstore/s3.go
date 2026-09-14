// SPDX-License-Identifier: Apache-2.0

package objstore

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Config configures an S3Store.
type Config struct {
	// Endpoint is the base URL of the S3 service, e.g.
	// "https://s3.us-east-1.amazonaws.com" or "http://127.0.0.1:9000" (MinIO).
	// If empty, defaults to https://s3.<Region>.amazonaws.com.
	Endpoint string
	// Region is the AWS region used for signing, e.g. "us-east-1".
	Region string
	// Bucket is the target bucket name.
	Bucket string
	// Creds are the AWS credentials. If AccessKeyID is empty they are read from
	// the AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN env vars.
	Creds Credentials
	// PathStyle selects path-style addressing (<endpoint>/<bucket>/<key>) when
	// true, vs virtual-host (<scheme>://<bucket>.<host>/<key>) when false.
	// Default true (MinIO / R2 / localstack compatibility).
	PathStyle bool
	// HTTPClient is the client used for all requests. Defaults to a client with
	// a sane timeout if nil.
	HTTPClient *http.Client
	// clock supplies the signing timestamp. Defaults to time.Now. Injected for
	// deterministic tests.
	clock func() time.Time
}

// S3Store is a minimal, stdlib-only S3-compatible ObjectStore. It signs every
// request with AWS Signature V4 and parses ListObjectsV2 XML by hand.
type S3Store struct {
	endpoint  *url.URL
	region    string
	bucket    string
	creds     Credentials
	pathStyle bool
	service   string
	client    *http.Client
	clock     func() time.Time

	// condState records whether the service enforces If-None-Match (condUnknown,
	// condSupported, condUnsupported); condMu serialises the one-time probe that
	// decides it. See verifyConditionalCreate.
	condState atomic.Int32
	condMu    sync.Mutex
}

const s3Service = "s3"

// NewS3Store builds an S3Store from cfg, applying defaults.
func NewS3Store(cfg Config) (*S3Store, error) {
	if cfg.Region == "" {
		return nil, fmt.Errorf("objstore: region is required")
	}
	if cfg.Bucket == "" {
		return nil, fmt.Errorf("objstore: bucket is required")
	}
	if !validBucketName(cfg.Bucket) {
		return nil, fmt.Errorf("objstore: invalid bucket name %q", cfg.Bucket)
	}

	endpoint := cfg.Endpoint
	if endpoint == "" {
		endpoint = "https://s3." + cfg.Region + ".amazonaws.com"
	}
	u, err := url.Parse(endpoint)
	if err != nil {
		return nil, fmt.Errorf("objstore: invalid endpoint %q: %w", endpoint, err)
	}
	if u.Host == "" {
		return nil, fmt.Errorf("objstore: endpoint %q has no host", endpoint)
	}

	creds := cfg.Creds
	if creds.AccessKeyID == "" {
		creds.AccessKeyID = os.Getenv("AWS_ACCESS_KEY_ID")
		creds.SecretAccessKey = os.Getenv("AWS_SECRET_ACCESS_KEY")
		creds.SessionToken = os.Getenv("AWS_SESSION_TOKEN")
	}
	// Both halves of a credential pair must be present. A non-empty AccessKeyID
	// with an empty SecretAccessKey (e.g. AWS_ACCESS_KEY_ID set but
	// AWS_SECRET_ACCESS_KEY missing) would derive a signing key over an empty
	// secret and produce a structurally valid but WRONG signature, failing at the
	// server with an opaque 403. Fail loud locally instead with a precise error.
	if (creds.AccessKeyID == "") != (creds.SecretAccessKey == "") {
		return nil, fmt.Errorf("objstore: both access key id and secret access key are required (got one without the other)")
	}

	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 5 * time.Minute}
	}

	clock := cfg.clock
	if clock == nil {
		clock = time.Now
	}

	return &S3Store{
		endpoint:  u,
		region:    cfg.Region,
		bucket:    cfg.Bucket,
		creds:     creds,
		pathStyle: cfg.PathStyle,
		service:   s3Service,
		client:    client,
		clock:     clock,
	}, nil
}

// validBucketName reports whether name conforms to the S3 bucket-name grammar:
// 3-63 chars; lowercase letters, digits, hyphens, and dots only; must start and
// end with a letter or digit; no consecutive dots; and not formatted as an IPv4
// address. Validating here stops a bucket value with dots/'@'/other host bytes
// from flowing unchecked into the virtual-host Host header (used for both signing
// and the connection target).
func validBucketName(name string) bool {
	if len(name) < 3 || len(name) > 63 {
		return false
	}
	// No leading/trailing dot or hyphen.
	if !isBucketAlnum(name[0]) || !isBucketAlnum(name[len(name)-1]) {
		return false
	}
	prevDot := false
	allDigitsOrDots := true
	for i := 0; i < len(name); i++ {
		c := name[i]
		switch {
		case c >= 'a' && c <= 'z':
			allDigitsOrDots = false
		case c >= '0' && c <= '9':
			// digit — could still be an IP component
		case c == '-':
			allDigitsOrDots = false
		case c == '.':
			if prevDot {
				return false // consecutive dots
			}
		default:
			return false // any other byte (uppercase, '@', '_', ':', ...) is invalid
		}
		prevDot = c == '.'
	}
	// Reject an IPv4-address-formatted name (only digits and dots, with dots) per
	// the S3 grammar.
	if allDigitsOrDots && strings.Contains(name, ".") {
		return false
	}
	return true
}

func isBucketAlnum(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
}

// objectURL builds the request URL for an object key (path-style or
// virtual-host) and returns the URL plus the Host header to use for signing.
func (s *S3Store) objectURL(key string) (*url.URL, string) {
	u := *s.endpoint // copy
	if s.pathStyle {
		u.Path = "/" + s.bucket
		if key != "" {
			u.Path += "/" + key
		}
		return &u, u.Host
	}
	// virtual-host: bucket becomes a subdomain of the endpoint host.
	host := s.bucket + "." + s.endpoint.Host
	u.Host = host
	u.Path = "/" + key
	return &u, host
}

// bucketURL builds the URL used for bucket-level operations (ListObjectsV2),
// returning the URL and the Host header for signing. rawQuery is set verbatim.
func (s *S3Store) bucketURL(rawQuery string) (*url.URL, string) {
	u := *s.endpoint
	if s.pathStyle {
		u.Path = "/" + s.bucket
		u.RawQuery = rawQuery
		return &u, u.Host
	}
	host := s.bucket + "." + s.endpoint.Host
	u.Host = host
	u.Path = "/"
	u.RawQuery = rawQuery
	return &u, host
}

// Put streams r (size bytes) to key with a SigV4-signed PUT.
//
// Body integrity: over HTTPS the body is protected by TLS, so we sign with
// UNSIGNED-PAYLOAD and stream r without buffering/hashing it. Over a plaintext
// http:// endpoint TLS gives no integrity, so an UNSIGNED-PAYLOAD signature would
// let an on-path attacker tamper with the (snapshot) body without invalidating the
// signature. In that case we bind the real SHA-256 payload hash into the signature
// instead: if r is seekable (the backup temp file is) we pre-hash in one pass and
// rewind; otherwise we buffer the body to hash it. The hash then covers the bytes
// actually sent, so a tampered body fails server-side signature verification.
func (s *S3Store) Put(ctx context.Context, key string, r io.Reader, size int64) error {
	resp, err := s.put(ctx, key, r, size, false)
	if err != nil {
		return err
	}
	defer drainClose(resp.Body)
	if resp.StatusCode/100 != 2 {
		return s3Error("PUT", key, resp)
	}
	return nil
}

// put sends the signed PUT for Put and PutIfAbsent and returns the response,
// whose body the caller must drainClose. ifAbsent adds If-None-Match: *; like
// every header present at signing time it is covered by the signature, so it
// cannot be stripped in transit without failing verification.
func (s *S3Store) put(ctx context.Context, key string, r io.Reader, size int64, ifAbsent bool) (*http.Response, error) {
	u, host := s.objectURL(key)

	payloadHash := unsignedPayload
	if s.endpoint.Scheme != "https" {
		hash, body, err := hashPayload(r, size)
		if err != nil {
			return nil, fmt.Errorf("objstore: hash payload for %q: %w", key, err)
		}
		payloadHash = hash
		r = body
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPut, u.String(), r)
	if err != nil {
		return nil, err
	}
	req.Host = host
	req.ContentLength = size
	req.Header.Set("Content-Length", strconv.FormatInt(size, 10))
	if ifAbsent {
		req.Header.Set("If-None-Match", "*")
	}

	signV4(req, payloadHash, s.creds, s.region, s.service, s.clock().UTC())

	return s.client.Do(req) //nolint:bodyclose // the caller drainCloses
}

// Conditional-create support states for S3Store.condState.
const (
	condUnknown int32 = iota
	condSupported
	condUnsupported
)

// condProbePrefix names the throwaway objects verifyConditionalCreate writes.
const condProbePrefix = ".rostam-conditional-create-probe-"

// PutIfAbsent streams r (size bytes) to key with a SigV4-signed PUT carrying
// If-None-Match: *, so the service itself refuses the write when key exists:
// 412 Precondition Failed is reported as ErrExists. S3 decides that atomically
// against concurrent writers.
//
// The danger is a service that does not implement the header and IGNORES it —
// older MinIO releases and some other S3-compatible stores answer 200 and
// overwrite, which would make this call claim a guarantee it does not deliver.
// A response cannot tell the two apart, so before the first conditional write
// the store verifies the behaviour once (see verifyConditionalCreate) and, on a
// service that ignores or rejects the header, refuses every PutIfAbsent with
// ErrConditionalWriteUnsupported without writing the key. It never degrades to
// a plain overwrite.
//
// A 409 (S3's ConditionalRequestConflict, answered when conditional writes to
// one key collide in flight) is returned as a plain error, not ErrExists: it
// does not establish that another writer's object was stored.
func (s *S3Store) PutIfAbsent(ctx context.Context, key string, r io.Reader, size int64) error {
	if err := s.verifyConditionalCreate(ctx, key); err != nil {
		return err
	}
	return s.putIfAbsent(ctx, key, r, size)
}

// putIfAbsent is PutIfAbsent's request without the capability check.
func (s *S3Store) putIfAbsent(ctx context.Context, key string, r io.Reader, size int64) error {
	resp, err := s.put(ctx, key, r, size, true)
	if err != nil {
		return err
	}
	defer drainClose(resp.Body)
	switch {
	case resp.StatusCode/100 == 2:
		return nil
	case resp.StatusCode == http.StatusPreconditionFailed:
		return fmt.Errorf("objstore: PUT %q: %w", key, ErrExists)
	case resp.StatusCode == http.StatusNotImplemented:
		return fmt.Errorf("%w: %w", ErrConditionalWriteUnsupported, s3Error("PUT", key, resp))
	default:
		return s3Error("PUT", key, resp)
	}
}

// verifyConditionalCreate establishes, once per store, that the service
// enforces If-None-Match: * rather than ignoring it. It writes a probe object
// with a random name conditionally, then writes it conditionally AGAIN: a
// service that enforces the header refuses the second write with 412, one that
// ignores it accepts both. The probe is then deleted.
//
// The probe sits in the same "directory" as key, so credentials scoped to the
// backup prefix can write it. Its random name keeps concurrent probes — from
// other processes too — from colliding, so a 412 on the second write can only
// come from this probe's own first write.
//
// Only a definitive answer is remembered: enforced, or ignored/rejected (501).
// A transport error or any other status fails this call without deciding, so
// the next call probes again. The check proves that the service enforces the
// header, not that it does so atomically; a service that checks and then writes
// non-atomically is indistinguishable from outside.
func (s *S3Store) verifyConditionalCreate(ctx context.Context, key string) error {
	if s.condState.Load() != condUnknown {
		return s.conditionalCreateVerdict(key)
	}
	s.condMu.Lock()
	defer s.condMu.Unlock()
	if s.condState.Load() != condUnknown {
		return s.conditionalCreateVerdict(key) // decided while we waited
	}

	var nonce [8]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return fmt.Errorf("objstore: conditional-create probe name: %w", err)
	}
	probe := condProbePrefix + hex.EncodeToString(nonce[:])
	if i := strings.LastIndex(key, "/"); i >= 0 {
		probe = key[:i+1] + probe
	}

	first := s.putIfAbsent(ctx, probe, strings.NewReader(""), 0)
	if errors.Is(first, ErrConditionalWriteUnsupported) {
		s.condState.Store(condUnsupported)
		return s.conditionalCreateVerdict(key)
	}
	if first != nil {
		// Nothing decided; the probe may or may not exist.
		_ = s.Delete(ctx, probe)
		return fmt.Errorf("objstore: verify conditional create: %w", first)
	}
	second := s.putIfAbsent(ctx, probe, strings.NewReader(""), 0)
	_ = s.Delete(ctx, probe)
	switch {
	case errors.Is(second, ErrExists):
		s.condState.Store(condSupported)
	case second == nil, errors.Is(second, ErrConditionalWriteUnsupported):
		s.condState.Store(condUnsupported)
	default:
		return fmt.Errorf("objstore: verify conditional create: %w", second)
	}
	return s.conditionalCreateVerdict(key)
}

// conditionalCreateVerdict turns a decided condState into PutIfAbsent's answer
// for key.
func (s *S3Store) conditionalCreateVerdict(key string) error {
	if s.condState.Load() == condSupported {
		return nil
	}
	return fmt.Errorf("objstore: PUT %q: %w (the S3 endpoint ignores or rejects If-None-Match)", key, ErrConditionalWriteUnsupported)
}

// Get returns the object body for key, or ErrNotFound on 404.
func (s *S3Store) Get(ctx context.Context, key string) (io.ReadCloser, error) {
	u, host := s.objectURL(key)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Host = host
	signV4(req, emptyPayloadHash, s.creds, s.region, s.service, s.clock().UTC())

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode == http.StatusNotFound {
		drainClose(resp.Body)
		return nil, ErrNotFound
	}
	if resp.StatusCode/100 != 2 {
		err := s3Error("GET", key, resp)
		drainClose(resp.Body)
		return nil, err
	}
	return resp.Body, nil
}

// Delete removes key. A 404 is reported as ErrNotFound.
func (s *S3Store) Delete(ctx context.Context, key string) error {
	u, host := s.objectURL(key)
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, u.String(), nil)
	if err != nil {
		return err
	}
	req.Host = host
	signV4(req, emptyPayloadHash, s.creds, s.region, s.service, s.clock().UTC())

	resp, err := s.client.Do(req) //nolint:bodyclose // drainClose handles drain+close
	if err != nil {
		return err
	}
	defer drainClose(resp.Body)
	if resp.StatusCode == http.StatusNotFound {
		return ErrNotFound
	}
	// S3 returns 204 No Content on success; some compat servers return 200.
	if resp.StatusCode/100 != 2 {
		return s3Error("DELETE", key, resp)
	}
	return nil
}

// listObjectsV2Result mirrors the ListObjectsV2 XML response we care about.
type listObjectsV2Result struct {
	XMLName               xml.Name       `xml:"ListBucketResult"`
	IsTruncated           bool           `xml:"IsTruncated"`
	NextContinuationToken string         `xml:"NextContinuationToken"`
	Contents              []listObjEntry `xml:"Contents"`
}

type listObjEntry struct {
	Key          string    `xml:"Key"`
	Size         int64     `xml:"Size"`
	LastModified time.Time `xml:"LastModified"`
}

// List returns every object under prefix, following ListObjectsV2 pagination.
//
// NOTE: List BUFFERS all matching keys in memory (it appends every page's entries
// to one slice until the listing is exhausted). For the backup/retention use this
// is bounded — a handful of snapshots per collection prefix — but a caller that
// lists a prefix spanning millions of objects will materialize them all at once.
// Scope List to a narrow prefix (e.g. one collection's key prefix) for large
// buckets; a streaming/callback variant is a follow-up if an unbounded scan is
// ever needed.
func (s *S3Store) List(ctx context.Context, prefix string) ([]ObjectInfo, error) {
	var out []ObjectInfo
	var token string
	for {
		q := url.Values{}
		q.Set("list-type", "2")
		if prefix != "" {
			q.Set("prefix", prefix)
		}
		if token != "" {
			q.Set("continuation-token", token)
		}
		// url.Values.Encode sorts keys and RFC3986-ish encodes; the signer
		// re-canonicalizes the query independently, so its exact form here only
		// has to be a valid, parseable query.
		u, host := s.bucketURL(q.Encode())

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
		if err != nil {
			return nil, err
		}
		req.Host = host
		signV4(req, emptyPayloadHash, s.creds, s.region, s.service, s.clock().UTC())

		resp, err := s.client.Do(req) //nolint:bodyclose // drainClose handles drain+close
		if err != nil {
			return nil, err
		}
		if resp.StatusCode/100 != 2 {
			err := s3Error("LIST", prefix, resp)
			drainClose(resp.Body)
			return nil, err
		}
		body, err := io.ReadAll(resp.Body)
		drainClose(resp.Body)
		if err != nil {
			return nil, err
		}
		var res listObjectsV2Result
		if err := xml.Unmarshal(body, &res); err != nil {
			return nil, fmt.Errorf("objstore: parse ListObjectsV2: %w", err)
		}
		for _, c := range res.Contents {
			out = append(out, ObjectInfo(c))
		}
		if !res.IsTruncated || res.NextContinuationToken == "" {
			break
		}
		token = res.NextContinuationToken
	}
	return out, nil
}

// emptyPayloadHash is the SHA-256 of the empty string, used as the
// x-amz-content-sha256 for bodyless requests (GET/DELETE/LIST).
const emptyPayloadHash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

// hashPayload computes the lowercase-hex SHA-256 of r's body for a SIGNED-payload
// PUT (used only over plaintext http:// endpoints, where UNSIGNED-PAYLOAD would
// leave the body unprotected). It returns the hash and a reader positioned at the
// start of the body to send. If r is an io.ReadSeeker (the backup temp file is)
// it hashes in a single streaming pass and rewinds — no full-body buffering;
// otherwise it buffers the body in memory to hash it. size is the expected body
// length (only used to size the buffer in the non-seekable fallback).
func hashPayload(r io.Reader, size int64) (string, io.Reader, error) {
	h := sha256.New()
	if rs, ok := r.(io.ReadSeeker); ok {
		if _, err := io.Copy(h, rs); err != nil {
			return "", nil, err
		}
		if _, err := rs.Seek(0, io.SeekStart); err != nil {
			return "", nil, err
		}
		return hex.EncodeToString(h.Sum(nil)), rs, nil
	}
	buf := bytes.NewBuffer(make([]byte, 0, max64(size, 0)))
	if _, err := io.Copy(io.MultiWriter(h, buf), r); err != nil {
		return "", nil, err
	}
	return hex.EncodeToString(h.Sum(nil)), buf, nil
}

// max64 returns the larger of a and b (used to size a buffer non-negatively).
func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// s3ErrorXML captures the S3 <Error> response body.
type s3ErrorXML struct {
	XMLName xml.Name `xml:"Error"`
	Code    string   `xml:"Code"`
	Message string   `xml:"Message"`
}

// s3Error builds a descriptive error from a non-2xx S3 response, reading the
// XML <Error> body when present.
func s3Error(op, key string, resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
	var xe s3ErrorXML
	if err := xml.Unmarshal(body, &xe); err == nil && xe.Code != "" {
		return fmt.Errorf("objstore: %s %q failed: %d %s: %s", op, key, resp.StatusCode, xe.Code, xe.Message)
	}
	return fmt.Errorf("objstore: %s %q failed: %d %s", op, key, resp.StatusCode, strings.TrimSpace(string(body)))
}

// drainClose drains and closes a response body so the connection can be reused.
func drainClose(rc io.ReadCloser) {
	if rc == nil {
		return
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(rc, 1<<16))
	_ = rc.Close()
}

var _ ObjectStore = (*S3Store)(nil)
