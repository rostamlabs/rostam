"""Python<->Go cross-stack smoke for the binary bulk-ingest wire (RVB1).

It launches the REAL rostam-server binary and drives it via the Rostam facade's
binary methods: bulk_stage + bulk_build (the vectors-only initial-load fast path), and
batch_upsert with per-point payloads (the filter-case path). The point of running
against the live Go server is that the framing is a byte-level contract between
two languages — a mock would only prove the client agrees with itself.

The key assertion is EQUIVALENCE, not merely "it worked": the same points are
loaded once over the binary wire and once over the JSON wire, and both
collections must return identical search results. That is what makes the binary
path a pure re-encoding rather than a second ingest semantics.

The server binary is located via $ROSTAM_SERVER_BIN, or a `rostam-server*` built
at the repo root. The whole module is skipped when no binary is found.
"""

from __future__ import annotations

import socket
import subprocess
import tempfile
import time
import unittest

from _serverbin import find_server_bin
from rostam import Rostam, RostamError

DIM = 16
N = 200




def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _vec(i):
    # Unique per id (so a point's own nearest neighbour is itself, with no ties)
    # and deliberately full of negatives and fractions: a byte-order or f32/f64
    # slip in the framing shows up as a wrong distance, not a wrong count.
    return [i * 0.037 - 3.1 + d * 0.011 for d in range(DIM)]


_BIN, _WHY = find_server_bin()


@unittest.skipUnless(_BIN, _WHY)
class CrossStackBinaryBulkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        cls.datadir = tempfile.mkdtemp(prefix="rostam-binbulk-")
        cls.proc = subprocess.Popen(
            [_BIN, "-http", f"127.0.0.1:{cls.port}", "-data", cls.datadir],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.c = Rostam(cls.base, timeout=120)
        deadline = time.time() + 20
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                raise RuntimeError("rostam-server exited during startup")
            try:
                if cls.c.health():
                    break
            except Exception:
                time.sleep(0.1)
        else:
            cls.proc.kill()
            raise RuntimeError("rostam-server did not become healthy in time")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except Exception:
            cls.proc.kill()

    def test_binary_stage_matches_json_stage(self):
        ids = list(range(1, N + 1))
        vecs = [_vec(i) for i in ids]

        self.c.create_collection("binstage", dim=DIM, metric="l2")
        self.c.create_collection("jsonstage", dim=DIM, metric="l2")

        staged = self.c._t.bulk_stage("binstage", ids, vecs)
        self.assertEqual(staged, N)
        self.c.bulk_build("binstage")

        # Same points over the untouched JSON staging body.
        self.c._t._request(
            "POST",
            "/v1/collections/jsonstage/points/bulk",
            {"points": [{"id": i, "vector": v} for i, v in zip(ids, vecs)]},
        )
        self.c.bulk_build("jsonstage")

        q = _vec(42)
        bin_hits = self.c.search("binstage", q, k=10)
        json_hits = self.c.search("jsonstage", q, k=10)
        self.assertTrue(bin_hits, "binary-staged collection returned no hits")
        self.assertEqual(
            [(h.id, h.distance) for h in bin_hits],
            [(h.id, h.distance) for h in json_hits],
            "binary and JSON staging produced different indexes",
        )
        # The nearest neighbour of a point that IS in the set is that point, so a
        # byte-order slip cannot hide behind a plausible-looking ordering.
        self.assertEqual(bin_hits[0].id, 42)

    def test_binary_batch_with_payloads_is_filterable(self):
        ids = list(range(1, N + 1))
        vecs = [_vec(i) for i in ids]
        metas = [{"id": i, "bucket": "even" if i % 2 == 0 else "odd"} for i in ids]

        self.c.create_collection("binbatch", dim=DIM, metric="l2")
        count = self.c._t.batch_upsert("binbatch", ids, vecs, metadatas=metas)
        self.assertEqual(count, N)

        from rostam import filters as f

        hits = self.c.search("binbatch", _vec(42), k=25, filter=f.gte("id", 150))
        self.assertTrue(hits, "filtered search over binary-loaded payloads returned nothing")
        for h in hits:
            self.assertGreaterEqual(h.id, 150)

        hits = self.c.search("binbatch", _vec(42), k=25, filter=f.eq("bucket", "odd"))
        self.assertTrue(hits)
        for h in hits:
            self.assertEqual(h.id % 2, 1)

        # The vectors themselves survived the wire byte-for-byte.
        got = self.c.get_batch("binbatch", [7])
        self.assertEqual(len(got), 1)
        self.assertEqual([round(x, 5) for x in got[0].vector], [round(x, 5) for x in _vec(7)])

    def test_payload_flag_accepted_on_staging_route(self):
        # The staging op used to store vectors ONLY, so a payload-bearing binary
        # body was refused (refusing beat silently dropping). It now carries the
        # payload section through to the multi-core build, so the same body is
        # ACCEPTED — and the payload has to actually be there afterwards, which is
        # what the old refusal was protecting.
        from rostam._http import _encode_bulk_body

        self.c.create_collection("binflag", dim=DIM, metric="l2")
        body = _encode_bulk_body([1], [_vec(1)], payloads=[{"id": 1}])
        self.c._t._send(
            "POST", "/v1/collections/binflag/points/bulk", body, "application/octet-stream"
        )
        self.c.bulk_build("binflag")
        got = self.c.get_batch("binflag", [1])
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].metadata, {"id": 1})

    def test_large_load_is_split_across_requests(self):
        # The server caps one binary body at 256 MiB / 262,144 points. bulk_stage
        # must split rather than 413, or its advertised "load a million vectors"
        # use case does not work. Force many chunks with a tiny per-request span.
        import rostam._http as rc

        ids = list(range(1, 5001))
        vecs = [_vec(i) for i in ids]
        self.c.create_collection("binsplit", dim=DIM, metric="l2")

        orig = rc._points_per_request
        rc._points_per_request = lambda dim, payload_bytes=0: 500  # 10 requests
        try:
            staged = self.c._t.bulk_stage("binsplit", ids, vecs)
        finally:
            rc._points_per_request = orig
        # The count is summed across requests, not taken from the last one.
        self.assertEqual(staged, len(ids))

        self.c.bulk_build("binsplit")
        hits = self.c.search("binsplit", _vec(4321), k=5)
        self.assertEqual(hits[0].id, 4321, "a point from a later chunk went missing")

    def test_chunk_sizing_accounts_for_payload_bytes(self):
        # Sizing a chunk on the vector row alone overshot the server's 256 MiB
        # body cap once metadata was attached: at low dim the point ceiling binds,
        # and 131,072 points x ~2 KB of metadata is ~266 MiB of payload before a
        # single vector. Those requests 413'd.
        from rostam._http import _payload_bytes, _points_per_request

        SERVER_CAP = 256 << 20
        meta = {"blob": "x" * 2000}
        pb = _payload_bytes([meta] * 100)
        self.assertGreater(pb, 2000, "payload estimate ignored the metadata size")

        for dim in (4, 16, 128, 768):
            per_point = 8 + dim * 4 + 4 + 2100  # row + length prefix + payload
            n = _points_per_request(dim, pb)
            self.assertLess(
                n * per_point, SERVER_CAP,
                f"dim={dim}: {n} points x {per_point}B exceeds the server's body cap",
            )
        # Without payloads the chunk stays large — the accounting must not
        # pessimize the plain staging path.
        self.assertGreater(_points_per_request(768, 0), _points_per_request(768, pb))

    def test_bulk_stage_carries_payloads_end_to_end(self):
        # The staging route used to REFUSE metadata (it had nowhere to put it),
        # which forced every filter case onto the inline batch route. It now
        # carries payloads through to the multi-core build, over both encodings.
        #
        # Asserted end to end rather than on the status code alone: staging a
        # payload and getting a 200 proves nothing if the payload is not there
        # afterwards, and "accepted then dropped" is the exact failure the old
        # refusal existed to prevent.
        from rostam import filters as f

        self.c.create_collection("binpayload", dim=DIM, metric="l2")
        ids = list(range(1, N + 1))
        vecs = [_vec(i) for i in ids]
        # Every fifth point carries NO payload, so the mixed batch — the shape that
        # makes the client's per-point length prefixes load-bearing — is covered.
        metas = [None if i % 5 == 0 else {"id": i, "bucket": i % 7} for i in ids]
        staged = self.c._t.bulk_stage("binpayload", ids, vecs, metadatas=metas)
        self.assertEqual(staged, N)
        self.c.bulk_build("binpayload")

        # The payloads survived the build, and landed on the RIGHT points.
        got = {p.id: p.metadata for p in self.c.get_batch("binpayload", ids)}
        self.assertEqual(len(got), N)
        for i in ids:
            if i % 5 == 0:
                self.assertFalse(got[i], f"id {i} was staged without a payload but has {got[i]}")
            else:
                self.assertEqual(got[i], {"id": i, "bucket": i % 7})

        # And they are FILTERABLE, which is the whole reason to carry them.
        hits = self.c.search("binpayload", _vec(7), k=N, filter=f.eq("bucket", 3))
        want = {i for i in ids if i % 5 != 0 and i % 7 == 3}
        self.assertTrue(want, "the filter must select a non-empty subset")
        self.assertEqual({h.id for h in hits}, want)

    def test_bulk_stage_still_refuses_what_it_cannot_carry(self):
        # Metadata is carried now; content, sparse, TTLs and CAS preconditions
        # still are not, and must be REFUSED rather than dropped.
        self.c.create_collection("binrefuse", dim=DIM, metric="l2")
        for name, point in (
            ("content", {"id": 1, "vector": _vec(1), "content": "hello"}),
            ("key_ttl_ms", {"id": 1, "vector": _vec(1),
                            "metadata": {"t": {"kind": "int", "int": 1}},
                            "key_ttl_ms": {"t": 86400000}}),
            ("ttl_ms", {"id": 1, "vector": _vec(1), "ttl_ms": 60000}),
        ):
            with self.subTest(field=name):
                with self.assertRaises(RostamError) as cm:
                    self.c._t._request(
                        "POST",
                        "/v1/collections/binrefuse/points/bulk",
                        {"points": [point]},
                    )
                self.assertEqual(cm.exception.status, 400)

    def test_dim_mismatch_rejected(self):
        from rostam._http import _encode_bulk_body

        self.c.create_collection("bindim", dim=DIM, metric="l2")
        wrong = [[0.5] * (DIM + 3)]
        body = _encode_bulk_body([1], wrong)
        with self.assertRaises(RostamError) as cm:
            self.c._t._send(
                "POST", "/v1/collections/bindim/points/bulk", body, "application/octet-stream"
            )
        # The rejection comes from the shard that owns the collection config, so
        # the message names the offending vector and both dimensions.
        self.assertEqual(cm.exception.status, 400)
        self.assertIn("dim", cm.exception.message.lower())

    def test_truncated_body_rejected(self):
        from rostam._http import _encode_bulk_body

        self.c.create_collection("bintrunc", dim=DIM, metric="l2")
        body = _encode_bulk_body([1, 2], [_vec(1), _vec(2)])
        with self.assertRaises(RostamError) as cm:
            self.c._t._send(
                "POST", "/v1/collections/bintrunc/points/bulk", body[:-8], "application/octet-stream"
            )
        self.assertEqual(cm.exception.status, 400)
        # Nothing landed: the collection is still empty, so a build succeeds.
        self.c.bulk_build("bintrunc")


if __name__ == "__main__":
    unittest.main()
