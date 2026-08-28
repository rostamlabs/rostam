"""Python<->Go cross-stack test for the r.kv namespace on the unified Rostam
facade.

The KV store is only on the binary TCP protocol, so there is no HTTP path and no
fake worth trusting: the whole point is that the Python framing, the protocol-v2
auth prefix, and each op's byte layout agree with the Go server exactly. A slip
in any of them produces a wrong value or a spurious error, not a clean failure.
So this launches the real server with `-tcp` and drives every op against it via
``Rostam("tcp://...").kv.*``.

Uses the public package API (``from rostam import Rostam, RostamError``).

Skipped when no server binary is found (same rule as the other cross-stack
modules): $ROSTAM_SERVER_BIN, or a `rostam-server*` built at the repo root.
"""

from __future__ import annotations

import socket
import subprocess
import tempfile
import time
import unittest

from _serverbin import find_server_bin
from rostam import Rostam, RostamError


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


_BIN, _WHY = find_server_bin()


def _wait_tcp(host, port, deadline):
    while time.time() < deadline:
        try:
            socket.create_connection((host, port), timeout=0.5).close()
            return True
        except OSError:
            time.sleep(0.1)
    return False


@unittest.skipUnless(_BIN, _WHY)
class CrossStackKVTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = _free_port()
        cls.tcp = _free_port()
        cls.datadir = tempfile.mkdtemp(prefix="rostam-kv-")
        cls.proc = subprocess.Popen(
            [_BIN, "-http", f"127.0.0.1:{cls.http}", "-tcp", f"127.0.0.1:{cls.tcp}",
             "-data", cls.datadir],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not _wait_tcp("127.0.0.1", cls.tcp, time.time() + 20):
            cls.proc.kill()
            raise RuntimeError("rostam-server -tcp did not come up in time")
        cls.r = Rostam(f"tcp://127.0.0.1:{cls.tcp}")

    @classmethod
    def tearDownClass(cls):
        cls.r.close()
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except Exception:
            cls.proc.kill()

    def test_ping(self):
        self.assertTrue(self.r.kv.ping())

    def test_put_get_bytes_and_str(self):
        self.r.kv.put("k:bytes", b'{"coins":100}')
        self.assertEqual(self.r.kv.get("k:bytes"), b'{"coins":100}')
        self.r.kv.put("k:str", "hello")            # str encodes UTF-8
        self.assertEqual(self.r.kv.get("k:str"), b"hello")

    def test_miss_returns_none_not_empty(self):
        # Absent must be None, and distinct from a stored empty value.
        self.assertIsNone(self.r.kv.get("k:absent"))
        self.r.kv.put("k:empty", b"")
        self.assertEqual(self.r.kv.get("k:empty"), b"")

    def test_incr_from_missing_and_negative(self):
        self.assertEqual(self.r.kv.incr("k:ctr", 1), 1)   # missing = 0, so +1
        self.assertEqual(self.r.kv.incr("k:ctr", 5), 6)
        self.assertEqual(self.r.kv.incr("k:ctr", -2), 4)  # signed delta

    def test_delete_reports_existence(self):
        self.r.kv.put("k:del", "x")
        self.assertTrue(self.r.kv.delete("k:del"))        # existed
        self.assertFalse(self.r.kv.delete("k:del"))       # already gone
        self.assertIsNone(self.r.kv.get("k:del"))

    def test_expire_on_a_live_key(self):
        self.r.kv.put("k:ttl", "x")
        self.r.kv.expire("k:ttl", 60_000)                 # 60s — still present now
        self.assertEqual(self.r.kv.get("k:ttl"), b"x")

    def test_binary_safe_keys_and_values(self):
        key = bytes(range(256))
        val = bytes([0, 1, 2, 255, 254])
        self.r.kv.put(key, val)
        self.assertEqual(self.r.kv.get(key), val)

    def test_set_nx_stores_once(self):
        self.assertTrue(self.r.kv.set_nx("k:nx", "first"))    # absent -> stored
        self.assertFalse(self.r.kv.set_nx("k:nx", "second"))  # present -> refused
        self.assertEqual(self.r.kv.get("k:nx"), b"first")     # unchanged

    def test_cas_swaps_on_match(self):
        self.r.kv.put("k:cas", "v1")
        self.assertTrue(self.r.kv.cas("k:cas", "v2", "v1"))   # match -> swap
        self.assertEqual(self.r.kv.get("k:cas"), b"v2")
        self.assertFalse(self.r.kv.cas("k:cas", "v3", "WRONG"))  # mismatch -> no-op
        self.assertEqual(self.r.kv.get("k:cas"), b"v2")

    def test_cas_expect_absent(self):
        self.assertTrue(self.r.kv.cas("k:casabs", "v1", None))  # absent -> store
        self.assertEqual(self.r.kv.get("k:casabs"), b"v1")
        self.assertFalse(self.r.kv.cas("k:casabs", "v2", None))  # present -> refuse

    def test_compare_and_delete(self):
        self.r.kv.put("k:cad", "tok")
        self.assertFalse(self.r.kv.compare_and_delete("k:cad", "WRONG"))  # mismatch
        self.assertEqual(self.r.kv.get("k:cad"), b"tok")
        self.assertTrue(self.r.kv.compare_and_delete("k:cad", "tok"))     # match -> delete
        self.assertIsNone(self.r.kv.get("k:cad"))


@unittest.skipUnless(_BIN, _WHY)
class CrossStackKVAuthTest(unittest.TestCase):
    """The protocol-v2 auth prefix, end to end: a token guards every op."""

    @classmethod
    def setUpClass(cls):
        cls.http = _free_port()
        cls.tcp = _free_port()
        cls.datadir = tempfile.mkdtemp(prefix="rostam-kvauth-")
        # A token makes the authenticator active even on loopback.
        cls.proc = subprocess.Popen(
            [_BIN, "-http", f"127.0.0.1:{cls.http}", "-tcp", f"127.0.0.1:{cls.tcp}",
             "-api-key", "s3cret", "-data", cls.datadir],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not _wait_tcp("127.0.0.1", cls.tcp, time.time() + 20):
            cls.proc.kill()
            raise RuntimeError("authed rostam-server did not come up in time")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except Exception:
            cls.proc.kill()

    def test_correct_token_works(self):
        r = Rostam(f"tcp://127.0.0.1:{self.tcp}", auth_token="s3cret")
        try:
            r.kv.put("k", "v")
            self.assertEqual(r.kv.get("k"), b"v")
        finally:
            r.close()

    def test_missing_token_is_unauthorized(self):
        r = Rostam(f"tcp://127.0.0.1:{self.tcp}")
        try:
            with self.assertRaises(RostamError) as cm:
                r.kv.get("k")
            # The unified _types.RostamError (unlike the pre-unification
            # client.RostamError) doesn't carry a `.status` field — the status
            # is folded into the message instead (see _tcp._status_message).
            self.assertIn("unauthorized", str(cm.exception).lower())
        finally:
            r.close()

    def test_wrong_token_is_unauthorized(self):
        r = Rostam(f"tcp://127.0.0.1:{self.tcp}", auth_token="nope")
        try:
            with self.assertRaises(RostamError):
                r.kv.get("k")
        finally:
            r.close()


if __name__ == "__main__":
    unittest.main()
