"""The TCP transport's stale-pooled-connection retry.

A server may close an idle pooled connection at any time (idle timeout, a
restart, a middlebox). The next op to land on it then fails even though
nothing about the request was wrong. `_call` retries exactly once, on a fresh
connection, but only for a read op (`idempotent=True`) and only when the
failure happened before any response byte arrived — see `_tcp.py`'s `_call`/
`_exchange`/`_recv_exactly`.

These tests use a real listening socket rather than mocks: the server accepts
a connection, answers exactly one request on it, then closes it without
saying so (no `Connection: close`-equivalent on this wire) — exactly what an
idle-timeout close looks like from the client, and the same shape as
`test_transport.py`'s `StaleServer`/`StaleConnectionTest` for the HTTP
transport.
"""

from __future__ import annotations

import socket
import struct
import threading
import unittest

from rostam import Rostam, RostamError


def _recv_all(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


class _OneShotFakeServer:
    """Accepts connections in a loop; each connection answers exactly one
    request with `responder()`, then closes — never keeps a connection alive
    for a second request. Records how many connections it accepted."""

    def __init__(self, responder):
        self._responder = responder
        self.connections = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(5)
        self.port = self._sock.getsockname()[1]
        self._stop = False
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def _serve(self):
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self.connections += 1
            try:
                with conn:
                    hdr = _recv_all(conn, 4)
                    if hdr is None:
                        continue
                    n = struct.unpack(">I", hdr)[0]
                    if _recv_all(conn, n) is None:
                        continue
                    reply = self._responder()
                    if reply is not None:
                        conn.sendall(reply)
                    # Then the `with conn` block closes the socket: exactly
                    # one request per connection, silently, like an idle
                    # timeout landing right after the response.
            except OSError:
                continue

    def close(self):
        self._stop = True
        try:
            self._sock.close()
        except OSError:
            pass


def _ok_reply(payload: bytes = b"") -> bytes:
    body = bytes([0]) + struct.pack(">I", len(payload)) + payload
    return struct.pack(">I", len(body)) + body


class TcpStaleConnectionRetryTest(unittest.TestCase):
    """`r.kv.get`/`ping` etc. are idempotent=True calls; a stale pooled
    socket must be transparently retried once."""

    def setUp(self):
        self.srv = _OneShotFakeServer(lambda: _ok_reply())
        self.addCleanup(self.srv.close)
        self.r = Rostam(f"tcp://127.0.0.1:{self.srv.port}", timeout=2.0)
        self.addCleanup(self.r.close)

    def test_a_read_survives_a_connection_that_died_while_idle(self):
        # First ping opens connection #1, gets answered, and is pooled — the
        # server has already closed it from its side by the time it is
        # reused, exactly like an idle timeout.
        self.assertTrue(self.r.kv.ping())
        # The pooled socket is now dead. Without the retry this would raise
        # RostamError; with it, the failure is invisible to the caller.
        self.assertTrue(self.r.kv.ping())
        self.assertEqual(2, self.srv.connections,
                          "the retry must have opened a second connection")

    def test_the_retry_is_exactly_one_not_a_loop(self):
        # Every connection in this fixture dies after one reply, so a second
        # consecutive call must ALSO pay for a fresh connection — the retry
        # is per-call, not a standing fix.
        self.assertTrue(self.r.kv.ping())
        self.assertTrue(self.r.kv.ping())
        self.assertTrue(self.r.kv.ping())
        self.assertEqual(3, self.srv.connections)


class TcpStaleConnectionNoRetryForWritesTest(unittest.TestCase):
    """A write (idempotent=False) must never be silently replayed."""

    def setUp(self):
        self.srv = _OneShotFakeServer(lambda: _ok_reply())
        self.addCleanup(self.srv.close)
        self.r = Rostam(f"tcp://127.0.0.1:{self.srv.port}", timeout=2.0)
        self.addCleanup(self.r.close)

    def test_a_write_is_not_replayed_on_a_stale_connection(self):
        self.assertTrue(self.r.kv.ping())          # leaves a dead connection pooled
        with self.assertRaises(RostamError):
            self.r.kv.put("k", "v")                 # must fail, not risk a double write
        self.assertEqual(1, self.srv.connections,
                          "a write must not retry onto a second connection — "
                          "it must simply fail on the dead pooled socket")


class FakePoolSocketRetryTest(unittest.TestCase):
    """A narrower, mock-level check: `_call` retries exactly once when a
    *reused* socket's send fails outright (e.g. ECONNRESET/EPIPE) before any
    response byte arrives, and does not retry a freshly connected socket."""

    def test_call_retries_once_on_reused_socket_send_failure(self):
        from rostam._tcp import TcpTransport

        t = TcpTransport("127.0.0.1", 1, timeout=1.0)

        calls = []

        class _DeadSocket:
            """A pooled socket the server already closed: send fails outright,
            before any response byte comes back."""

            def sendall(self, frame):
                calls.append("send")
                raise BrokenPipeError("simulated stale pooled connection")

        class _FreshSocket:
            """A brand-new connection that answers normally."""

            def __init__(self):
                self._buf = _ok_reply()

            def sendall(self, frame):
                calls.append("send")

            def recv(self, n):
                out, self._buf = self._buf[:n], self._buf[n:]
                return out

        fresh = _FreshSocket()

        # First acquire hands back the dead *reused* pooled socket. The retry
        # must open a brand-new connection via _connect() (NOT acquire() again,
        # which could return another idle pooled socket) — so the fresh socket
        # is served through _connect.
        def fake_acquire():
            return _DeadSocket(), True

        discarded = []
        released = []
        t._pool.acquire = fake_acquire
        t._pool._connect = lambda: fresh
        t._pool.discard = lambda s: discarded.append(s)
        t._pool.release = lambda s: released.append(s)

        status_payload = t._call("get", b"", idempotent=True)
        self.assertEqual(b"", status_payload)
        self.assertEqual(["send", "send"], calls)
        self.assertEqual(1, len(discarded), "the dead reused socket must be discarded")
        self.assertEqual(1, len(released), "the fresh socket's successful response must be pooled")

    def test_call_does_not_retry_a_freshly_connected_socket(self):
        from rostam._tcp import TcpTransport

        t = TcpTransport("127.0.0.1", 1, timeout=1.0)

        class _FailingFreshSocket:
            def sendall(self, frame):
                raise BrokenPipeError("should not be retried: never pooled")

            def recv(self, n):  # pragma: no cover - sendall raises first
                return b""

        def fake_acquire():
            return _FailingFreshSocket(), False  # reused=False: a brand-new connect

        t._pool.acquire = fake_acquire
        t._pool.discard = lambda s: None
        t._pool.release = lambda s: None

        with self.assertRaises(RostamError):
            t._call("get", b"", idempotent=True)

    def test_call_does_not_retry_a_non_idempotent_op(self):
        from rostam._tcp import TcpTransport

        t = TcpTransport("127.0.0.1", 1, timeout=1.0)

        class _StaleSocket:
            def sendall(self, frame):
                raise ConnectionResetError("stale pooled connection")

            def recv(self, n):  # pragma: no cover - sendall raises first
                return b""

        def fake_acquire():
            return _StaleSocket(), True  # reused=True, but idempotent=False below

        t._pool.acquire = fake_acquire
        t._pool.discard = lambda s: None
        t._pool.release = lambda s: None

        with self.assertRaises(RostamError):
            t._call("vector_upsert", b"")  # idempotent defaults to False


if __name__ == "__main__":
    unittest.main()
