"""Unit tests for r.kv's response-frame handling, against a fake socket.

These do not need the server binary: they assert the client is robust to a peer
that sends a hostile or truncated frame — an oversized length must not make the
client buffer unboundedly, and a malformed body must surface as RostamError
(not IndexError/struct.error) with the socket discarded, not returned to the
pool. A real server never does this; a broken proxy or a bug might.

r.kv shares TcpTransport's framing (``_call``) with the flat vector API, so
this exercises the framing through ``r.kv.get`` — the same code path a vector
op would hit.
"""

from __future__ import annotations

import socket
import struct
import threading
import unittest

from rostam._tcp import _MAX_FRAME
from rostam._types import RostamError
from rostam.rostam import Rostam


def _recv_all(conn, n):
    """Read exactly n bytes, or None if the peer closes first."""
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


class _FakeServer:
    """Accepts one connection and replies to each request with `responder()`."""

    def __init__(self, responder):
        self._responder = responder
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def _serve(self):
        try:
            conn, _ = self._sock.accept()
        except OSError:
            return
        # A client that got a deliberately bad frame hangs up abruptly, so recv
        # or sendall can hit ECONNRESET — expected here, not a test failure.
        try:
            with conn:
                while True:
                    hdr = _recv_all(conn, 4)   # TCP may split even a 4-byte header
                    if hdr is None:
                        return
                    n = struct.unpack(">I", hdr)[0]
                    if _recv_all(conn, n) is None:
                        return
                    reply = self._responder()
                    if reply is None:
                        return
                    conn.sendall(reply)
        except OSError:
            return

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass


class KVFramingTest(unittest.TestCase):
    def _client(self, responder):
        srv = _FakeServer(responder)
        self.addCleanup(srv.close)
        r = Rostam(f"tcp://127.0.0.1:{srv.port}", timeout=2.0)
        self.addCleanup(r.close)
        return r

    def test_oversized_body_length_is_rejected_not_buffered(self):
        # Advertise a body far larger than a frame but send nothing more. The
        # client must reject the length outright rather than block reading it.
        r = self._client(lambda: struct.pack(">I", _MAX_FRAME + 1))
        with self.assertRaises(RostamError) as cm:
            r.kv.get("k")
        self.assertIn("invalid response frame length", str(cm.exception))

    def test_body_shorter_than_header_is_rostam_error(self):
        # body_len = 3 (< the 5-byte minimum): must be RostamError, not IndexError.
        r = self._client(lambda: struct.pack(">I", 3) + b"\x00\x00\x00")
        with self.assertRaises(RostamError):
            r.kv.get("k")

    def test_payload_length_mismatch_is_rostam_error(self):
        # Frame says body is 5 bytes (status + payloadLen=100) but no payload.
        def reply():
            return struct.pack(">I", 5) + bytes([0]) + struct.pack(">I", 100)
        r = self._client(reply)
        with self.assertRaises(RostamError):
            r.kv.get("k")

    def test_connect_failure_is_rostam_error(self):
        # Nothing is listening on this port: acquire()'s connect must convert
        # OSError into the client's RostamError contract.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()  # port now free → connection refused
        r = Rostam(f"tcp://127.0.0.1:{port}", timeout=1.0)
        self.addCleanup(r.close)
        with self.assertRaises(RostamError):
            r.kv.get("k")


if __name__ == "__main__":
    unittest.main()
