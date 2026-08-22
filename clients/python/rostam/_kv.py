"""Key-value operations, reached as ``client.kv.*``.

The KV store is not on the REST API — it lives only on the binary TCP
protocol, because it is built for sub-microsecond operations that an HTTP
round trip would defeat. So ``_KV`` speaks that protocol directly, via the
``TcpTransport`` it is handed: it reuses that transport's connection pool,
wire framing (``_call``), and auth token rather than opening a second pool,
so KV and vector ops interleave freely on the same pooled sockets.

Wire, for reference (all big-endian) — unchanged from the pre-unification
``kv.py``:

    frame     [len u32][body]
    body v1   [opNameLen u8][opName][argsLen u32][args]
    body v2   [0x02][tokenLen u8][token][opNameLen u8][opName][argsLen u32][args]
    response  [bodyLen u32][status u8][payloadLen u32][payload]

On the HTTP backend, ``client.kv`` is instead a ``_KVUnavailable`` sentinel:
the KV store has no REST surface, so any attribute access raises
``TransportError`` rather than silently no-op-ing.
"""

from __future__ import annotations

import struct
from typing import Optional, Union

from ._types import TransportError

Key = Union[str, bytes]


def _as_bytes(x: Key) -> bytes:
    return x.encode("utf-8") if isinstance(x, str) else bytes(x)


def _enc_key(key: bytes) -> bytes:
    if len(key) > 0xFFFF:
        raise ValueError(f"key length {len(key)} exceeds 65535")
    return struct.pack(">H", len(key)) + key


class _KV:
    """Key-value operations over Rostam's native binary TCP protocol.

    Holds the parent ``TcpTransport`` and calls its ``_call`` directly, so KV
    ops share that transport's connection pool and auth token with the flat
    vector API (``r.search``, ``r.upsert``, ...) rather than owning a second
    pool of their own.
    """

    def __init__(self, transport):
        self._t = transport

    def get(self, key: Key) -> Optional[bytes]:
        """Return the value bytes, or ``None`` if the key is absent."""
        return self._t._call("get", _enc_key(_as_bytes(key)), idempotent=True)

    def put(self, key: Key, value: Key, *, ttl_ms: int = 0) -> None:
        """Store ``value`` under ``key``. ``ttl_ms`` > 0 sets an expiry."""
        k = _as_bytes(key)
        v = _as_bytes(value)
        args = _enc_key(k) + struct.pack(">I", len(v)) + v + struct.pack(">Q", ttl_ms)
        self._t._call("put", args)

    def delete(self, key: Key) -> bool:
        """Delete ``key``. Returns whether it existed."""
        payload = self._t._call("del", _enc_key(_as_bytes(key)))
        return bool(payload and payload[0])

    def incr(self, key: Key, delta: int = 1) -> int:
        """Atomically add ``delta`` (may be negative) and return the new value.

        A missing key is treated as 0, so the first ``incr`` returns ``delta``.
        """
        args = _enc_key(_as_bytes(key)) + struct.pack(">q", delta)
        payload = self._t._call("incr", args)
        return struct.unpack(">q", payload)[0]

    def expire(self, key: Key, ttl_ms: int) -> None:
        """Set a TTL (in milliseconds) on an existing key."""
        args = _enc_key(_as_bytes(key)) + struct.pack(">Q", ttl_ms)
        self._t._call("expire", args)

    def ping(self) -> bool:
        """Round-trip a heartbeat; True if the server answered."""
        self._t._call("__ping__", b"", idempotent=True)
        return True


class _KVUnavailable:
    """Installed as ``client.kv`` on the HTTP backend.

    The KV store has no REST surface, so any attribute access (``.get``,
    ``.put``, or anything else) raises ``TransportError`` — never on
    construction, only when a caller actually reaches for ``r.kv.*``.
    """

    def __getattr__(self, name: str):
        raise TransportError(
            "key-value operations require the TCP transport; connect with tcp://host:7000"
        )
