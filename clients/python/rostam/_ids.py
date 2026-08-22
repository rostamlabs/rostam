"""Stable mapping from external string ids to Rostam's uint64 point ids.

A purely-numeric string is used verbatim; anything else is hashed (BLAKE2b,
8 bytes) to a stable 64-bit id, so repeated upserts/deletes of the same external
id address the same point across runs.

.. warning:: **Do not mix numeric-string ids and non-numeric ids in the same
   collection.** Numeric strings ("42") map verbatim into the uint64 space;
   non-numeric strings hash (BLAKE2b-64) into the *same* space with no tag bit.
   An astronomically unlikely BLAKE2b collision would cause one point to
   silently overwrite another.  In practice this is not a concern, but callers
   who derive ids from user input should use one id form consistently per
   collection, or call the int-id ``Rostam`` methods directly.
"""

from __future__ import annotations

import hashlib


def to_uint64(external: str) -> int:
    if external.isdigit():
        v = int(external)
        if v < (1 << 64):
            return v
    return int.from_bytes(hashlib.blake2b(external.encode("utf-8"), digest_size=8).digest(), "big")
