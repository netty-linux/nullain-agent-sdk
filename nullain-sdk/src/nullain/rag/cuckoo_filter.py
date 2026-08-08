"""Nullain Agent SDK — Cuckoo Filter for RAG existence pre-filtering.

A Cuckoo Filter is a probabilistic set membership structure, like a Bloom
filter, but — unlike Bloom — it supports deletion. That matters here: RAG
Tree nodes get deleted/re-indexed over a document's lifetime, and a Bloom
filter has no way to "forget" a key without rebuilding the whole structure.

Used as a cheap existence pre-check before a vector search
(docs/FUSION_PLAN.md §4): given a `(tenant_id, key)` pair — `key` typically
a document id, chunk id, or source path — `might_contain` answers "could
this possibly be indexed?" in O(1) with no network call. A `False` means
"definitely not indexed, skip the embedding + vector search entirely."
A `True` may be a false positive (bounded by `false_positive_rate`), so
callers must still verify against the real store — the filter only ever
saves work, never replaces the source of truth.

Keys are always tenant-scoped (`_fingerprint` hashes `f"{tenant_id}:{key}"`)
so this structure carries the same multi-tenant discipline as the rest of
the RAG module — one tenant's insert can never produce a false "yes" that
looks like another tenant's data, since the hashed key differs entirely.

Implementation: a standard partial-key cuckoo hash table — each key maps to
two candidate buckets (`_index` and its "partner" via `fingerprint XOR
hash(fingerprint)`), each bucket holds up to `bucket_size` fingerprints.
Insertion tries both candidate buckets, then evicts and relocates on
collision up to `max_kicks` times before reporting the filter full.
"""

from __future__ import annotations

import hashlib
import math
import random

_EMPTY = 0  # a fingerprint of 0 marks an empty slot; never a valid fingerprint


class CuckooFilter:
    """Probabilistic, deletable set membership over tenant-scoped keys.

    Args:
        capacity: Expected number of items. The table is sized to
            `capacity / bucket_size`, rounded up to a power of two (required
            for the XOR-based partner-bucket trick).
        bucket_size: Fingerprints per bucket (4 is the standard cuckoo-filter
            choice — balances load factor against false-positive rate).
        fingerprint_bits: Bits per fingerprint. More bits means a lower
            false-positive rate and more memory per entry.
        max_kicks: Eviction attempts before `add` gives up and reports the
            filter full (`False`) rather than looping forever on a
            near-saturated table.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        *,
        bucket_size: int = 4,
        fingerprint_bits: int = 16,
        max_kicks: int = 500,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        num_buckets = max(1, math.ceil(capacity / bucket_size))
        self._num_buckets = 1 << (num_buckets - 1).bit_length()  # next power of two
        self._bucket_size = bucket_size
        self._fingerprint_mask = (1 << fingerprint_bits) - 1
        self._max_kicks = max_kicks
        self._buckets: list[list[int]] = [[] for _ in range(self._num_buckets)]
        self._count = 0
        self._rng = random.Random(0)  # noqa: S311 - deterministic eviction order, not crypto

    def __len__(self) -> int:
        return self._count

    def _fingerprint(self, tenant_id: str, key: str) -> int:
        digest = hashlib.blake2b(f"{tenant_id}:{key}".encode(), digest_size=8).digest()
        fp = int.from_bytes(digest, "big") & self._fingerprint_mask
        # A fingerprint of 0 is indistinguishable from "empty slot" — remap
        # the rare collision to 1 rather than special-casing zero everywhere.
        return fp or 1

    def _index(self, tenant_id: str, key: str) -> int:
        digest = hashlib.blake2b(f"{tenant_id}:{key}".encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._num_buckets

    def _partner_index(self, index: int, fingerprint: int) -> int:
        fp_hash = int.from_bytes(
            hashlib.blake2b(fingerprint.to_bytes(4, "big"), digest_size=8).digest(), "big"
        )
        return (index ^ fp_hash) % self._num_buckets

    def add(self, tenant_id: str, key: str) -> bool:
        """Insert `key` for `tenant_id`. Returns False if the filter is full
        (both candidate buckets full and eviction exceeded `max_kicks`) —
        callers should size `capacity` generously rather than handle this
        as a normal path."""
        fp = self._fingerprint(tenant_id, key)
        i1 = self._index(tenant_id, key)
        i2 = self._partner_index(i1, fp)

        if len(self._buckets[i1]) < self._bucket_size:
            self._buckets[i1].append(fp)
            self._count += 1
            return True
        if len(self._buckets[i2]) < self._bucket_size:
            self._buckets[i2].append(fp)
            self._count += 1
            return True

        # Both candidate buckets full: evict-and-relocate (cuckoo hashing).
        index = self._rng.choice((i1, i2))
        for _ in range(self._max_kicks):
            slot = self._rng.randrange(len(self._buckets[index]))
            fp, self._buckets[index][slot] = self._buckets[index][slot], fp
            index = self._partner_index(index, fp)
            if len(self._buckets[index]) < self._bucket_size:
                self._buckets[index].append(fp)
                self._count += 1
                return True
        return False

    def might_contain(self, tenant_id: str, key: str) -> bool:
        """True means "possibly indexed" (subject to the false-positive
        rate); False means "definitely not indexed" — safe to skip the
        embedding + vector search for this key entirely."""
        fp = self._fingerprint(tenant_id, key)
        i1 = self._index(tenant_id, key)
        i2 = self._partner_index(i1, fp)
        return fp in self._buckets[i1] or fp in self._buckets[i2]

    def remove(self, tenant_id: str, key: str) -> bool:
        """Remove `key` for `tenant_id`. Returns False if the key's
        fingerprint wasn't present in either candidate bucket."""
        fp = self._fingerprint(tenant_id, key)
        i1 = self._index(tenant_id, key)
        i2 = self._partner_index(i1, fp)
        for index in (i1, i2):
            bucket = self._buckets[index]
            if fp in bucket:
                bucket.remove(fp)
                self._count -= 1
                return True
        return False


__all__ = ["CuckooFilter"]
