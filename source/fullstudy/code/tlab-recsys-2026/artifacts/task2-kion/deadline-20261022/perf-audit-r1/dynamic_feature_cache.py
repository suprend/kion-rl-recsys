"""Opt-in bounded raw DynamicIndex feature cache; no Torch/provider loading.

Single-owner CPU cache only. Install immediately after provider admission and
before benchmark instrumentation. Does not cache RelatedInterestIndex, personal
features, composed/normalized features, model projections, or autograd objects.
"""
from collections import OrderedDict
import hashlib
from pathlib import Path

import numpy as np

DYNAMIC_SOURCE_SHA256 = "7f53e94ddd5940514de8a132711ff5d782885983af9f0c62ae4c4515261bdd7a"
DEFAULT_MAX_BYTES = 64 << 20


class BoundedDynamicFeatureCache:
    def __init__(self, index, *, max_bytes=DEFAULT_MAX_BYTES, max_origins=2):
        # Only this reviewed implementation explicitly ignores user_ids. Related
        # interest has query-user exclusion and MUST never use this wrapper.
        original = index.features
        code = getattr(getattr(original, "__func__", None), "__code__", None)
        if (type(index).__name__ != "DynamicIndex" or code is None
                or hashlib.sha256(Path(code.co_filename).read_bytes()).hexdigest() != DYNAMIC_SOURCE_SHA256):
            raise ValueError("Cache requires the pinned, unwrapped DynamicIndex.features")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1 <= max_bytes <= DEFAULT_MAX_BYTES:
            raise ValueError("Cache byte bound must be an integer in 1..64MiB")
        if isinstance(max_origins, bool) or not isinstance(max_origins, int) or not 1 <= max_origins <= 2:
            raise ValueError("Origin bound must be an integer in 1..2")
        self.index, self.original = index, original
        self.max_bytes, self.max_origins = max_bytes, max_origins
        self._banks = OrderedDict()
        self.cache_bytes = 0
        self.calls = self.hits = self.misses = self.fallback_calls = self.evictions = 0
        self._installed = False

    def _fallback(self, tokens, queries, history_cutoffs, user_ids):
        self.fallback_calls += 1
        return self.original(tokens, queries, history_cutoffs=history_cutoffs, user_ids=user_ids)

    def features(self, item_tokens, query_dates, history_cutoffs=None, user_ids=None):
        self.calls += 1
        # Same validation order and helpers as the pinned implementation. Dates
        # are normalized before cache lookup, so invalid inputs cannot hit cache.
        tokens = self.index._tokens(item_tokens)
        n = len(tokens)
        dates = np.minimum(self.index._dates(query_dates, n, "query_dates"), self.index.fit_cutoff)
        if history_cutoffs is not None:
            dates = np.minimum(dates, self.index._dates(history_cutoffs, n, "history_cutoffs"))
        if user_ids is not None and np.asarray(user_ids).shape != (n,):
            raise ValueError("user_ids must have one value per item")
        # Keep mixed dates, unknowns, empty inputs and large generic API calls on
        # the original path. The production provider admits <=4096 point rows.
        if (not n or n > 4096 or np.any(tokens >= self.index.vocab_size)
                or np.any(dates != dates[0])):
            return self._fallback(item_tokens, query_dates, history_cutoffs, user_ids)
        size, width = self.index.vocab_size, len(self.index.columns)
        bank_bytes = size * width * np.dtype(np.float32).itemsize + size * np.dtype(np.bool_).itemsize
        if bank_bytes > self.max_bytes:
            return self._fallback(item_tokens, query_dates, history_cutoffs, user_ids)
        cutoff = int(dates[0])
        if cutoff not in self._banks:
            while self._banks and (len(self._banks) >= self.max_origins
                    or self.cache_bytes + bank_bytes > self.max_bytes):
                _, (old_values, old_valid) = self._banks.popitem(last=False)
                self.cache_bytes -= old_values.nbytes + old_valid.nbytes
                self.evictions += 1
                del old_values, old_valid
            values = np.empty((size, width), np.float32)
            valid = np.zeros(size, np.bool_)
            self._banks[cutoff] = (values, valid)
            self.cache_bytes += bank_bytes
        self._banks.move_to_end(cutoff)
        values, valid = self._banks[cutoff]
        needed = np.unique(tokens)
        missing = needed[~valid[needed]]
        self.hits += int(np.count_nonzero(valid[tokens]))
        self.misses += len(missing)
        if len(missing):
            # Raw feature values at the same strict effective cutoff. No query
            # user is required: the pinned DynamicIndex explicitly ignores it.
            block = self.original(missing, cutoff, history_cutoffs=cutoff)
            if block.shape != (len(missing), width) or block.dtype != np.float32:
                raise ValueError("Pinned DynamicIndex output shape/dtype changed")
            # Publish validity only after the complete original call succeeds.
            values[missing] = block
            valid[missing] = True
        # Advanced indexing creates independent mutable storage for the caller;
        # composed-provider normalization cannot mutate cached raw values.
        return values[tokens]

    def stats(self):
        return {"contract": "bounded-raw-dynamic-origin-token-cache-v1",
                "dynamic_source_sha256": DYNAMIC_SOURCE_SHA256,
                "max_bytes": self.max_bytes, "max_origins": self.max_origins,
                "cache_bytes": self.cache_bytes, "cached_origins": len(self._banks),
                "calls": self.calls, "hits": self.hits, "misses": self.misses,
                "fallback_calls": self.fallback_calls, "evictions": self.evictions}

    def clear(self):
        self._banks.clear()
        self.cache_bytes = 0

    def uninstall(self):
        if self._installed:
            current = self.index.__dict__.get("features")
            if current != self.features:
                raise RuntimeError("DynamicIndex.features ownership changed")
            del self.index.__dict__["features"]
            self._installed = False
        self.clear()


def install_dynamic_cache(provider, *, max_bytes=DEFAULT_MAX_BYTES, max_origins=2):
    index = provider.dynamic
    if index is None:
        raise ValueError("Provider has no DynamicIndex")
    if "features" in index.__dict__:
        raise ValueError("Install cache before wrappers; features already overridden")
    cache = BoundedDynamicFeatureCache(index, max_bytes=max_bytes, max_origins=max_origins)
    index.features = cache.features
    cache._installed = True
    return cache
