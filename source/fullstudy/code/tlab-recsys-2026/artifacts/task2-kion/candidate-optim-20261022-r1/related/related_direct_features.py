"""Exact related-feature assembly using fixed schema slices and singleton stats.

Opt-in single-owner CPU adapter for a new provider. Keeps the original bounded
user-aware result/portfolio/global caches and strict LOO methods untouched.
No provider/data imports, new persistent cache, model or autograd objects.
"""
import hashlib
from pathlib import Path
from types import MethodType
import numpy as np

SOURCE_SHA256 = '0999efdccd9d682e4c1d0ce0393bdab12a3abdbc8b2e7fb14a4288a6dde8fae8'
KINDS = ('studios', 'franchises')
AGGREGATES = ('mean', 'max', 'median', 'top3_mean')
WINDOW_FIELDS = ('pairs', 'distinct_viewers', 'pairs_log1p', 'distinct_viewers_log1p',
    'previous_pairs', 'previous_distinct_viewers', 'observed_works', 'global_pairs',
    'global_distinct_viewers', 'previous_global_pairs', 'previous_global_distinct_viewers',
    'pair_growth', 'viewer_growth', 'growth_supported', 'growth_observed', 'observed_days',
    'full_window', 'empty_global', 'previous_empty_global')
RELEASE_FIELDS = tuple(f'{f}_{w}d' for w in (7, 30, 60, 90) for f in
    ('count_lower', 'count_upper', 'observed_count_upper', 'observed_exact_day_count',
     'has_lower', 'has_upper', 'count_exact', 'window_known')) + (
    'known_date_works', 'unknown_date_works', 'exact_date_works', 'interval_date_works',
    'history_complete', 'recency_lower_days', 'recency_upper_days', 'recency_bounds_known',
    'days_since_latest', 'recency_exact_known', 'observed_recency_lower_days',
    'observed_recency_upper_days', 'observed_recency_bounds_known', 'release_scope_unknown')
SUPPORT_FIELDS = ('relation_known', 'listed_entities', 'other_cards', 'other_work_units',
    'confirmed_other_works', 'unresolved_work_fraction', 'candidate_work_confirmed', 'has_other_work')
SUMMARY_FIELDS = ('pairs_log1p', 'distinct_viewers_log1p', 'pair_growth', 'viewer_growth')


def _expected_columns():
    columns = ['related.unknown_item', 'related.query_user_excluded', 'related.cutoff_ordinal']
    for kind in KINDS:
        prefix = f'related.{kind}.'
        columns.extend(prefix + name for name in SUPPORT_FIELDS)
        for window in (1, 7, 30):
            columns.extend(prefix + f'union_{window}d.' + name for name in WINDOW_FIELDS)
            columns.extend(prefix + f'entity_{window}d.{field}.{agg}'
                           for field in SUMMARY_FIELDS for agg in AGGREGATES)
        columns.extend(prefix + 'release.' + name for name in RELEASE_FIELDS)
    return columns


def _features_direct(self, item_tokens, query_dates, history_cutoffs=None, user_ids=None):
    # Validation and iteration/cache identity deliberately match pinned features.
    tokens = np.asarray(item_tokens)
    if tokens.ndim != 1 or tokens.dtype.kind not in 'iu' or np.any(tokens < 0) or np.any(tokens >= self.vocab_size):
        raise ValueError('One-dimensional mapped integer tokens required')
    n = len(tokens)
    if n > 4096:
        raise ValueError('At most 4096 candidate rows per call')
    def dates(values, name):
        if len(values) != n:
            raise ValueError(name + ' must match tokens')
        return [self._direct_day(value) for value in values]
    queries = dates(query_dates, 'query_dates')
    cuts = dates(history_cutoffs, 'history_cutoffs') if history_cutoffs is not None else queries
    if user_ids is not None:
        uids = np.asarray(user_ids)
        if uids.shape != (n,) or uids.dtype.kind not in 'iu' or np.any(uids < 0):
            raise ValueError('Nonnegative query user IDs must match tokens')
    else:
        uids = [None] * n
    output = np.zeros((n, len(self.columns)), np.float32)
    for row, (token, query, cut, uid) in enumerate(zip(tokens, queries, cuts, uids)):
        if token < 3:
            output[row, 0] = 1
            continue
        cutoff = min(query, cut, self.fit_cutoff)
        wid = int(self.token_work[token])
        key = (int(token), cutoff, int(uid) if uid is not None else None)
        if key not in self._result_cache:
            values = np.zeros(len(self.columns), np.float32)
            values[1:3] = uid is not None, cutoff
            position = 3
            for kind in KINDS:
                labels = self.item_entities[kind][int(token)]
                portfolios = [set(self.portfolios[kind][entity]) - {wid} for entity in labels]
                others = sorted(set().union(*portfolios)) if portfolios else []
                confirmed = sum(self.works[work]['confirmed'] for work in others)
                values[position:position + 8] = (
                    bool(labels), len(labels), sum(len(self.works[work]['cards']) for work in others), len(others),
                    confirmed, (len(others) - confirmed) / len(others) if others else 0,
                    self.works[wid]['confirmed'], bool(others))
                position += 8
                if labels:
                    for window in (1, 7, 30):
                        metrics = self._metrics(others, wid, cutoff, window, uid)
                        values[position:position + len(WINDOW_FIELDS)] = [metrics[name] for name in WINDOW_FIELDS]
                        position += len(WINDOW_FIELDS)
                        per_entity = ([metrics] if len(portfolios) == 1 else
                            [self._metrics(sorted(works), wid, cutoff, window, uid) for works in portfolios])
                        for field in SUMMARY_FIELDS:
                            if len(per_entity) == 1:
                                # NumPy means/median sum from +0, so -0 becomes
                                # +0 in those fields; maximum keeps its sign.
                                value = per_entity[0][field]
                                mean = value + 0.0
                                values[position:position + 4] = mean, value, mean, mean
                            else:
                                a = np.array([entity[field] for entity in per_entity])
                                summary = (a.mean(), a.max(), np.median(a), np.sort(a)[-3:].mean()) if len(a) else (0,) * 4
                                values[position:position + 4] = summary
                            position += 4
                    release = self._release(others, cutoff)
                    values[position:position + len(RELEASE_FIELDS)] = [release[name] for name in RELEASE_FIELDS]
                    position += len(RELEASE_FIELDS)
                else:
                    position += 3 * (len(WINDOW_FIELDS) + 4 * len(SUMMARY_FIELDS)) + len(RELEASE_FIELDS)
            self._result_cache[key] = values
            while len(self._result_cache) > self.vocab_size * self.parameters['cache_days']:
                self._result_cache.popitem(last=False)
        # Independent row assignment preserves the original no-alias guarantee.
        output[row] = self._result_cache[key]
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite related features')
    return output


class RelatedDirectFeatures:
    def __init__(self, index):
        original = index.features
        function = getattr(original, '__func__', None)
        code = getattr(function, '__code__', None)
        if (type(index).__name__ != 'RelatedInterestIndex' or code is None or code.co_name != 'features'
                or hashlib.sha256(Path(code.co_filename).read_bytes()).hexdigest() != SOURCE_SHA256
                or 'features' in index.__dict__ or '_direct_day' in index.__dict__
                or index.columns != _expected_columns()):
            raise ValueError('Requires pinned, unwrapped RelatedInterestIndex.features/schema')
        # Reuse the pinned validator, including precise bool/uint/date handling.
        self.index = index
        self.day = function.__globals__['_day']
        self.method = MethodType(_features_direct, index)
        index._direct_day = self.day
        index.features = self.method
        self.installed = True

    def uninstall(self):
        if self.installed:
            if (self.index.__dict__.get('features') != self.method
                    or self.index.__dict__.get('_direct_day') is not self.day):
                raise RuntimeError('RelatedInterestIndex adapter ownership changed')
            del self.index.__dict__['features']
            del self.index.__dict__['_direct_day']
            self.installed = False


def install_related_direct_features(provider):
    if provider.related is None:
        raise ValueError('Provider has no RelatedInterestIndex')
    return RelatedDirectFeatures(provider.related)
