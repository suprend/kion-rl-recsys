"""Optional CPU prototype; call candidate_features(index, profiles, tokens).

This copies the pinned real method's non-familiarity arithmetic unchanged.
Only Counter reductions are grouped: one ragged reduction per query/role,
with candidate lookup preparation reused for a shared candidate vector.
It creates no candidate union across queries and persists no mutable cache.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

SOURCE = Path('D:/tlab-recsys-2026/artifacts/task2-kion/ready-pack-20261020/provider/runtime-native-r5/scripts')
sys.path.insert(0, SOURCE.as_posix())
from kion_personal_candidates import ROLES, multi_interest_scores


def _ragged(rows):
    lengths = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
    nonempty = np.flatnonzero(lengths)
    flat = np.concatenate(rows) if len(nonempty) else np.empty(0, np.int64)
    keys, inverse = np.unique(flat, return_inverse=True)
    starts = np.cumsum(lengths) - lengths
    return lengths, nonempty, starts[nonempty], keys, inverse


def familiarity(index, profiles, tokens, role):
    """All counts exact; float64 integer sum/divide matches np.mean(list)."""
    b, c = len(profiles.n), tokens.shape[-1]
    aligned = tokens.ndim == 2
    mean, maximum, matched = (np.zeros((b, c), np.float64) for _ in range(3))
    m = np.zeros((b, c), np.int64)
    # Tiny aligned C4 profiling showed per-query unique/reduceat setup cost
    # exceeded its savings. Short rows use exact integer Python reductions,
    # avoiding the reference's separate NumPy conversion per candidate.
    if c <= 8:
        shared_rows = None if aligned else [index.key_rows[role][token] for token in tokens]
        for row, counter in enumerate(profiles.keys[role]):
            rows = shared_rows if shared_rows is not None else [index.key_rows[role][token] for token in tokens[row]]
            for col, keys in enumerate(rows):
                m[row, col] = len(keys)
                if len(keys):
                    values = [counter[int(key)] for key in keys]
                    mean[row, col] = sum(values) / len(values)
                    maximum[row, col] = max(values)
                    matched[row, col] = sum(value != 0 for value in values)
        return mean, maximum, matched, m
    shared = None if aligned else _ragged([index.key_rows[role][token] for token in tokens])
    for row, counter in enumerate(profiles.keys[role]):
        lengths, nonempty, starts, keys, inverse = (shared if shared is not None else
            _ragged([index.key_rows[role][token] for token in tokens[row]]))
        m[row] = lengths
        if not len(nonempty):
            continue
        unique_values = np.fromiter((counter[int(key)] for key in keys), np.int64, count=len(keys))
        values = unique_values[inverse]
        mean[row, nonempty] = np.add.reduceat(values, starts) / lengths[nonempty]
        maximum[row, nonempty] = np.maximum.reduceat(values, starts)
        matched[row, nonempty] = np.add.reduceat(values != 0, starts)
    return mean, maximum, matched, m


def candidate_features(self, profiles, item_tokens, *, max_interests=None):
    """Pinned contract: shared [C] or aligned [B,C] -> float32 [B,C,D]."""
    self._check_profiles(profiles)
    tokens = np.asarray(item_tokens)
    b = len(profiles.n)
    if (tokens.ndim not in (1, 2) or (tokens.ndim == 2 and tokens.shape[0] != b)
            or tokens.dtype.kind not in 'iu' or np.any(tokens < 3)
            or np.any(tokens >= self.vocab_size)):
        raise ValueError('Candidates must be shared [C] or aligned [B,C] real mapped item tokens')
    if max_interests is not None and (max_interests not in (1, 2, 3) or isinstance(max_interests, bool)):
        raise ValueError('max_interests must be 1, 2 or 3')
    aligned = tokens.ndim == 2
    c = tokens.shape[-1]
    output = np.zeros((b, c, len(self.columns)), np.float32)
    def put(group, values):
        sl = self.groups[group]
        output[:, :, sl['start']:sl['stop']] = np.asarray(values).transpose(1, 2, 0)
    def broad(values):
        return np.broadcast_to(np.asarray(values)[:, None], (b, c))
    n = profiles.n
    director_matched = None
    for role in ROLES:
        mean, maximum, matched, m = familiarity(self, profiles, tokens, role)
        nf, df = profiles.known_n[role], profiles.distinct_n[role]
        fraction = np.divide(matched, m, out=np.zeros_like(matched), where=m > 0)
        put('familiarity_' + role, [mean, maximum, fraction, m, matched,
            m == 0, broad(nf), broad(df),
            broad(np.divide(nf, n, out=np.zeros(b), where=n > 0)),
            broad(np.divide(n - nf, n, out=np.zeros(b), where=n > 0)), broad(nf == 0)])
        if role == 'director':
            director_matched = matched, fraction, m, nf
    put('familiarity', [broad(n), broad(n == 0)])
    matched, fraction, m, nf = director_matched
    valid = (nf[:, None] > 0) & (m > 0)
    put('director_novelty', [(matched > 0) & valid, np.where(valid, 1 - fraction, 0), ~valid,
        broad(n), broad(nf), m, m == 0, broad(n == 0), broad(nf == 0)])
    for k in (1, 2, 3):
        states = profiles.interests[k]
        centers = np.stack([state['centers'] for state in states]) if b else np.zeros((0, 3, len(self.genre_columns)))
        masks = np.array([state['valid'] for state in states], bool).reshape(b, 3)
        mass = np.array([state['mass'] for state in states]).reshape(b, 3)
        score, cross_valid = multi_interest_scores(centers, masks, mass, self.genre_vectors[tokens])
        lse, _ = multi_interest_scores(centers, masks, mass, self.genre_vectors[tokens], method='lse')
        ng = np.array([state['n_known'] for state in states])
        candidate_missing = ~np.any(self.genre_incidence[tokens], axis=-1)
        values = [score, lse, cross_valid, candidate_missing if aligned else np.broadcast_to(candidate_missing, (b, c))]
        values += [broad(np.array([state[name] for state in states])) for name in ('k_eff',)]
        values += [broad(n), broad(ng), broad(np.divide(ng, n, out=np.zeros(b), where=n > 0)), broad(ng == 0)]
        values += [broad(np.array([state[name] for state in states])) for name in ('reason', 'objective', 'gain', 'max_pair_cosine')]
        values += [broad(np.array([min(state['support'][state['valid']], default=0) for state in states]))]
        values += [broad(np.divide(n - ng, n, out=np.zeros(b), where=n > 0)), broad(n == 0), broad(profiles.genre_has_unseen)]
        put(f'interest_k{k}', values)
    for field, state in profiles.text.items():
        missing = self.text_missing[field][tokens]
        cross_valid = state['valid'][:, None] & ~missing
        vectors = self.text_matrices[field][tokens]
        cosine = (np.matmul(state['profile'][:, None, :], np.swapaxes(vectors, 1, 2))[:, 0, :]
                  if aligned else state['profile'] @ vectors.T)
        support = state['n_known']
        put('text_' + field, [np.where(cross_valid, cosine, 0), cross_valid, np.broadcast_to(missing, (b, c)),
            broad(n), broad(support), broad(np.divide(support, n, out=np.zeros(b), where=n > 0)),
            broad(np.divide(n - support, n, out=np.zeros(b), where=n > 0)), broad(n == 0), broad(support == 0),
            broad(state['zero_norm']), broad(state['valid']), broad(state['mean_norm'])])
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite compatibility output')
    if max_interests is not None:
        return output[:, :, self.selected_indices(max_interests)]
    return output
