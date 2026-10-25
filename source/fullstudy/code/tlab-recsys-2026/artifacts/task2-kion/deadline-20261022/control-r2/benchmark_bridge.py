"""Benchmark-only observation/cache bridge around unchanged frozen model/update."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


class CacheBenchmark:
    def __init__(self, train, install_cache, output, cell_order):
        self.train, self.install_cache = train, install_cache
        self.output, self.order = Path(output), cell_order
        self.make_model, self.update = train.make_model, train.update
        self.provider, self.cache, self.original_candidate = None, None, None
        self.cell_index, self.local_update = -1, 0
        self.started = time.perf_counter()
        self.evidence = {'cpu_threads': 2, 'cache_cells': [], 'raw_dynamic_parity_passed': True,
            'full_candidate_parity_passed': True, 'parity_cells': 0, 'cells': [],
            'raw_parity_scope': 'all raw DynamicIndex and full candidate feature arrays in first warm update of each ON cell; no extra model forward',
            'timing_limit': 'two warm/four measured updates percell; parity cost in warm update only; immutable provider caches shared; AB/BA/AB is a fixture, not independent repeats'}

    def event(self, name, **data):
        self.output.mkdir(exist_ok=True)
        value = {'event': name, 'utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.perf_counter() - self.started, **data}
        with (self.output / 'stage-progress.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')

    def end_cell(self):
        if self.provider is None:
            return
        if self.original_candidate is not None:
            self.provider.candidate_features = self.original_candidate
            self.original_candidate = None
        if self.cache is not None:
            self.evidence['cells'][-1]['cache_stats'] = self.cache.stats()
            self.cache.uninstall()
            self.cache = None
        self.event('cell_end', cell=self.order[self.cell_index], updates=self.local_update)
        self.provider = None

    def install(self):
        def make_model(provider, args):
            self.end_cell()
            self.cell_index += 1
            if self.cell_index >= len(self.order) or (args.hard_negatives, args.uniform_negatives) != (0, 3):
                raise ValueError('Exactly six fresh-model H0U3 cache cells')
            self.local_update = 0
            self.provider = provider
            name = self.order[self.cell_index]
            enabled = name.endswith('ON')
            self.evidence['cache_cells'].append('ON' if enabled else 'OFF')
            record = {'cell': name, 'cell_index': self.cell_index, 'dynamic_cache': enabled,
                'parity': [], 'cache_stats': None}
            self.evidence['cells'].append(record)
            if enabled:
                self.cache = self.install_cache(provider)
                self.original_candidate = provider.candidate_features
                original_candidate, cache = self.original_candidate, self.cache
                def candidate(rows, ids, split):
                    # Warm update only: compare the complete raw candidate pipeline
                    # with original DynamicIndex against the cached pipeline.
                    if self.local_update != 1:
                        return original_candidate(rows, ids, split)
                    dynamic_method = provider.dynamic.features
                    dynamic_records = []
                    def parity_dynamic(*args, **kwargs):
                        expected = cache.original(*args, **kwargs)
                        actual = dynamic_method(*args, **kwargs)
                        if expected.dtype != actual.dtype or expected.shape != actual.shape or expected.tobytes() != actual.tobytes():
                            self.evidence['raw_dynamic_parity_passed'] = False
                            raise ValueError('Raw dynamic cache feature bytes changed')
                        dynamic_records.append({'shape': list(actual.shape), 'dtype': str(actual.dtype),
                            'sha256': hashlib.sha256(actual.tobytes()).hexdigest()})
                        return actual
                    tick = time.perf_counter()
                    provider.dynamic.features = cache.original
                    try:
                        expected = original_candidate(rows, ids, split)
                    finally:
                        provider.dynamic.features = dynamic_method
                    provider.dynamic.features = parity_dynamic
                    try:
                        actual = original_candidate(rows, ids, split)
                    finally:
                        provider.dynamic.features = dynamic_method
                    if expected.dtype != actual.dtype or expected.shape != actual.shape or expected.tobytes() != actual.tobytes() or not dynamic_records:
                        self.evidence['full_candidate_parity_passed'] = False
                        raise ValueError('Full candidate raw feature bytes changed/no dynamic parity call')
                    record['parity'].append({'raw_dynamic': dynamic_records,
                        'candidate_shape': list(actual.shape), 'candidate_dtype': str(actual.dtype),
                        'candidate_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
                        'warm_parity_wall_seconds': time.perf_counter() - tick})
                    return actual
                provider.candidate_features = candidate
            self.event('cell_start', cell=name, cell_index=self.cell_index, dynamic_cache=enabled)
            return self.make_model(provider, args)

        def update(*args, **kwargs):
            self.local_update += 1
            name = self.order[self.cell_index]
            self.event('update_start', cell=name, local_update=self.local_update, measured=self.local_update >= 3)
            result = self.update(*args, **kwargs)
            self.event('update_end', cell=name, local_update=self.local_update, measured=self.local_update >= 3)
            return result

        self.train.make_model, self.train.update = make_model, update
        return self

    def finish(self):
        self.end_cell()
        self.train.make_model, self.train.update = self.make_model, self.update
        self.evidence['parity_cells'] = sum(bool(cell['parity']) for cell in self.evidence['cells'])
        return self.evidence
