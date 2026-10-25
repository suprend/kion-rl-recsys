"""Same36 row fixture, both new sampler, balanced cacheOFF/ON; metadata only."""
from pathlib import Path
import importlib.util
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_d22_probe_control', HERE / 'control.py')
c = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = c
exec(compile((HERE / 'control.py').read_bytes(), str(HERE / 'control.py'), 'exec'), c.__dict__)
require, read, sha, g = c.require, c.read, c.sha, c.g
COMMON, BASE_PLAN, BASE_SHA, BASE_CONFIG, BASE_CONFIG_SHA = c.COMMON, c.BASE_PLAN, c.BASE_SHA, c.BASE_CONFIG, c.BASE_CONFIG_SHA
old = c.direct(c.OLD / 'pair-probe-r1/probe.py', '757b0fb52e13fb62195e9f849f9a262b8f148db49f4b6f6dfe87589c21608811', '_d22_original_fixture_metadata')
fixture_rows, write_new = old.fixture_rows, old.write_new
ORIGINS, GROUPS = old.ORIGINS, old.GROUPS
CELLS = [{'id': 'retrieval128-cacheOFF', 'hard_negatives': 0, 'uniform_negatives': 3, 'auxiliary_weight': 1.0},
    {'id': 'retrieval128-cacheON', 'hard_negatives': 0, 'uniform_negatives': 3, 'auxiliary_weight': 1.0}]
ORDERS = [[CELLS[0]['id'], CELLS[1]['id']], [CELLS[1]['id'], CELLS[0]['id']], [CELLS[0]['id'], CELLS[1]['id']]]
OUTPUT, STATE, RESOURCE = c.paths('benchmark')
KIND = c.BENCH_KIND
_sampler_audit, _cache_bridge = None, None


def specification():
    result = old.specification()
    result.update(contract=KIND, cells=CELLS, origin_cell_orders=ORDERS,
        sampler_contract=c.SAMPLER_CONTRACT, sampler_specification=c.sampler.specification(),
        scientific_cells='same new retrieval128 proposal in both cells; cacheOFF/ON only',
        performance_contract='CPU2; raw dynamic/candidate byte parity in first warm update of each ON cell; cache reset percell; stage UTC progress',
        worker_wall_seconds=810, outer_wall_seconds=900, goal_contract={'path': c.CONTRACT.as_posix(), 'sha256': c.CONTRACT_SHA},
        prerequisites='new root benchmark decision; no old interrupted export dependency')
    return result


def validate_spec(value):
    require(value == specification() and sha(BASE_PLAN) == BASE_SHA and sha(BASE_CONFIG) == BASE_CONFIG_SHA,
        'exact new balanced36-update sampler/cache specification')


def load(path, digest, name):
    global _sampler_audit, _cache_bridge
    module = c.direct(path, digest, name)
    if Path(path).resolve() == COMMON / 'scripts/kion_complete_train.py':
        require(_sampler_audit is None, 'one actual frozen train import')
        _sampler_audit = c.sampler.install_benchmark_sampler(module)
        cache = c.direct(c.CACHE, c.CACHE_SHA, '_d22_benchmark_cache')
        bridge = c.direct(HERE / 'benchmark_bridge.py', sha(HERE / 'benchmark_bridge.py'), '_d22_benchmark_observation')
        _cache_bridge = bridge.CacheBenchmark(module, cache.install_dynamic_cache, OUTPUT / 'benchmark', sum(ORDERS, [])).install()
    return module


def compare_losses(left, right):
    expected = {'loss', 'retrieval_ce', 'sampled_auxiliary_ce'}
    require(set(left) == set(right) == expected, 'exact original frozen loss keyset')
    import math
    for key in sorted(expected):
        a, b = left[key], right[key]
        require(type(a) in (int, float) and type(b) in (int, float) and math.isfinite(a) and math.isfinite(b)
            and abs(a - b) <= max(.01, abs(a) * .001), 'same seeded cache cell loss within numerical tolerance: ' + key)

def benchmark_evidence():
    require(_sampler_audit is not None and _cache_bridge is not None, 'actual admitted sampler/cache execution evidence exists')
    perf = _cache_bridge.finish()
    # Same fresh seed, row groups and sampler must produce same candidates percell.
    report = read(OUTPUT / 'benchmark/report.json')
    records = report['records']
    for origin in ORIGINS:
        for update in range(1, 7):
            pair = [r for r in records if r['origin'] == origin and r['local_update'] == update]
            require(len(pair) == 2 and pair[0]['raw_pair_calls'] == pair[1]['raw_pair_calls'], 'same seeded candidate hashes and raw pair counts cacheOFF/ON')
            compare_losses(pair[0]['loss'], pair[1]['loss'])
    perf['same_seed_candidate_hashes_match'] = True
    perf['measured_examples_per_second'] = {
        cell['id']: 128 * len(selected) / sum(r['update_wall_seconds'] for r in selected)
        for cell in CELLS for selected in [[r for r in records if r['cell'] == cell['id'] and r['measured']]]}
    return {'sampler_contract': c.SAMPLER_CONTRACT, 'sampler_specification': c.sampler.specification(),
        'sampler_audit': _sampler_audit, 'runtime_performance': perf}

sources = c.sources
