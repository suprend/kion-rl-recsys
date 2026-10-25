"""Explicit I/O adapters for the retained exact-E1 bank hooks.

No work occurs on import. The caller owns provider/runtime lifetime and admits
the immutable historical request. This module owns only reversible hooks and
the readonly query reader, never provider.close().
"""
import ast
import hashlib
from pathlib import Path
from types import MethodType

from gpt_provider import SOURCE_LIMIT, canonical_hash

ROOT = 'D:/tlab-recsys-2026/artifacts/task2-kion/'
EXPECTED_ROWS = {'train': 3477437, 'val': 158760}


def coverage(io, options):
    if (options.get('enabled') is not True or options.get('purpose') != 'full-E1'
            or options.get('allow_missing_fallback') is not False):
        raise ValueError('Bank requires enabled full-E1 store with no missing fallback')
    pin = options['coverage_receipt']
    receipt = io.read_json(pin['path'], expected=pin['sha256'])
    if (receipt.get('state') != 'completed' or receipt.get('mode') != 'full'
            or receipt.get('split_rows') != EXPECTED_ROWS
            or receipt.get('store_path') != options['store_path']
            or receipt.get('store_source_pins_sha256') != canonical_hash(options['store_source_pins'])):
        raise ValueError('Full training requires exact completed train/val coverage')
    if not receipt.get('store_sha256'):
        raise ValueError('Full coverage receipt lacks query store hash')
    return io.verify(options['store_path'], 'payload', receipt['store_sha256'])


def prepared_query(io, pin, original):
    path = io.verify(pin['path'], 'source', pin['sha256'])
    with path.open('rb') as stream:
        source = stream.read(SOURCE_LIMIT + 1)
    io.check()
    if len(source) > SOURCE_LIMIT or hashlib.sha256(source).hexdigest() != pin['sha256']:
        raise ValueError('Prepared-query source changed after admission')
    tree = ast.parse(source, filename=pin['path'])
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CompleteProvider']
    methods = ([n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == '_query']
               if len(classes) == 1 else [])
    if len(methods) != 1 or [n.arg for n in methods[0].args.kwonlyargs] != ['prepared_profile']:
        raise ValueError('Reviewed prepared-profile _query seam changed')
    namespace = dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
                 pin['path'], 'exec'), namespace)
    return namespace['_query']


class _PerformanceView:
    """Expose retained arithmetic behind the admitted composition live guard."""
    def __init__(self, provider, original):
        from gpt_complete import _Composition
        if (provider._original is not original
                or provider.candidate_features.__func__ is not _Composition.candidate_features):
            raise ValueError('Expected admitted composition candidate boundary')
        self.provider = provider
        self.candidate_features = MethodType(original.CompleteProvider.candidate_features, provider)

    def __getattr__(self, name):
        return getattr(self.provider, name)


class HookHandle:
    def __init__(self, provider):
        self.provider = provider
        self.reader = self.performance = self.dynamic = self.query = None
        self.counts = {}
        self.closed = False
        self.closed_evidence = None

    def evidence(self):
        if self.closed:
            return self.closed_evidence
        return {'contract': 'portable-head-bank-hooks-v1', 'allow_missing_fallback': False,
                'query': dict(self.counts),
                'reader': self.reader.stats() if self.reader else None,
                'performance': self.performance.evidence() if self.performance else None,
                'dynamic': self.dynamic.stats() if self.dynamic else None}

    def uninstall(self):
        if self.closed:
            return
        errors = []
        try:
            self.closed_evidence = self.evidence()
        except BaseException as error:
            errors.append(error)
        # Every owner gets a teardown attempt even if evidence or another owner
        # fails. Never remove a replacement method belonging to another caller.
        for owner in (self.dynamic, self.performance):
            if owner is not None:
                try:
                    owner.uninstall()
                except BaseException as error:
                    errors.append(error)
        if self.query is not None:
            if self.provider.__dict__.get('_query') != self.query:
                errors.append(RuntimeError('Prepared query hook ownership changed'))
            else:
                del self.provider.__dict__['_query']
                self.query = None
        if self.reader is not None:
            try:
                self.reader.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup('Bank hook teardown failed', errors)
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Hook handle is closed')
        return self

    def __exit__(self, *exc):
        self.uninstall()


def install_hooks(provider, historical_request, io, load_source):
    """Install retained query/performance/dynamic hooks on one admitted provider.

    load_source(logical) must load only the request's finite pinned source set,
    using provider.runtime's import resolver for retained kion modules. It must
    provide a private sys.path proxy for the historical personal adapter.
    """
    request = historical_request
    store_pin, perf_pin = request['provider_hook'], request['performance_hook']
    options = store_pin['options']
    if '_query' in provider.__dict__:
        raise ValueError('Prepared query boundary already overridden')
    store_path = coverage(io, options)
    if provider.personal is None or any(len(provider.splits[s]['y']) != n for s, n in EXPECTED_ROWS.items()):
        raise ValueError('Expected complete corrected E1 provider splits')
    original = provider.runtime.load('kion_complete_pack')
    provider_pin = options['provider_source']
    if provider.runtime.logical_source(original) != provider_pin['path']:
        raise ValueError('Provider logical source differs')
    io.verify(provider_pin['path'], 'source', provider_pin['sha256'])

    def load(pin):
        io.verify(pin['path'], 'source', pin['sha256'])
        return load_source(pin['path'])

    store = load(store_pin)
    codec = load(options['codec_source'])
    prepared = prepared_query(io, options['query_method_source'], original)
    personal = provider.runtime.load('kion_personal_candidates')
    perf = load(perf_pin)
    accepted = perf_pin['acceptance_receipt']
    if io.read_json(accepted['path'], expected=accepted['sha256']).get('state') != 'accepted':
        raise ValueError('Original performance acceptance required')
    # Freeze the dynamic loader to the reviewed finite source census. Keep the
    # original logical PINS intact for provenance, remap only actual reads.
    paths, modules = {}, {}
    for key, (relative, digest) in perf.PINS.items():
        logical = ROOT + relative
        paths[key] = io.verify(logical, 'source', digest)
        if key.endswith('_adapter'):
            modules[key] = load_source(logical)
    perf.checked_path = lambda key: paths[key]
    perf._load = lambda key: modules[key]
    perf.ROOT = Path(ROOT)
    dynamic_path = request['dynamic_cache_path']
    dynamic = load({'path': dynamic_path, 'sha256': request['source_hashes'][dynamic_path]})
    if provider.audience is not None or provider.people_dynamics is not None:
        raise ValueError('Historical deferred feature contract differs')
    handle = HookHandle(provider)
    try:
        handle.reader = codec.QueryStoreReader(store_path,
            expected_schema_hash=provider.personal.schema_hash,
            expected_source_pins=options['store_source_pins'])
        query = store.reader_query(type(provider)._query, prepared, handle.reader,
                                   personal.QueryProfiles, allow_missing=False, counters=handle.counts)
        handle.query = MethodType(query, provider)
        provider._query = handle.query
        handle.performance = perf.install(_PerformanceView(provider, original), **perf_pin['options'])
        handle.dynamic = dynamic.install_dynamic_cache(provider, max_bytes=64 << 20, max_origins=2)
        return handle
    except BaseException:
        handle.uninstall()
        raise
