"""Execute-gated historical E2 nested bank; default planning is stdlib-only.

Inputs retain their historical logical names. Exact file bindings relocate I/O.
The original bank.run and candidate sampler perform all numerical computation.
The execution envelope is cooperative (not the historical Windows Job guard).
"""
import argparse
import builtins
from contextlib import ExitStack
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from types import ModuleType

from gpt_provider import BoundIO, FileBinding
from gpt_sources import runtime_source_bindings
from input_inventory import checked_path
import prepare_head_bank as plan_source

ROOT = Path(__file__).resolve().parents[1]
BANK = plan_source.ART + '/three-head-262k-20261024-r1/bank.py'
MODE = 'historical-E2-nested32768-to262144-v1'
MANIFEST_SHA = 'a8e6e78334c1ba1ca5519a09785f0fbd26972a1142d500e3403afb07c2a748bd'
PROFILE_PINS = {'gpt-inputs.json': '81dd552e17148a97d99d85a848f57078c8402f84a7c7755bf77b2872624f7bdf',
                'gpt-transitive.json': 'afe06d3f514a94d083f99277bebbde65a6457ad68c29ac2e8147c641d1a15f01'}


def assemble_spec(bindings, *, input_root, output, reuse_root, graph_root, max_seconds):
    """Finite metadata-only historical handoff builder; no payload hashing.

    input_root is the parent of gpt_transitive transfer_recipe destinations.
    Explicit bindings override only exact file identities, never root prefixes.
    Missing files produce a runnable-shape spec plus an honest blocked inventory.
    """
    request = historical_request()
    require(isinstance(bindings, dict) and len(bindings) <= 8192, 'finite exact bindings required')
    for key, value in bindings.items():
        plan_source.logical_path(key)
        checked_path(Path(value))
    roots = {key: checked_path(Path(value)) for key, value in
             dict(input_root=input_root, reuse_root=reuse_root, graph_root=graph_root).items()}
    rows, blocked_metadata, known_profile_paths = {}, [], set()
    def role(logical):
        return 'source' if logical.endswith('.py') else 'metadata' if logical.endswith('.json') else 'payload'
    def local(logical):
        if logical in bindings:
            return checked_path(Path(bindings[logical]))
        retained = ROOT / 'source/fullstudy/code' / logical[3:]
        if logical.startswith('D:/') and logical.endswith('.py') and retained.is_file():
            return checked_path(retained)
        key = hashlib.sha256(logical.encode()).hexdigest()[:20]
        return roots['input_root'] / 'inputs' / key / Path(logical).name
    def add(logical, pin, physical=None):
        plan_source.logical_path(logical)
        plan_source.pin(pin)
        row = FileBinding(logical, checked_path(Path(physical) if physical is not None else local(logical)).as_posix(), pin, role(logical))
        if logical in rows:
            require(rows[logical].sha256 == pin, 'conflicting historical builder pins: ' + logical)
            if physical is not None:
                require(rows[logical].physical == row.physical, 'conflicting physical binding: ' + logical)
        rows[logical] = row
    for filename, pin in PROFILE_PINS.items():
        profile = plan_source.metadata(Path(__file__).with_name('profiles') / filename, pin)
        for entry in profile['documents'] + profile['entries']:
            known_profile_paths.add(entry['logical'])
            # Recovery ancestry is not an input to an already completed E2 bank.
            if entry.get('id') == 'recovery' or entry.get('document') == 'recovery':
                continue
            add(entry['logical'], entry['sha256'])
    for logical, pin in {**request['source_hashes'], **request['input_hashes'],
                         **request['provider_hook']['options']['store_source_pins']}.items():
        add(logical, pin)
    add(plan_source.ORIGINAL, plan_source.ORIGINAL_SHA)
    for pin in (request['provider_hook']['options']['coverage_receipt'],
                request['performance_hook']['acceptance_receipt']):
        add(pin['path'], pin['sha256'])
    old_logical = request['reuse_bank_dir'] + '/bank.json'
    old_local = roots['reuse_root'] / 'bank.json'
    # The old bank root is an explicit bundle binding, including the manifest.
    require(old_logical not in bindings or Path(bindings[old_logical]) == old_local,
            'old manifest override conflicts with reuse root')
    rows.pop(old_logical, None)
    add(old_logical, request['reuse_bank_manifest_sha256'], old_local)
    def read_named(logical):
        row = rows[logical]
        path = Path(row.physical)
        if not path.is_file():
            blocked_metadata.append(logical)
            return None
        return plan_source.metadata(path, row.sha256, maximum=16 << 20)
    old = read_named(old_logical)
    if old is not None:
        require(old.get('state') == 'completed' and old.get('train_rows') == 32768
                and old.get('teacher') == request['teacher'] and set(old.get('arrays', {})) == plan_source.ARRAYS,
                'old bank schema/teacher differs')
        for name, entry in old['arrays'].items():
            require(entry['path'] == name + '.npy', 'old bank array filename differs')
            logical = request['reuse_bank_dir'] + '/' + entry['path']
            physical = roots['reuse_root'] / entry['path']
            require(logical not in bindings or Path(bindings[logical]) == physical, 'old array override conflicts')
            add(logical, entry['sha256'], physical)
        counts = {int(k): v for k, v in old['date_counts'].items()}
        require(0 < len(counts) <= 4096 and all(type(n) is int and n > 0 for n in counts.values()), 'bounded date census required')
        total = sum(counts.values())
        require(total >= 262144, 'old full date census too small')
        quotas = {d: n * 262144 // total for d, n in counts.items()}
        order = sorted(counts, key=lambda d: (-(counts[d] * 262144 % total), d))
        for day in order[:262144 - sum(quotas.values())]:
            quotas[day] += 1
        key = plan_source.pin(old['candidate_contract']['cache_key'])
        for day in sorted(d for d, n in quotas.items() if n):
            filename = key + '-' + str(day) + '.json'
            logical = request['graph_cache'] + '/' + filename
            path = roots['graph_root'] / filename
            require(logical not in bindings or Path(bindings[logical]) == path, 'graph override conflicts')
            if not path.is_file():
                blocked_metadata.append(logical)
                continue
            # This named descriptor has no standalone historical file pin. Its
            # canonical signature must match the byte-pinned old bank cache key.
            from gpt_inputs import read_small
            data = read_small(path, 8 << 20)
            doc = json.loads(data, object_pairs_hook=plan_source.unique_object)
            signature = hashlib.sha256(json.dumps(doc['signature'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            require(signature == key == doc.get('cache_key') and doc.get('state') == 'completed'
                    and doc.get('origin') == doc.get('cutoff') == day
                    and doc['signature'].get('context') == old['context_signature']
                    and doc['signature'].get('adapter_sha256') == request['source_hashes'][request['candidate_source_path']],
                    'graph signature/origin differs')
            add(logical, hashlib.sha256(data).hexdigest(), path)
            require(set(doc['arrays']) == {'neighbors', 'weights', 'global_counts', 'recent_counts'}, 'graph arrays differ')
            for name, entry in doc['arrays'].items():
                filename = key + '-' + str(day) + '-' + name + '.npy'
                require(entry['path'] == filename, 'graph array filename differs')
                logical = request['graph_cache'] + '/' + filename
                physical = roots['graph_root'] / filename
                require(logical not in bindings or Path(bindings[logical]) == physical, 'graph array override conflicts')
                add(logical, entry['sha256'], physical)
    coverage_pin = request['provider_hook']['options']['coverage_receipt']
    coverage = read_named(coverage_pin['path'])
    if coverage is not None:
        from head_bank_hooks import EXPECTED_ROWS
        options = request['provider_hook']['options']
        from gpt_provider import canonical_hash
        require(coverage.get('state') == 'completed' and coverage.get('mode') == 'full'
                and coverage.get('split_rows') == EXPECTED_ROWS and coverage.get('store_path') == options['store_path']
                and coverage.get('store_source_pins_sha256') == canonical_hash(options['store_source_pins']),
                'full query-store coverage differs')
        add(options['store_path'], coverage['store_sha256'])
    sidecar = request['teacher']['path'].rsplit('.', 1)[0] + '.json'
    sidepath = local(sidecar)
    if sidepath.is_file():
        from gpt_inputs import read_small
        data = read_small(sidepath, 1 << 20)
        doc = json.loads(data, object_pairs_hook=plan_source.unique_object)
        require(doc.get('sha256') == request['teacher']['sha256'], 'E2 sidecar teacher differs')
        add(sidecar, hashlib.sha256(data).hexdigest(), sidepath)
    else:
        blocked_metadata.append(sidecar)
    # Imported inventories can include unused recovery records; caller overrides
    # themselves must only name this finite bank closure or pending graph names.
    allowed = set(rows) | set(blocked_metadata) | known_profile_paths
    require(not set(bindings) - allowed, 'undeclared explicit bank binding')
    spec = dict(mode=MODE, bindings=[vars(r) for r in rows.values()], output=str(output),
                reuse_root=roots['reuse_root'].as_posix(), graph_root=roots['graph_root'].as_posix(),
                max_seconds=max_seconds, manifest_sha256=MANIFEST_SHA)
    plan = build_plan(spec)
    missing = sorted(set(plan['blocked_input']) | set(blocked_metadata))
    return dict(spec=spec, ready=False, metadata_complete=not blocked_metadata,
                blocked_input=missing, payloads_hashed=False, inference_executed=False,
                metadata_prerequisites=[dict(logical=p, physical=(rows[p].physical if p in rows else
                    (roots['graph_root'] / Path(p).name).as_posix() if p.startswith(request['graph_cache'] + '/')
                    else local(p).as_posix())) for p in blocked_metadata],
                inventory=[dict(vars(r), exists=Path(r.physical).is_file()) for r in rows.values()],
                note='Named metadata may reveal further exact arrays; execution revalidates all pins and contracts.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def historical_request():
    return plan_source.metadata(plan_source.REQUEST, plan_source.REQUEST_SHA)


def source_module(io, logical, expected, *, importer=None):
    path = io.verify(logical, 'source', expected)
    data = path.read_bytes()
    require(len(data) <= 1 << 20 and hashlib.sha256(data).hexdigest() == expected,
            'source changed after verification')
    module = ModuleType('_bank_' + expected[:16])
    module.__file__ = path.as_posix()
    module.__builtins__ = dict(vars(builtins), **({'__import__': importer} if importer else {}))
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


def build_plan(spec):
    require(set(spec) == {'mode', 'bindings', 'output', 'reuse_root', 'graph_root',
                          'max_seconds', 'manifest_sha256'}, 'exact execution schema required')
    require(spec['mode'] == MODE and spec['manifest_sha256'] == MANIFEST_SHA,
            'only exact historical E2/provider mode admitted')
    require(type(spec['max_seconds']) in (int, float) and math.isfinite(spec['max_seconds'])
            and 1 <= spec['max_seconds'] <= 604800, 'finite fresh execution budget required')
    require(type(spec['bindings']) is list and len(spec['bindings']) <= 8192, 'finite file bindings required')
    rows = [FileBinding(**row) for row in spec['bindings']]
    present = {row.logical for row in rows}
    rows += [row for row in runtime_source_bindings(ROOT, existing=rows) if row.logical not in present]
    # Construction is stat-only; this callback is never an execution guard.
    io = BoundIO(rows, planning_read_forbidden)
    request = historical_request()
    output = checked_path(Path(spec['output']))
    reuse = checked_path(Path(spec['reuse_root']))
    graph = checked_path(Path(spec['graph_root']))
    require(not output.exists(), 'output must be fresh')
    protected = [ROOT / 'source', ROOT / 'reproduction', reuse, graph] + [Path(r.physical) for r in rows]
    require(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected),
            'output overlaps protected input/source')
    required = dict(request['source_hashes'], **request['input_hashes'])
    required[plan_source.ORIGINAL] = plan_source.ORIGINAL_SHA
    blocked = []
    for logical, pin in required.items():
        if logical not in io.bindings:
            blocked.append(logical)
        else:
            require(io.bindings[logical].sha256 == pin, 'historical pin changed: ' + logical)
    for row in rows:
        if not Path(row.physical).is_file():
            blocked.append(row.logical)
    sidecar = request['teacher']['path'].rsplit('.', 1)[0] + '.json'
    if sidecar not in io.bindings:
        blocked.append(sidecar)
    for directory in (reuse, graph):
        if not directory.is_dir():
            blocked.append(directory.as_posix())
    require(reuse != graph, 'distinct old-bank and graph roots required')
    for name in plan_source.ARRAYS | {'bank'}:
        logical = request['reuse_bank_dir'] + '/' + name + ('.json' if name == 'bank' else '.npy')
        if logical not in io.bindings:
            blocked.append(logical)
        else:
            require(Path(io.bindings[logical].physical) == reuse / Path(logical).name,
                    'old bank must have exact explicitly bound colocated filenames')
    return dict(mode='plan', execution_mode=MODE, inference_executed=False, ready=False,
                payloads_verified=False, bindings=[vars(r) for r in rows], blocked_input=sorted(set(blocked)),
                source_sha256=plan_source.SOURCE_SHA, request_sha256=plan_source.REQUEST_SHA,
                rows=262144, retained_rows=32768, new_rows=229376, candidate_width=32,
                guard='cooperative RSS12GiB/host-reserve2GiB/disk-reserve15GiB/affinity0x55/STOP/deadline',
                numerical_equivalence='untested', spec=copy.deepcopy(spec))


def planning_read_forbidden():
    raise RuntimeError('planning must not resolve or read payloads')


class Guard:
    def __init__(self, spec):
        self.output = Path(spec['output'])
        self.until = time.monotonic() + spec['max_seconds']
        self.deadline = datetime.now(timezone.utc) + timedelta(seconds=spec['max_seconds'])
        self.calls = 0
        self.snapshots = {}
        self.next_resource_check = 0.0

    def __call__(self, **fields):
        self.calls += 1
        if time.monotonic() >= self.until or datetime.now(timezone.utc) >= self.deadline:
            raise TimeoutError('bank deadline exhausted')
        if (self.output / 'STOP').exists():
            raise InterruptedError('bank cancelled by STOP')
        now = time.monotonic()
        if now < self.next_resource_check:
            return
        self.next_resource_check = now + 1.0
        import psutil
        process = psutil.Process()
        require(process.memory_info().rss <= 12 * 2**30, 'cooperative RSS cap exceeded')
        require(psutil.virtual_memory().available >= 2 * 2**30, 'host memory reserve exhausted')
        require(process.cpu_affinity() == [0, 2, 4, 6], 'CPU affinity differs from0x55')
        probe = self.output
        while not probe.exists():
            probe = probe.parent
        require(shutil.disk_usage(probe).free >= 15 * 2**30, 'disk reserve exhausted')
        for path, old in self.snapshots.items():
            info = checked_path(path).stat()
            require((info.st_size, info.st_mtime_ns, info.st_ino) == old, 'input changed during execution')

    def watch(self, rows):
        for row in rows:
            path = checked_path(Path(row.physical))
            info = path.stat()
            self.snapshots[path] = (info.st_size, info.st_mtime_ns, info.st_ino)


def checked_digest(path, check):
    digest = hashlib.sha256()
    with checked_path(Path(path)).open('rb') as stream:
        while True:
            check(stage='hash')
            data = stream.read(1 << 20)
            if not data:
                return digest.hexdigest()
            digest.update(data)


def graph_class(module, io, logical_root, physical_root):
    """Finite graph inventory and paths; original table validation/math retained."""
    names = ('neighbors', 'weights', 'global_counts', 'recent_counts')
    class BoundCandidates(module.LiveCandidates):
        def _build(self, *a, **kw):
            raise RuntimeError('missing graph: reconstruction forbidden')

        def _disk_check(self, extra=0):
            self.check(stage='graph-disk-check')
            require(extra == 0, 'read-only graphs cannot allocate')
            paths = [Path(r.physical) for key, r in io.bindings.items() if key.startswith(logical_root + '/')]
            require(paths and all(p.parent == physical_root for p in paths), 'finite colocated graph inventory required')
            require(sum(p.stat().st_size for p in paths) <= self.max_disk_bytes, 'graph disk cap exceeded')
            require(shutil.disk_usage(physical_root).free >= self.min_disk_free_bytes, 'graph disk reserve exhausted')

        def _paths(self, origin):
            prefix = logical_root + '/' + self.cache_key + '-' + str(int(origin))
            descriptor = io.verify(prefix + '.json', 'metadata')
            doc = io.read_json(prefix + '.json')
            require(doc.get('signature') == self.signature and doc.get('cache_key') == self.cache_key
                    and doc.get('state') == 'completed' and doc.get('origin') == doc.get('cutoff') == int(origin),
                    'graph descriptor signature differs')
            require(set(doc['arrays']) == set(names), 'exact graph arrays required')
            paths = {}
            for name in names:
                logical = prefix + '-' + name + '.npy'
                require(doc['arrays'][name]['path'] == Path(logical).name, 'graph filename differs')
                paths[name] = io.verify(logical, 'payload', doc['arrays'][name]['sha256'])
                require(paths[name] == physical_root / Path(logical).name, 'graph file escaped physical root')
            return descriptor, paths

        def contract(self):
            result = super().contract()
            result['cache_root'] = logical_root
            return result
    return BoundCandidates


def open_context(bank, request, spec, io, check):
    from gpt_complete import CompleteProvider
    from train_gpt import scoped_integration
    from head_bank_hooks import install_hooks
    stack = ExitStack()
    ctx = {'events': {}, 'check': check, 'closed': False, 'adapter_stack': stack}
    try:
        original = io.read_json(plan_source.ORIGINAL, expected=plan_source.ORIGINAL_SHA)
        require(all(request['source_hashes'].get(p) == h for p, h in original['source_hashes'].items()),
                'original runtime source closure differs')
        require(request['teacher']['sha256'] == bank.TEACHER_SHA and
                request['teacher']['config_sha256'] == bank.CONFIG_SHA, 'historical E2 teacher required')
        teacher = request['teacher']
        checkpoint = io.verify(teacher['path'], 'payload', bank.TEACHER_SHA)
        config = io.read_json(teacher['config_path'], expected=bank.CONFIG_SHA)
        argv = original['train_argv']
        def arg(name):
            return argv[argv.index(name) + 1]
        provider = stack.enter_context(CompleteProvider.load(arg('--data'), arg('--feature-dir'),
            bindings=list(io.bindings.values()), check=io.check, scope_decision=arg('--scope-decision'),
            manifest_sha256=spec['manifest_sha256'], external=('numpy', 'torch', 'transformers')))
        train = provider.runtime.load('kion_complete_train')
        ctx.update(train=train, provider=provider, np=train.np, torch=train.torch)
        require(train.np.__version__ == '2.2.6' and train.torch.__version__ == '2.8.0+cu126'
                and train.torch.cuda.is_available(), 'original CUDA numerical runtime required')
        args = train.parser().parse_args(argv)
        train.torch.set_num_threads(4)
        train.torch.manual_seed(bank.SEED)
        train.torch.backends.cudnn.deterministic, train.torch.backends.cudnn.benchmark = True, False
        for name in ('update', 'run', 'main'):
            setattr(train, name, bank.denied)
        train.integration_contract = scoped_integration
        def source_hash(path):
            name = Path(path).stem
            require(name in provider.runtime.pins, 'unknown config source')
            logical, pin = provider.runtime.pins[name]
            io.verify(logical, 'source', pin)
            return pin
        train.sha256 = source_hash
        def hook_import(name, globals=None, locals=None, fromlist=(), level=0):
            # These pinned hooks import util at module scope; their dynamic
            # loaders are replaced by install_hooks before invocation.
            if name in ('importlib', 'importlib.util'):
                return builtins.__import__(name, globals, locals, fromlist, level)
            return provider.runtime._import(name, globals, locals, fromlist, level)
        def load(logical):
            return source_module(io, logical, request['source_hashes'][logical], importer=hook_import)
        require(request['provider_hook'] == original['provider_hook'] and
                request['performance_hook'] == original['performance_hook'], 'original hooks differ')
        ctx['perf'] = install_hooks(provider, request, io, load)
        require(provider.audience is None and provider.people_dynamics is None, 'deferred features changed')
        q = source_module(io, request['q_source_path'], bank.Q_SHA)
        event_logical = plan_source.ART + '/all-agreed-20261020/completion-diagnostics/event-store-r1/full-r1/manifest.json'
        event = io.read_json(event_logical, expected=bank.EVENT_SHA)
        require(event['state'] == 'completed' and event['events_count'] == 4649162, 'event census differs')
        require(request['input_hashes'][str(bank.DATA / 'train.npz').replace('\\', '/')] ==
                event['prepared_pins']['train.npz']['sha256'], 'prepared train pin differs')
        for key, (dtype, pin) in bank.EVENT_PINS.items():
            require(event['arrays'][key] == dict(path=key + '.npy', sha256=pin, dtype=dtype, shape=[4649162]),
                    'event array declaration differs')
            path = io.verify(event_logical.rsplit('/', 1)[0] + '/' + key + '.npy', 'payload', pin)
            values = train.np.load(path, mmap_mode='r', allow_pickle=False)
            ctx['events'][key] = values
            require(values.shape == (4649162,) and str(values.dtype) == dtype and not values.flags.writeable,
                    'event header differs')
        module = source_module(io, request['candidate_source_path'], bank.CANDIDATE_SHA)
        module.digest = lambda path: checked_digest(path, check)
        candidates = graph_class(module, io, request['graph_cache'], Path(spec['graph_root']))(
            provider, train.np, q, ctx['events'], bank._raw_ids(provider, train.np), spec['graph_root'],
            bank.context_signature(request), check=check, seed=bank.SEED)
        ctx['candidates'] = candidates
        require(len(provider.splits['train']['y']) == 3477437 and len(provider.splits['val']['y']) == 158760,
                'full provider census differs')
        schedule = train.TrainingSchedule(provider, args)
        require(schedule.updates_per_epoch == 27235 and len(schedule.eligible) == 3477437, 'schedule identity differs')
        device = train.torch.device('cuda')
        check(stage='teacher-model-create')
        model = train.make_model(provider, args).to(device)
        ctx['model'] = model
        require(train.make_config(provider, model, args, device, schedule=schedule) == config, 'exact E2 configuration differs')
        # Explicit I/O port of shared.load_checkpoint; retain receipt digest and
        # legacy.validate_resume, without an implicit sibling sidecar read.
        sidecar = io.read_json(teacher['path'].rsplit('.', 1)[0] + '.json')
        require(sidecar['sha256'] == checked_digest(checkpoint, check), 'checkpoint sidecar digest differs')
        saved = train.torch.load(checkpoint, map_location='cpu', weights_only=False)
        train.legacy.validate_resume(saved['config'], config)
        require(saved['state']['epoch'] == 2 and saved['state']['offset'] == 0 and saved['state']['updates'] == 54470,
                'completed historical E2 state required')
        model.load_state_dict(saved['state_dict'])
        del saved
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        columns = model.feature_columns['candidate']
        schema = hashlib.sha256(json.dumps(columns, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        require(model.candidate_feature_dim == 1068 and schema == bank.SCHEMA_SHA, 'original schema differs')
        require(list(candidates.source_feature_names[12:14]) == bank.SOURCE_COLUMNS, 'source columns differ')
        with train.torch.inference_mode(), train.legacy.autocast(device):
            table = model.candidate_table()
        ctx.update(args=args, device=device, table=table, metadata=provider.metadata, teacher=copy.deepcopy(teacher),
                   feature_columns=columns, feature_schema_sha256=schema, source_columns=bank.SOURCE_COLUMNS,
                   candidate_contract=candidates.contract())
        return ctx
    except BaseException:
        bank.close_context(ctx)
        raise


def run_bound(bank, request, spec, io, check):
    """Original builder body, with explicit I/O and context lifetime seams."""
    old_close, old_reuse, old_write = bank.close_context, bank.open_reuse, bank.write_new
    bank.REUSE_DIR = Path(spec['reuse_root'])
    bank.BANK_OUTPUT = Path(spec['output']) / 'train'
    bank.DATA = Path('D:/tlab-recsys-2026/data/processed/kion-full-context64')
    def close(ctx):
        result = old_close(ctx)
        stack = ctx.pop('adapter_stack', None)
        if stack is not None:
            try:
                stack.close()
            except BaseException as exc:
                result['errors'].append(dict(owner='provider_runtime', error=repr(exc)))
                result['teardown_completed'] = False
        return result
    bank.close_context = close
    bank.open_context = lambda req, callback: open_context(bank, request, spec, io, callback)
    def reuse(req, np, ctx, callback, maps):
        return old_reuse(dict(req, reuse_bank_dir=spec['reuse_root']), np, ctx, callback, maps)
    bank.open_reuse = reuse
    bank.digest = lambda path: checked_digest(path, check)
    def publish(path, value):
        # Preserve logical provenance while recording physical relocation separately.
        value['reuse']['bank_dir'] = request['reuse_bank_dir']
        value['portable_execution'] = dict(mode=MODE, adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                           original_request_sha256=plan_source.REQUEST_SHA)
        old_write(path, value)
    bank.write_new = publish
    actual = dict(request, bank_output=bank.BANK_OUTPUT.as_posix())
    return bank.run(actual, check)


def execute(spec, *, execute_bank=False):
    require(execute_bank is True, 'explicit --execute-bank required')
    plan = build_plan(spec)
    require(not plan['blocked_input'], 'blocked historical inputs: ' + repr(plan['blocked_input']))
    output = Path(spec['output'])
    guard = Guard(spec)
    import psutil
    process = psutil.Process()
    affinity = process.cpu_affinity()
    process.cpu_affinity([0, 2, 4, 6])
    receipt = dict(state='started', mode=MODE, original_request_sha256=plan_source.REQUEST_SHA,
                   numerical_equivalence='untested', guard=plan['guard'], bindings=plan['bindings'],
                   interpreter=sys.executable, deadline_utc=guard.deadline.isoformat(),
                   adapters={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ('execute_head_bank.py', 'head_bank_hooks.py', 'gpt_complete.py',
                                          'gpt_provider.py', 'gpt_sources.py', 'train_gpt.py')})
    try:
        guard()
        rows = [FileBinding(**r) for r in plan['bindings']]
        guard.watch(rows)
        io = BoundIO(rows, guard)
        request = historical_request()
        output.mkdir(parents=True, exist_ok=False)
        def write(name, value):
            with (output / name).open('x', encoding='utf-8', newline='\n') as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
        write('started.json', receipt)
        try:
            # All original source/input pins remain mandatory before activation.
            for logical, pin in {**request['source_hashes'], **request['input_hashes']}.items():
                io.verify(logical, io.bindings[logical].role, pin)
            bank = source_module(io, BANK, plan_source.SOURCE_SHA)
            result = run_bound(bank, request, spec, io, guard)
            guard.next_resource_check = 0.0
            guard(stage='completed')
            receipt.update(state='completed', bank_manifest_sha256=checked_digest(output / 'train/bank.json', guard))
            write('completed.json', receipt)
            return result
        except BaseException as exc:
            write('failed.json', dict(receipt, state='failed', error=repr(exc)))
            raise
    finally:
        process.cpu_affinity(affinity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--spec', type=Path)
    mode.add_argument('--build-spec', action='store_true')
    parser.add_argument('--execute-bank', action='store_true')
    parser.add_argument('--bindings', type=Path)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--reuse-root', type=Path)
    parser.add_argument('--graph-root', type=Path)
    parser.add_argument('--max-seconds', type=float)
    parser.add_argument('--spec-output', type=Path)
    args = parser.parse_args(argv)
    if args.build_spec:
        require(not args.execute_bank, 'spec builder never executes inference')
        require(all(value is not None for value in (args.input_root, args.output_dir, args.reuse_root,
                    args.graph_root, args.max_seconds, args.spec_output)), 'builder paths and budget required')
        bindings = plan_source.metadata(args.bindings) if args.bindings else {}
        result = assemble_spec(bindings, input_root=args.input_root, output=args.output_dir,
                               reuse_root=args.reuse_root, graph_root=args.graph_root, max_seconds=args.max_seconds)
        target = checked_path(args.spec_output.absolute())
        protected = [checked_path(args.output_dir), checked_path(args.reuse_root), checked_path(args.graph_root),
                     ROOT / 'source', ROOT / 'reproduction'] + [Path(r['physical']) for r in result['spec']['bindings']]
        if args.bindings:
            protected.append(checked_path(args.bindings.absolute()))
        require(not target.exists() and all(not target.is_relative_to(p) and not p.is_relative_to(target) for p in protected),
                'spec output must be fresh and disjoint from inputs/source/run')
        with target.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result['spec'], stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        result['spec_path'] = target.as_posix()
        print(json.dumps({k: v for k, v in result.items() if k != 'spec'}, ensure_ascii=False, indent=2))
        return 0
    spec = plan_source.metadata(args.spec)
    result = execute(spec, execute_bank=True) if args.execute_bank else build_plan(spec)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
