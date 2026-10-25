"""Finite relocated head validator/loader, distinct from historical source identity.

Only stdlib imports. Numerical architecture is reused from verified frozen bytes;
the explicitly supplied torch backend is touched only by load_heads_bound.
"""
import hashlib
from io import BytesIO
from pathlib import PurePosixPath
from types import ModuleType

from rl_bindings import require, safe_physical

ADAPTER_CONTRACT = 'kion-rl-bound-heads-v1'
HEADS_SHA256 = '795cdfe87f08079cadf067dc9c58811f7bcdacf9d1ebf2dbcb6c019dd68d6799'
POLICY_SHA256 = 'da7baf199b685cb467edd7d5eedaada151798ec40bbc23e9edfaee8e36340dba'


def load_frozen_heads(source_path):
    path = safe_physical(source_path)
    require(path.stat().st_size <= 8 << 20, 'Oversized frozen source')
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == HEADS_SHA256, 'Frozen heads source changed')
    module = ModuleType('_rl_frozen_heads')
    module.__file__ = str(path)
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


def _child(root, name):
    return str(PurePosixPath(str(root).replace('\\', '/')) / name)


def preflight_heads_bound(request, bindings, heads_source):
    h = load_frozen_heads(heads_source)
    h.validate_request(request)
    directory = request.get('head_dir', request['output_dir'])
    receipt = bindings.read_bound(_child(directory, 'training.json'))
    epochs = request.get('epochs', h.EPOCHS)
    steps = epochs * h.STEPS_PER_EPOCH
    expected = dict(contract=h.CONTRACT, status='completed', epochs_requested=epochs,
        epochs_completed=epochs, steps_per_head=steps,
        head_steps={name: steps for name in h.HEAD_NAMES}, rows_per_epoch=h.TRAIN_ROWS,
        batch_size=128, seed=20261024, common_batches=True, shuffle=h.SHUFFLE,
        raw_gather=h.RAW_GATHER, trained_from='fresh-common-init-final-zero')
    require(all(receipt.get(k) == v for k, v in expected.items()), 'Completed exact common training budget required')
    records = receipt.get('epoch_records')
    require(isinstance(records, list) and len(records) == h.EPOCHS, 'Exact completed epoch schedule lineage required')
    for epoch, record in enumerate(records):
        require(isinstance(record, dict) and all(record.get(k) == v for k, v in h.epoch_record(epoch).items()),
                'Completed epoch row schedule lineage differs')
    pin = request.get('bank_manifest_sha256')
    require(isinstance(pin, str) and len(pin) == 64, 'Explicit bank SHA256 required')
    bank_path = _child(request['bank_dir'], 'bank.json')
    require(bindings.sha_bound(bank_path) == pin == receipt.get('bank_manifest_sha256'), 'Bank pin differs')
    bank = bindings.read_bound(bank_path)
    require(bank.get('contract') == h.BANK_CONTRACT and bank.get('state') == 'completed'
        and bank.get('split') == 'train' and bank.get('train_rows') == h.TRAIN_ROWS
        and bank.get('natural_covered_only') is False and bank.get('same_day_positives_excluded') is True,
        'Completed unfiltered train262144 bank required')
    require(isinstance(request.get('teacher'), dict) and request['teacher']
        and request.get('feature_schema_sha256') == h.FEATURE_SCHEMA_SHA256
        and request.get('source_columns') == h.SOURCE_COLUMNS, 'Explicit teacher/schema/source pins required')
    for key in ('teacher', 'feature_schema_sha256', 'source_columns'):
        require(receipt.get(key) == request[key] == bank.get(key), 'Training lineage differs: ' + key)
    hashes = receipt.get('checkpoint_sha256')
    require(isinstance(hashes, dict) and set(hashes) == set(h.HEAD_NAMES), 'All A/B/C hashes required')
    for name in h.HEAD_NAMES:
        require(bindings.sha_bound(_child(directory, name + '/head.pt')) == hashes[name], 'Checkpoint changed: ' + name)
    return receipt


def load_heads_bound(request, bindings, torch, heads_source, device='cpu'):
    preflight = preflight_heads_bound(request, bindings, heads_source)
    expected_receipt = {k: v for k, v in preflight.items() if k != 'checkpoint_sha256'}
    h = load_frozen_heads(heads_source)
    models = h.make_heads(torch, seed=request.get('seed', 20261024), device=device)
    directory = request.get('head_dir', request.get('output_dir', ''))
    shared = None
    for name in h.HEAD_NAMES:
        path = bindings.resolve(_child(directory, name + '/head.pt'))
        with path.open('rb') as stream:
            data = stream.read((8 << 20) + 1)
        require(len(data) <= 8 << 20 and hashlib.sha256(data).hexdigest() == preflight['checkpoint_sha256'][name],
                'Checkpoint changed/oversized before decode')
        payload = torch.load(BytesIO(data), map_location='cpu', weights_only=True)
        require(payload.get('contract') == h.CONTRACT and payload.get('name') == name
            and payload.get('complete') is True and payload.get('config') == h.CONFIG
            and payload.get('raw_indices') == list(h.mask_indices(name)), 'Unrecognized/incomplete head checkpoint')
        receipt = payload['receipt']
        require(receipt == expected_receipt, 'Checkpoint receipt differs from preflight')
        identity = (receipt['epochs_completed'], receipt['steps_per_head'], receipt['bank_manifest_sha256'])
        require(identity[0] == h.EPOCHS and identity[1] == h.EPOCHS * h.STEPS_PER_EPOCH, 'Incomplete epoch budget')
        require(shared is None or shared == identity, 'Head training budgets/banks differ')
        shared = identity
        model = models[name]
        model.load_state_dict(payload['state_dict'], strict=True)
        require(all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite checkpoint')
        model.pilot_receipt = receipt
        model.eval()
    return models


def validate_preregistration_bound(request, bindings):
    require(bindings.sha_bound(request['preregistration']) == request['preregistration_sha256'], 'Preregistration pin changed')
    value = bindings.read_bound(request['preregistration'])
    required = dict(contract='kion-C-residual-RL-preregistered-v1', chosen_utility='rm-minus-frozen-item-mean',
        formula='r_RM(s,i)-m_fit(i)', coefficient=1.0, beta=.5, temperature=1., sampler_seed=20261024,
        checkpoint_steps=[0, 250, 500, 1000], start_from='original-C', original_results_known=True,
        one_additional_fit_only=True, primary_contrast='fixed-latest1000', no_gpt_unfreeze=True, no_rm_fit=True, no_sft=True)
    require(all(value.get(k) == v for k, v in required.items()), 'Residual preregistration differs')
    require(value.get('optimizer') == dict(name='Adam', updates=1000, batch_size=16, lr=1e-5,
        grad_clip=1, weight_decay=0, alpha_ce=0), 'Fixed optimizer required')
    return value


def validate_parity_proof_bound(request, manifest, bindings):
    pin = request['parity_proof']
    require(bindings.sha_bound(pin['path']) == pin['sha256'], 'FP16 parity/gradient proof changed')
    proof = bindings.read_bound(pin['path'])
    require(proof.get('state') == 'completed' and proof.get('contract') == 'kion-current-C1070-natural1024-H1-v1'
        and proof.get('implementation') == 'producer-shape-slot-v2'
        and proof.get('implementation_source_sha256') == POLICY_SHA256
        and proof.get('cache_sha256') == bindings.sha_bound(_child(request['cache_dir'], 'cache.json'))
        and proof.get('C_checkpoint_sha256') == manifest['C_checkpoint_sha256']
        and proof.get('states') == 512 and proof.get('all_actor_reference_exact') is True
        and proof.get('all_cached_allclose') is True and proof.get('all_cached_top10_exact') is True
        and proof.get('optimizer_created') is False and proof.get('learning_updates') == 0
        and proof.get('actor_reference_weights_unchanged') is True
        and proof['gradient_fixture'].get('logical_rows') == 16
        and proof['gradient_fixture'].get('dummy_output_gradient_max') == 0,
        'Exact immutable all512 producer-slot FP16 and gradient proof required')
    gradients = proof['gradient_fixture']['gradients']
    require(gradients and all(g['finite'] for g in gradients.values()) and any(g['nonzero'] for g in gradients.values()),
        'Finite nonzero head gradient proof required')
    return proof


def _request(plan, bindings):
    from train_rl import BANK_SHA, LOGICAL, METADATA
    entries = {e['role']: e for e in plan['bindings']}
    require(len(entries) == len(plan['bindings']), 'Duplicate plan binding role')
    for role in ('cache', 'heads', 'preregistration', 'proof'):
        entry = entries['metadata:' + role]
        require(entry['logical'] == LOGICAL + METADATA[role][0]
            and entry['sha256'] == METADATA[role][1]
            and bindings.sha_bound(entry['logical']) == METADATA[role][1], 'Frozen metadata pin differs: ' + role)
    require(entries['metadata:bank']['sha256'] == BANK_SHA
        and bindings.sha_bound(entries['metadata:bank']['logical']) == BANK_SHA, 'Frozen bank pin differs')
    def logical(role):
        return entries['metadata:' + role]['logical']
    cache = bindings.read_bound(logical('cache'))
    return dict(head_dir=str(PurePosixPath(logical('heads')).parent), output_dir=plan['output'],
        bank_dir=str(PurePosixPath(logical('bank')).parent), bank_manifest_sha256=entries['metadata:bank']['sha256'],
        teacher=cache['teacher'], feature_schema_sha256=cache['feature_schema_sha256'], source_columns=cache['source_columns'],
        seed=20261024, batch_size=128, epochs=3, cache_dir=str(PurePosixPath(logical('cache')).parent),
        preregistration=logical('preregistration'), preregistration_sha256=entries['metadata:preregistration']['sha256'],
        parity_proof=dict(path=logical('proof'), sha256=entries['metadata:proof']['sha256']))


def load_original_heads(plan, bindings, torch, device):
    sources = [p for p, digest in plan['source_pins'].items() if digest == HEADS_SHA256]
    require(len(sources) == 1, 'Exactly one frozen heads source required')
    return load_heads_bound(_request(plan, bindings), bindings, torch, sources[0], device)


def validate_historical_metadata(plan, bindings, manifest):
    sources = [p for p, digest in plan['source_pins'].items() if digest == POLICY_SHA256]
    require(len(sources) == 1, 'Exactly one frozen policy source required')
    path = safe_physical(sources[0])
    require(path.stat().st_size <= 8 << 20 and hashlib.sha256(path.read_bytes()).hexdigest() == POLICY_SHA256,
        'Frozen policy source changed')
    request = _request(plan, bindings)
    return dict(preregistration=validate_preregistration_bound(request, bindings),
        proof=validate_parity_proof_bound(request, manifest, bindings))
