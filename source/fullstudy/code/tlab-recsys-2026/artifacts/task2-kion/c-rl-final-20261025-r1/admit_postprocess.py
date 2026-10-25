import datetime
import hashlib
import json
from pathlib import Path

Z = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


receipt = read(Z / 'operations/rollout-scientific3-r1/receipt.json')
assert receipt['state'] == 'succeeded' and receipt['exit_code'] == 0 and receipt['cleanup_quiescent']
campaign = read(Z / 'campaign.json')
entries = [
    ('screen-analysis-r2', 'eae634fa75fbb587adb042774ac7ba25f07908cf8133a7ef38af37cf68af0546',
     ['screen-analysis-r2-request.json', 'postprocess-requests-r2-control.json', 'analysis-v2-source-review.json']),
    ('rollout-analysis-r1', 'de4f54684ed8caa9c50198a1c929dab5f76f8ab3c6ed7c923048baedc16a43b3',
     ['rollout-summary-request.json', 'postprocess-requests-r2-control.json']),
]
for op, expected, reviews in entries:
    path = Z / f'request-{op}.json'
    assert sha(path) == expected
    request = read(path)
    for kind in ('source_hashes', 'input_hashes'):
        for name, digest in request[kind].items():
            assert sha(Path(name)) == digest, name
    assert not Path(request['output']).exists()
    assert op not in campaign['admitted_requests']
    campaign['admitted_requests'][op] = {
        'request_path': path.as_posix(), 'sha256': expected,
        'admitted_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reviews': reviews,
        'purpose': 'saved-output postprocessing only; zero optimizer/provider; root serial after quiescent receipt',
    }
(Z / 'campaign.json').write_text(json.dumps(campaign, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
state = read(Z / 'state.json')
state.update(phase='saved-output analysis and report verification', active_operation=None, rollout_completed=True,
             updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(Z / 'state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
with (Z / 'JOURNAL.md').open('a', encoding='utf-8', newline='\n') as stream:
    stream.write('\n- Actual rollout completed:192 trajectories/960 steps,336.282s,succeeded/exit0/cleanup_quiescent. Admitted independently reviewed screen-analysis-r2 and rollout-analysis-r1 sequential zero-update postprocessing. Removed five generated owned-new-Z bytecode files after resolved containment check; sources and old execution unchanged (review/owned-bytecode-cleanup.json). Initial metadata helper stdin through PAS did not execute; first launcher correctly rejected unadmitted request before any worker. Corrected by saved metadata helper; guards unchanged.\n')
print('Both requests freshly rehashed and admitted; no models run by this preparation.')
