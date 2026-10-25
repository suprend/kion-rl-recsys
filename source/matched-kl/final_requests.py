"""Root metadata factory for fixed six arms and shared evaluations. Never launches."""
from pathlib import Path
import importlib.util
import argparse
import math

Z=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_prepare',Z/'prepare.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)

def pin(p):return dict(path=p.as_posix(),sha256=P.sha(p))

def arms():
    f=P.read(Z/'protocols/frozen-beta.json')
    result={}
    for seed in [20261024,20261025,20261026]:
        for u in ['completion','residual']:
            name=f'{u}-{seed}'
            if seed==20261024 and u=='completion':
                folder=P.O/'policy/rl-r2'
            elif seed==20261024:
                folder=Z/'policy'/f['selected']['operation']
            else:folder=Z/'policy'/name
            t=P.read(folder/'training.json')
            assert t['state']=='completed' and t['updates']==1000 and t['frozen_encoder'] and t['supervised_updates']==0
            entry=dict(path=(folder/'head-1000.pt').as_posix(),sha256=P.sha(folder/'head-1000.pt'),step=1000,seed=seed,utility=u,training=pin(folder/'training.json'))
            if folder.is_relative_to(Z):
                assert t['seed']==seed and t['utility']==u
                assert t['beta']==(.5 if u=='completion' else f['beta'])
                entry['loader_module']=pin(Z/'policy/policy.py')
            else:entry['seed_provenance']=pin(P.O/'preregistration.json')
            result[name]=entry
    for seed in [20261025,20261026]:
        c=P.read(Path(result[f'completion-{seed}']['training']['path']))
        r=P.read(Path(result[f'residual-{seed}']['training']['path']))
        assert c['sampler_positions_sha256']==r['sampler_positions_sha256'],'Paired sampling order differs'
    return result

def train_kl():
    r=P.base_request('train-final-six-r1')
    a=arms()
    r['checkpoints']=[dict(name=name,path=e['path'],sha256=e['sha256'],step=1000,utility=e['utility']) for name,e in a.items()]
    r['evaluation_output']=(Z/'policy/train-final-six-r1.json').as_posix();r['output_paths']=[r['evaluation_output']]
    for e in a.values():
        r['input_hashes'][e['path']]=e['sha256'];r['input_hashes'][e['training']['path']]=e['training']['sha256']
    r['input_hashes'][(Z/'protocols/frozen-beta.json').as_posix()]=P.sha(Z/'protocols/frozen-beta.json')
    P.save_request(r)

def matching():
    receipt=P.read(Z/'operations/train-final-six-r1/receipt.json')
    assert receipt['state']=='succeeded' and receipt['cleanup_quiescent']
    x=P.read(Z/'policy/train-final-six-r1.json');a=arms();t=P.read(Z/'protocols/target.json')['target_train_KL']
    k={name:x['results'][name]['metrics']['completion']['exact_KL'] for name in a}
    assert math.isfinite(t) and t>1e-8
    assert all(math.isfinite(v) and v>=0 for v in k.values())
    for name,e in a.items():
        measured=x['results'][name]['checkpoint']
        assert measured['sha256']==e['sha256'] and Path(measured['path']).resolve()==Path(e['path']).resolve()
    assert k['completion-20261024']==t
    ratios={str(s):(k[f'residual-{s}']/k[f'completion-{s}'] if k[f'completion-{s}']>0 else None) for s in [20261024,20261025,20261026]}
    out=dict(contract='completed-train-KL-match-binding-v1',created_utc=P.utc(),selected_checkpoints={name:e['sha256'] for name,e in a.items()},
             target_train_KL=t,tolerance=[.8,1.2],train_KL_by_arm=k,ratio_by_seed=ratios,
             train_KL_match_by_seed={s:v is not None and .8<=v<=1.2 for s,v in ratios.items()},
             cache=pin(P.O/'policy/train-cache-r1/cache.json'),reference=pin(Z/'policy/train-target-r1.json'),
             final_train_diagnostics=pin(Z/'policy/train-final-six-r1.json'),beta_decision=pin(Z/'protocols/frozen-beta.json'),
             claim='operational mean TRAIN512 natural1024 forward-KL match; not perstate/catalogue equivalence',no_dev_selection=True)
    assert not (Z/'protocols/matching.json').exists();P.write(Z/'protocols/matching.json',out)
    P.write(Z/'protocols/scientific-arms.json',a)
    print(out)

def evaluation(part):
    op=part+'-six-r1'
    r=P.read(P.F/('request-'+('full' if part=='full' else 'screen')+'-scientific3-r1.json'))
    r.update(operation=op,deadline_utc='2026-10-25T09:45:00+03:00',maximum_seconds=1800 if part=='full' else 900,
             module_path=(Z/'policy/evaluate.py').as_posix(),function='evaluate' if part=='full' else 'score',
             rl_updates=0,supervised_updates=0,new_rm_fits=0,final_test_reads=0,
             owned_output_roots=[(Z/'policy').as_posix()],required_output_bytes={'D:':350000000},
             cpu_threads=4,cap_gib=8,reserve_gib=2,seed=20261024,batch_size=128,
             scientific_checkpoints=arms())
    field='full_evaluation_dir' if part=='full' else 'screen_evaluation_dir'
    r[field]=(Z/'policy'/op).as_posix();r['output_paths']=[r[field]]
    for p in [Z/'policy/evaluate.py',Z/'policy/policy.py',*(Z/'control'/n for n in ['common.py','launch.py','supervise.py','worker.py'])]:
        r['source_hashes'][p.as_posix()]=P.sha(p)
    for e in r['scientific_checkpoints'].values():
        for pin_entry in [dict(path=e['path'],sha256=e['sha256']),e['training'],e.get('seed_provenance')]:
            if pin_entry:r['input_hashes'][pin_entry['path']]=pin_entry['sha256']
    for p in [Z/'protocols/preregistration.json',Z/'protocols/frozen-beta.json',Z/'protocols/matching.json']:
        r['input_hashes'][p.as_posix()]=P.sha(p)
    P.save_request(r)

def analyze(part):
    op='analyze-'+part+'-r1'
    r=P.base_request(op)
    inp=Z/'policy'/(part+'-six-r1')/'evaluation.json'
    r.update(stage='analyze',module_path=(Z/'interpretation/analyze.py').as_posix(),function='run',
             evaluation=pin(inp),output_dir=(Z/'interpretation'/(part+'-r1')).as_posix(),matching_receipt=pin(Z/'protocols/matching.json'),
             owned_output_roots=[(Z/'interpretation').as_posix()],required_output_bytes={'D:':25000000},cap_gib=4,rows=4096 if part=='full' else 256)
    r['output_paths']=[r['output_dir']]
    r['source_hashes'][r['module_path']]=P.sha(r['module_path'])
    for p in [inp,Z/'protocols/matching.json']:r['input_hashes'][p.as_posix()]=P.sha(p)
    m=P.read(Z/'protocols/matching.json')
    for e in [m['cache'],m['reference']]:
        assert P.sha(e['path'])==e['sha256']
        r['input_hashes'][e['path']]=e['sha256']
    P.save_request(r)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train_kl','matching','evaluation','analyze']);p.add_argument('part',nargs='?');a=p.parse_args()
    if a.action in ('evaluation','analyze'):globals()[a.action](a.part)
    else:globals()[a.action]()
