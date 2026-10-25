"""Root-only bounded metadata decisions; no automatic launch or numerical imports."""
from pathlib import Path
from datetime import datetime
import argparse
import importlib.util
import json
import math

Z=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_prepare',Z/'prepare.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)

def completed(op):
    r=P.read(Z/'operations'/op/'receipt.json')
    assert r['state']=='succeeded' and r['cleanup_quiescent'] is True, r

def kl(result,name):
    return result['results'][name]['metrics']['completion']['exact_KL']

def target():
    completed('train-target-r1')
    x=P.read(Z/'policy/train-target-r1.json')
    assert x['state']=='completed' and x['rows']==512 and x['update0_max_abs_error']==0
    t=kl(x,'completion-20261024');k=kl(x,'residual-old-20261024')
    assert math.isfinite(t) and t>1e-8 and math.isfinite(k) and k>0
    beta=min(2.,max(.01,.5*math.sqrt(k/t)))
    out=dict(contract='train-KL-calibration-target-v1',created_utc=P.utc(),target_train_KL=t,old_residual_train_KL=k,
             beta1=beta,rule='clip(0.5*sqrt(K_old_residual/T),0.01,2)',
             probe=dict(path=(Z/'policy/train-target-r1.json').as_posix(),sha256=P.sha(Z/'policy/train-target-r1.json')),
             preregistration=dict(path=(Z/'protocols/preregistration.json').as_posix(),sha256=P.sha(Z/'protocols/preregistration.json')),
             dev_selection_used=False)
    assert not (Z/'protocols/target.json').exists();P.write(Z/'protocols/target.json',out)
    print(json.dumps(out))

def fit(op,utility,seed,beta):
    assert datetime.now().astimezone()<datetime.fromisoformat('2026-10-25T09:25:00+03:00')
    assert utility in ('completion','residual') and seed in (20261024,20261025,20261026) and beta>0
    ledger=P.read(Z/'fit-ledger.json')
    assert len(ledger['reservations'])<8
    assert all((Z/'operations'/v['operation']/'receipt.json').exists() for v in ledger['reservations'])
    n=len(ledger['reservations'])+1
    reservation=dict(fit_ordinal=n,reservation_id=f'fit-{n:02d}-{op}',operation=op,utility=utility,seed=seed,beta=beta,
                     optimizer_updates_max=1000,state='reserved-before-launch',reserved_utc=P.utc())
    ledger['reservations'].append(reservation)
    snapshot=Z/'protocols'/f'fit-ledger-{n:02d}.json'
    P.write(snapshot,ledger);P.write(Z/'fit-ledger.json',ledger)
    r=P.base_request(op)
    r.update(stage='train_rl',mode='train',utility=utility,seed=seed,beta=beta,rl_updates=1000,batch_size=16,
             learning_rate=1e-5,optimizer='Adam',weight_decay=0,grad_clip=1,start_from='original-C',checkpoint_steps=[0,250,500,1000],
             rl_output=(Z/'policy'/op).as_posix(),
             fit_admission=dict(maximum_new_fits=8,fit_ordinal=n,reservation_id=reservation['reservation_id'],ledger_path=snapshot.as_posix(),ledger_sha256=P.sha(snapshot)))
    r['output_paths']=[r['rl_output']]
    r['input_hashes'][snapshot.as_posix()]=P.sha(snapshot)
    for p in (Z/'protocols/target.json',Z/'protocols/frozen-beta.json'):
        if p.exists():r['input_hashes'][p.as_posix()]=P.sha(p)
    P.save_request(r)

def evaluate(op,fit_ops):
    r=P.base_request(op)
    r['checkpoints']=[]
    for fit_op in fit_ops:
        completed(fit_op)
        p=Z/'policy'/fit_op/'training.json';t=P.read(p)
        assert t['state']=='completed' and t['actual_optimizer_updates']==1000
        r['checkpoints'].append(P.descriptor(fit_op,Z/'policy'/fit_op/'head-1000.pt',t['utility']))
        r['input_hashes'][p.as_posix()]=P.sha(p)
    r['evaluation_output']=(Z/'policy'/(op+'.json')).as_posix();r['output_paths']=[r['evaluation_output']]
    for d in r['checkpoints']:r['input_hashes'][d['path']]=d['sha256']
    P.save_request(r)

def decision(pilot_ops):
    target=P.read(Z/'protocols/target.json');t=target['target_train_KL']
    rows=[]
    for op in pilot_ops:
        completed('kl-'+op)
        x=P.read(Z/'policy'/('kl-'+op+'.json'));v=kl(x,op)
        b=P.read(Z/'policy'/op/'training.json')['beta']
        rows.append(dict(operation=op,beta=b,train_KL=v,ratio_to_target=v/t,checkpoint=P.descriptor('residual-20261024',Z/'policy'/op/'head-1000.pt','residual'),
                         diagnostics=dict(path=(Z/'policy'/('kl-'+op+'.json')).as_posix(),sha256=P.sha(Z/'policy'/('kl-'+op+'.json')))))
    valid=[x for x in rows if math.isfinite(x['train_KL']) and x['train_KL']>0]
    assert valid,'No valid pilot; stop branch'
    if len(rows)==1 and not .8<=rows[0]['ratio_to_target']<=1.2:
        beta2=min(2.,max(.01,rows[0]['beta']*math.sqrt(rows[0]['train_KL']/t)))
        if beta2!=rows[0]['beta']:
            print(json.dumps(dict(action='second-pilot',beta2=beta2,pilots=rows)));return
    selected=min(valid,key=lambda x:(abs(math.log(x['train_KL']/t)),-x['beta']))
    out=dict(contract='frozen-residual-beta-v1',frozen_utc=P.utc(),target_train_KL=t,tolerance=[.8,1.2],pilots=rows,
             selected=selected,beta=selected['beta'],calibration_matched=.8<=selected['ratio_to_target']<=1.2,
             no_dev_selection=True,prospective_seeds=[20261025,20261026],checkpoint_reused_as_seed24=True)
    assert not (Z/'protocols/frozen-beta.json').exists();P.write(Z/'protocols/frozen-beta.json',out)
    print(json.dumps(out))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['target','fit','evaluate','decision']);p.add_argument('args',nargs='*');a=p.parse_args()
    if a.action=='target':target()
    elif a.action=='fit':fit(a.args[0],a.args[1],int(a.args[2]),float(a.args[3]))
    elif a.action=='evaluate':evaluate(a.args[0],a.args[1:])
    else:decision(a.args)
