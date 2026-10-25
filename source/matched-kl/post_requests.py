"""Metadata-only factory for root-owned independent checks and report builds."""
from pathlib import Path
import argparse
import importlib.util

Z=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_prepare',Z/'prepare.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)

def pin(p):return dict(path=p.as_posix(),sha256=P.sha(p))

def request(op,module,output):
    sources=[Z/'control'/n for n in ['common.py','launch.py','supervise.py','worker.py']]
    sources.extend([Path(module),Path('D:/personal-agent-system/scripts/portability.py'),Z.parent/'ready-pack-20261020/provider/runtime-native-r5/scripts/kion_ready_runtime.py'])
    return dict(contract='kion-c-rl-native-v1',stage='analyze',operation=op,deadline_utc='2026-10-25T09:45:00+03:00',
                maximum_seconds=180,cpu_threads=4,reserve_gib=2,cap_gib=4,supervised_updates=0,new_rm_fits=0,final_test_reads=0,rl_updates=0,
                module_path=Path(module).as_posix(),function='run',source_hashes={p.as_posix():P.sha(p) for p in sources},
                input_hashes={(Z/'protocols/preregistration.json').as_posix():P.sha(Z/'protocols/preregistration.json')},
                owned_output_roots=[Z.as_posix()],output_paths=[Path(output).as_posix()],required_output_bytes={'D:':30000000})

def verify(part):
    stage='full4096' if part=='full' else 'screen'
    folder=Z/'interpretation'/(part+'-r1')
    output=Z/'review'/('numeric-'+stage+'.json')
    r=request('verify-'+part+'-r1',Z/'interpretation/verify.py',output)
    r.update(analysis=pin(folder/'analysis.json'),per_query=pin(folder/'per-query.csv'),matching_receipt=pin(Z/'protocols/matching.json'),output=output.as_posix())
    for key in ['analysis','per_query','matching_receipt']:
        e=r[key];r['input_hashes'][e['path']]=e['sha256']
    a=P.read(r['analysis']['path']);m=P.read(r['matching_receipt']['path'])
    for e in [a['evaluation'],m['cache'],m['reference'],m['final_train_diagnostics']]:
        assert P.sha(e['path'])==e['sha256']
        r['input_hashes'][e['path']]=e['sha256']
    P.save_request(r)

def figures():
    output=Z/'reports/figures-r1'
    r=request('figures-r1',Z/'reports/figures.py',output)
    r.update(analysis=pin(Z/'interpretation/full-r1/analysis.json'),matching_receipt=pin(Z/'protocols/matching.json'),output_dir=output.as_posix())
    r['source_hashes'][(Z/'interpretation/analyze.py').as_posix()]=P.sha(Z/'interpretation/analyze.py')
    for key in ['analysis','matching_receipt']:
        e=r[key];r['input_hashes'][e['path']]=e['sha256']
    P.save_request(r)

def pdf():
    output=Z/'reports/reader-r1.pdf'
    r=request('reader-pdf-r1',Z/'reports/report-build.py',output)
    r.update(part='reader',output_pdf=output.as_posix(),font='C:/Windows/Fonts/arial.ttf',font_bold='C:/Windows/Fonts/arialbd.ttf',allow_pending=False)
    for p in [Z/'reports/reader.md',Path(r['font']),Path(r['font_bold'])]:r['input_hashes'][p.as_posix()]=P.sha(p)
    P.save_request(r)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','figures','pdf']);p.add_argument('part',nargs='?');a=p.parse_args()
    if a.action=='verify':verify(a.part)
    else:globals()[a.action]()
