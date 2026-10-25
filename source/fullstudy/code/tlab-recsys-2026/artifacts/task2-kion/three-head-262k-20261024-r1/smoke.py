"""Root-only tiny CPU fixtures and stdlib tests; no real arrays/provider/GPU."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module
def run(request, check):
    check(stage="stdlib-fixtures-start")
    suite=unittest.TestSuite()
    for name in ("test_bank_source", "test_heads_source", "test_eval_source"):
        module=load(HERE/(name+".py"),name)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError(stream.getvalue())
    runtime_path=Path("D:/tlab-recsys-2026/artifacts/task2-kion/ready-pack-20261020/provider/runtime-native-r5/scripts/kion_ready_runtime.py")
    runtime=load(runtime_path,"_pilot_smoke_ready_runtime")
    runtime.configure_paths()
    check(stage="tiny-cpu-ML-fixture-start")
    heads_test=sys.modules["test_heads_source"]
    fixture=heads_test.tiny_cpu_fixture()
    import torch
    import numpy as np
    ev=load(HERE/"evaluate.py","_pilot_smoke_eval")
    source=np.zeros((1,3,14),dtype=np.float32)
    for i,ranks in enumerate(((1,2048,2049),(2049,2049,2049),(37,71,1024))):
        for j,rank in enumerate(ranks):
            source[0,i,3+j]=rank<=2048
            source[0,i,6+j]=rank/2049
            source[0,i,9+j]=rank==2049
    expected=np.array([[ev.rrf_row(row) for row in source[0]]],dtype=np.float32)
    np.testing.assert_array_equal(ev.rrf_scores(np,source),expected)
    malformed=source.copy()
    malformed[0,0,9]=1
    try:
        ev.rrf_scores(np,malformed)
    except (ValueError,RuntimeError,AssertionError):
        pass
    else:
        raise AssertionError("Malformed RRF flags accepted")
    import tempfile
    import random
    with tempfile.TemporaryDirectory(prefix="tiny-raw-gather-",dir=HERE) as directory:
        path=Path(directory)/"raw.npy"
        raw=np.broadcast_to(np.arange(128,dtype=np.float32)[:,None,None],(128,32,1068)).copy()
        np.save(path,raw,allow_pickle=False)
        identity={"raw_features":{"path":path.as_posix(),"bytes":path.stat().st_size,"mtime_ns":path.stat().st_mtime_ns}}
        arrays={"raw_features":np.load(path,mmap_mode="r",allow_pickle=False)}
        positions=list(range(128))
        random.Random(20261024).shuffle(positions)
        for chosen in (positions,list(reversed(positions))):
            gathered=heads_test.heads._copy_raw_batch(arrays,identity,chosen,np)
            np.testing.assert_array_equal(gathered,raw[chosen])
            if arrays["raw_features"] is not None:
                raise AssertionError("Raw mmap retained after batch")
            del gathered
        del raw
    if torch.cuda.is_initialized():
        raise RuntimeError("Tiny CPU fixture unexpectedly initialized CUDA")
    check(stage="tiny-cpu-ML-fixture-completed")
    return dict(state="passed",stdlib_tests=result.testsRun,stdlib_output=stream.getvalue(),tiny_cpu=fixture,tiny_raw_io="two-permuted-B128-gathers-mmap-closed",
                torch=torch.__version__,cuda_initialized=False,real_data_opened=False)
