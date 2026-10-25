"""Import only the exact existing PAS Windows psutil package; no sys.path change."""
import hashlib
import importlib.util
from pathlib import Path
import sys

PACKAGE = Path("D:/personal-agent-system/.venv/Lib/site-packages/psutil")
FILES = {
    "__init__.py": "7b6a0675824eb1fa2ff0cb1eb36e358dc454703e51dfa4e9a0e6ccd26a159f0c",
    "_common.py": "6fc6bf5f86491ba962521374472570238929003cebea8e9b6c55224084b52bb0",
    "_ntuples.py": "96f42bc24549636b5707a949b7cc92e89c87f7cea7d343a6d17821ac670dfcd2",
    "_pswindows.py": "0bbd52dcb214735be4168d11a2ae192d5bc7265c8cf72c611179476479687f54",
    "_psutil_windows.pyd": "0035450801bd7d938e9e146c5ec28e619cb5a5f4a18cdc53ac7e9734c7f94f78"}


def load():
    for name, expected in FILES.items():
        if hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned psutil runtime changed: " + name)
    if "psutil" not in sys.modules:
        spec = importlib.util.spec_from_file_location("psutil", PACKAGE / "__init__.py",
            submodule_search_locations=[PACKAGE.as_posix()])
        module = importlib.util.module_from_spec(spec)
        sys.modules["psutil"] = module
        spec.loader.exec_module(module)
    module = sys.modules["psutil"]
    modules = {"psutil": "__init__.py", "psutil._common": "_common.py", "psutil._ntuples": "_ntuples.py",
        "psutil._pswindows": "_pswindows.py", "psutil._psutil_windows": "_psutil_windows.pyd"}
    for name, filename in modules.items():
        actual = sys.modules.get(name)
        if actual is None or Path(actual.__file__).resolve() != (PACKAGE / filename).resolve():
            raise ValueError("Unexpected psutil runtime module origin: " + name)
    return module
