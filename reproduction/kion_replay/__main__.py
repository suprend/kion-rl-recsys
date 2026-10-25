"""Run with ``python -B reproduction/kion_replay ...`` or ``-m kion_replay``."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kion_replay.cli import main

raise SystemExit(main())
