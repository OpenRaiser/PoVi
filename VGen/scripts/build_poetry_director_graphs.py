#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


VGEN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VGEN_ROOT))

from vgen.tangshi300_graph_builder import main


if __name__ == "__main__":
    raise SystemExit(main())
