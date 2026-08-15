"""Makes scripts/ importable in tests without turning it into a package --
scripts/audit_data.py is a CLI tool, not part of the certus library."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
