"""Luôn kiểm thử đúng source tree hiện tại, kể cả khi có editable install cũ."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
