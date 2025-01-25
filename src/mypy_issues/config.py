from __future__ import annotations

from pathlib import Path
from typing import Final, TypedDict

SNIPPETS_ROOT: Final = Path("./downloaded").resolve()
INVENTORY_ROOT: Final = SNIPPETS_ROOT / "inventory.json"

RUN_OUTPUTS_ROOT: Final = Path("./outputs")
LEFT_OUTPUTS: Final = RUN_OUTPUTS_ROOT / "left"
RIGHT_OUTPUTS: Final = RUN_OUTPUTS_ROOT / "right"


class InventoryItem(TypedDict):
    filename: str
    mypy_version: str | None
    created_at: int
