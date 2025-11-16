# defines.py — paths & data-base discovery
from pathlib import Path
import os
from typing import List

ROOT = Path(__file__).parent.resolve()
BASE_DATA_PATH = ROOT / "Data"

def _possible_gdrive_bases() -> List[Path]:
    bases: List[Path] = []
    env_base = os.environ.get("FINALPROJECT_DATA_BASE", "").strip()
    if env_base:
        bases.append(Path(env_base))

    userprofile = Path(os.environ.get("USERPROFILE", "")).expanduser()
    candidates: List[Path] = [
        Path("G:/My Drive/FinalProject/Data"),
        Path("G:/Drive/My Drive/FinalProject/Data"),
        userprofile / "Google Drive/My Drive/FinalProject/Data",
        userprofile / "My Drive/FinalProject/Data",
        userprofile / "Google Drive/FinalProject/Data",
        userprofile / "OneDrive/Google Drive/My Drive/FinalProject/Data",
    ]
    for p in candidates:
        try:
            if p.exists():
                bases.append(p)
        except Exception:
            pass

    out: List[Path] = []
    seen = set()
    for b in bases:
        try:
            rb = b.resolve()
        except Exception:
            rb = b
        s = str(rb)
        if s not in seen:
            seen.add(s)
            out.append(rb)
    return out

EXTERNAL_DATA_BASES: List[Path] = _possible_gdrive_bases()

PROCESSED_DATA_DIR = ROOT / "processed_data"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS_CONFIG_PATH = ROOT / "config" / "patients_config.json"

DATA_BASES_FOR_SEARCH: List[Path] = []
DATA_BASES_FOR_SEARCH.append(BASE_DATA_PATH)
DATA_BASES_FOR_SEARCH.extend(EXTERNAL_DATA_BASES)
if ROOT not in DATA_BASES_FOR_SEARCH:
    DATA_BASES_FOR_SEARCH.append(ROOT)
if PROCESSED_DATA_DIR not in DATA_BASES_FOR_SEARCH:
    DATA_BASES_FOR_SEARCH.append(PROCESSED_DATA_DIR)
