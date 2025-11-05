# defines.py — unified paths + external data bases (Google Drive auto-detect)
from pathlib import Path
import os
from typing import List  # <-- חשוב: שימוש ב-typing.List במקום list[...] (תואם ל-Python 3.8)

# Project root (folder that contains this defines.py)
ROOT = Path(__file__).parent.resolve()

# -------- Try to detect external "Data" bases (e.g., Google Drive) --------
def _possible_gdrive_bases() -> List[Path]:
    bases: List[Path] = []
    userprofile = Path(os.environ.get("USERPROFILE", "")).expanduser()

    # 1) Env override (recommended)
    env_base = os.environ.get("FINALPROJECT_DATA_BASE", "").strip()
    if env_base:
        bases.append(Path(env_base))

    # 2) Common Google Drive locations on Windows (כולל G:\My Drive\...)
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
            if str(p) and p.exists():
                bases.append(p)
        except Exception:
            pass

    # De-duplicate while preserving order
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

# Primary in-repo "Data" folder (kept)
BASE_DATA_PATH = ROOT / "Data"

# External data bases we will also search (Google Drive etc.)
EXTERNAL_DATA_BASES: List[Path] = _possible_gdrive_bases()

# Patients config + processed dir
PATIENTS_CONFIG_PATH  = ROOT / "config" / "patients_config.json"
PROCESSED_DATA_DIR    = ROOT / "processed_data"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# A convenience merged list used by resolvers: search here in order
DATA_BASES_FOR_SEARCH: List[Path] = []
DATA_BASES_FOR_SEARCH.append(BASE_DATA_PATH)
DATA_BASES_FOR_SEARCH.extend(EXTERNAL_DATA_BASES)
if ROOT not in DATA_BASES_FOR_SEARCH:
    DATA_BASES_FOR_SEARCH.append(ROOT)

# ---- Backwards compatibility / defaults ----
PATIENT_ID    = "Patient_04"
PROCESSED_DIR = "processed_data"
BATCH_SIZE = 64
LR        = 5e-4
EPOCHS    = 60
DEVICE    = "cpu"
REBUILD_NPY = 0
