import os
from pathlib import Path

EXPERIMENTS_ROOT_DIR = Path(__file__).parent

EXPERIMENTS_OUT_DIR = EXPERIMENTS_ROOT_DIR.parent.joinpath("out")
os.makedirs(EXPERIMENTS_OUT_DIR, exist_ok=True)

# Fix CPATH for KeOps
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    cpath = os.environ.get("CPATH", "")
    os.environ["CPATH"] = f"{conda_prefix}/include:{cpath}"
