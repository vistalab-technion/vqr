import os
from pathlib import Path

EXPERIMENTS_ROOT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT_DIR = EXPERIMENTS_ROOT_DIR.parent.resolve()

EXPERIMENTS_DATA_DIR = PROJECT_ROOT_DIR.joinpath("data")

EXPERIMENTS_OUT_DIR = EXPERIMENTS_ROOT_DIR.parent.joinpath("out")
os.makedirs(EXPERIMENTS_OUT_DIR, exist_ok=True)

# When inside a conda env, prioritize lib and header dirs from the env
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    # Fix CPATH for KeOps
    cpath = os.environ.get("CPATH", "")
    os.environ["CPATH"] = f"{conda_prefix}/include:{cpath}"

    # Fix import errors with torch when global cuda toolkit is installed
    ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib64:{ld_lib_path}"
