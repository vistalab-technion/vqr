import os

# Fix CPATH for KeOps
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    cpath = os.environ.get("CPATH", "")
    os.environ["CPATH"] = f"{conda_prefix}/include:{cpath}"
