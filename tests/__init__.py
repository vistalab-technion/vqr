# Output dir for tests that need to generate output
import os
from pathlib import Path

TESTS_OUT_DIR = Path(__file__).parent.joinpath("out")
os.makedirs(TESTS_OUT_DIR, exist_ok=True)
