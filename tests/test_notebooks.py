import os
import re
import logging
from pathlib import Path

import pytest
import nbformat
import nbconvert

from experiments import PROJECT_ROOT_DIR

NOTEBOOKS_DIR = PROJECT_ROOT_DIR.joinpath("notebooks")

TEST_NOTEBOOKS_PATTERN = re.compile(r"\d{2}-[\w-]+\.ipynb")
TEST_NOTEBOOK_PATHS = [
    f for f in NOTEBOOKS_DIR.glob("*") if TEST_NOTEBOOKS_PATTERN.match(f.name)
]
CELL_TIMEOUT_SECONDS = 60 * 5

_LOG = logging.getLogger(__name__)


class TestNotebooks:
    @pytest.fixture(autouse=True)
    def change_cwd(self):
        pwd = os.getcwd()
        try:
            os.chdir(NOTEBOOKS_DIR)
            yield
        finally:
            os.chdir(pwd)

    @pytest.mark.parametrize(
        "notebook_path", TEST_NOTEBOOK_PATHS, ids=[f.stem for f in TEST_NOTEBOOK_PATHS]
    )
    def test_notebook(self, notebook_path: Path):

        _LOG.info(f"Executing notebook {notebook_path}...")

        # Parse notebook
        with open(str(notebook_path), "r") as f:
            nb = nbformat.read(f, as_version=4)

        # Create preprocessor which executes the notebbook in memory - nothing is
        # written back to the file.
        ep = nbconvert.preprocessors.ExecutePreprocessor(
            timeout=CELL_TIMEOUT_SECONDS, kernel_name="python3"
        )

        # Execute. If an exception is raised inside the notebook, this test will fail.
        ep.preprocess(nb)
