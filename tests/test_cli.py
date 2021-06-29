import pytest

from vqr.main import parse_cli


def test_version():
    with pytest.raises(SystemExit):
        parse_cli("--version")
