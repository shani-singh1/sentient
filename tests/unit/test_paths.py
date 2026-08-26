"""Unit tests for src/common/paths.py."""
from __future__ import annotations

import pytest

from src.common.paths import PROJECT_ROOT, ensure_dir, project_root

pytestmark = pytest.mark.unit


def test_project_root_contains_src_directory() -> None:
    assert (project_root() / "src").is_dir()


def test_project_root_is_stable_constant() -> None:
    assert PROJECT_ROOT == project_root()


def test_ensure_dir_creates_nested_directories(tmp_path) -> None:
    target = tmp_path / "a" / "b" / "c"
    assert not target.exists()

    result = ensure_dir(target)

    assert target.is_dir()
    assert result == target


def test_ensure_dir_is_idempotent_on_existing_directory(tmp_path) -> None:
    target = tmp_path / "already_there"
    target.mkdir()

    result = ensure_dir(target)

    assert result == target
    assert target.is_dir()
