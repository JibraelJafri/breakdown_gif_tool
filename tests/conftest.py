"""Pytest configuration and shared test fixtures for breakdown-animator."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.generate_test_assets import generate_all_fixtures


@pytest.fixture(scope="session")
def global_fixtures(tmp_path_factory) -> dict[str, Path]:
    """Generates a complete session-scoped set of synthetic breakdown test fixtures."""
    fixtures_dir = tmp_path_factory.mktemp("breakdown_fixtures")
    return generate_all_fixtures(fixtures_dir, verbose=False)
