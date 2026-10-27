import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """With EMMY_REQUIRE_ALL=1 a skipped test fails: a release check must not
    pass because the runner (EMMY_RUNNER) or the problem folders were missing."""
    outcome = yield
    report = outcome.get_result()
    if report.skipped and os.environ.get("EMMY_REQUIRE_ALL") == "1":
        report.outcome = "failed"
        report.longrepr = f"skipped, but EMMY_REQUIRE_ALL=1 requires every test to run: {report.longrepr}"
