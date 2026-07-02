"""Regression tests for the summarize flow in static/dashboard.js.

The frontend is vanilla JS with no build step or JS test runner, so these
tests assert source-level invariants instead of executing the code.

Bug they guard against: POST /summarize queues a Celery task and returns only
a task_id — never a "summary" field. The completion handler used to fall back
to formatMarkdown(data.summary) when GET /summaries came back not-ok, which
crashed with "Cannot read properties of undefined (reading 'replace')" and
masked the backend's real error message.
"""

import re
from pathlib import Path

import pytest

DASHBOARD_JS = Path(__file__).resolve().parents[3] / "src" / "static" / "dashboard.js"


@pytest.fixture(scope="module")
def dashboard_js() -> str:
    return DASHBOARD_JS.read_text(encoding="utf-8")


def extract_function_body(source: str, name: str) -> str:
    """Return the body of a `function name(...) { ... }` declaration."""
    match = re.search(rf"function {re.escape(name)}\s*\([^)]*\)\s*{{", source)
    assert match, f"function {name}(...) not found in dashboard.js"
    depth = 1
    start = match.end()
    for i in range(start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i]
    raise AssertionError(f"unbalanced braces extracting function {name}")


def test_format_markdown_guards_null_input(dashboard_js):
    """formatMarkdown must coerce a missing summary to "" before chaining
    .replace, otherwise a null/undefined summary throws a TypeError."""
    body = extract_function_body(dashboard_js, "formatMarkdown")
    assert re.search(r"return\s*\(\s*text\s*\|\|\s*(\"\"|'')\s*\)", body), (
        "formatMarkdown must start from (text || \"\") — calling .replace "
        "directly on its argument crashes when the summary is undefined"
    )


def test_summarize_handler_never_renders_data_summary(dashboard_js):
    """The summarize POST response contains only a task_id (the work runs on
    Celery), so the handler must not try to render data.summary from it."""
    assert "formatMarkdown(data.summary)" not in dashboard_js, (
        "the summarize handler must not fall back to data.summary — the "
        "queued-task response never carries that field"
    )


def test_summarize_handler_surfaces_summaries_fetch_error(dashboard_js):
    """When GET /summaries returns ok:false after the task completes, the
    modal must show the backend's error instead of rendering a summary."""
    assert re.search(r"summariesData\.error", dashboard_js), (
        "the non-ok /summaries branch must surface summariesData.error so "
        "the real failure reason isn't masked"
    )
