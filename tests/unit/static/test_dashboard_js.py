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


def test_poll_treats_revoked_as_terminal(dashboard_js):
    """A cancelled task ends in REVOKED, not FAILURE. The poll loop must stop
    on it (flagged err.cancelled) instead of spinning until the 1-hour timeout."""
    body = extract_function_body(dashboard_js, "pollTaskStatus")
    assert '"REVOKED"' in body, "pollTaskStatus must treat REVOKED as terminal"
    assert re.search(r"\.cancelled\s*=\s*true", body), (
        "cancellation must be distinguishable from failure (err.cancelled) so "
        "callers don't show a 'Transcription failed' notification for a user stop"
    )


def test_cancel_transcription_resets_stale_status(dashboard_js):
    """When /tasks/active has no lock for the recording (worker died mid-task,
    lock TTL expired), the stop button must still clear the stale DB status via
    POST /transcribe/{name}/reset — otherwise the row shows "Transcribing…"
    forever with no UI path out."""
    body = extract_function_body(dashboard_js, "cancelTranscription")
    assert re.search(r"/reset`?\s*,\s*{\s*method:\s*\"POST\"", body), (
        "cancelTranscription must POST to the /transcribe/{name}/reset endpoint "
        "when no active task is found for the recording"
    )


def test_speaker_editor_offers_voice_enrollment(dashboard_js):
    """Rows for original "Speaker N" labels must render the Remember-voice
    checkbox; voiceprints are stored under those labels, so enrollment is
    impossible for already-renamed speakers."""
    body = extract_function_body(dashboard_js, "renderSpeakerEditor")
    assert "transcript-speaker-enroll" in body, (
        "renderSpeakerEditor must render the transcript-speaker-enroll checkbox"
    )
    assert re.search(r"\^Speaker \\d\+\$", body), (
        "the enroll checkbox must be limited to original 'Speaker N' labels"
    )


def test_apply_handler_reads_enroll_checkboxes_before_rerender(dashboard_js):
    """renderSpeakerEditor() wipes the checkbox states; the apply handler must
    collect enrollments before re-rendering or 'Remember voice' silently does
    nothing."""
    read_pos = dashboard_js.index(".transcript-speaker-enroll[data-label=")
    rerender_pos = dashboard_js.index("renderSpeakerEditor(newTranscript)")
    assert read_pos < rerender_pos, (
        "the apply handler must read the enroll checkboxes before calling "
        "renderSpeakerEditor(newTranscript), which destroys them"
    )
    assert re.search(r"/enroll`?\s*,\s*{\s*method:\s*\"POST\"", dashboard_js), (
        "checked speakers must be enrolled via POST /speakers/enroll"
    )


def test_voices_modal_can_apply_profiles_retroactively(dashboard_js):
    """The Voices modal's apply button must POST /speakers/apply so past
    transcripts get renamed, and must ask for confirmation first (it rewrites
    stored transcripts)."""
    assert re.search(r"/apply`?\s*,\s*{\s*method:\s*\"POST\"", dashboard_js), (
        "the voices-apply button must POST to /speakers/apply"
    )
    apply_pos = dashboard_js.index("voicesApplyBtn.addEventListener")
    confirm_pos = dashboard_js.index("confirm(", apply_pos)
    fetch_pos = dashboard_js.index("/apply`", apply_pos)
    assert confirm_pos < fetch_pos, (
        "the retroactive apply must be confirmed before the POST fires"
    )


def test_transcribing_row_has_stop_button(dashboard_js):
    """The in-flight row indicator is a disabled button; without a companion
    stop button there is no UI path to DELETE /tasks/status/{task_id}."""
    body = extract_function_body(dashboard_js, "actionButtons")
    assert "btn-cancel-transcribe" in body, (
        "the Transcribing… indicator must render a btn-cancel-transcribe stop button"
    )
