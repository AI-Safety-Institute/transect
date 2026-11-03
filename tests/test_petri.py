"""The Petri eval family: detection, the target conversation tree and
surviving path, and the report's target conversation section."""

import asyncio

import pytest
from helpers import StubTranscript, auditor_scaffold, run_item
from inspect_scout import TranscriptContent, transcripts_from

from transect.petri import is_petri
from transect.petri.conversation import target_conversation
from transect.scanners.base import eval_setup

PETRI_LOGS = "examples/petri_logs"


def read_petri_transcripts():
    """The demo audit's transcripts as the report reads them, by epoch."""

    async def read():
        content = TranscriptContent(
            messages="all",
            events=["model", "span_begin", "span_end", "tool", "anchor", "branch"],
            timeline=True,
        )
        async with transcripts_from(PETRI_LOGS).reader() as reader:
            infos = [info async for info in reader.index()]
            return {
                info.task_repeat: await reader.read(info, content) for info in infos
            }

    return asyncio.run(read())


@pytest.mark.parametrize(
    ("roles", "family"),
    [
        ({"auditor": {"model": "a"}, "target": {"model": "t"}, "judge": {}}, "petri"),
        ({"auditor": {"model": "a"}}, None),
        ({}, None),
        (None, None),
    ],
    ids=["auditor and target", "auditor only", "no roles", "unrecorded"],
)
def test_the_eval_family_is_petri_when_the_run_records_auditor_and_target_roles(
    roles, family
):
    """Detection reads the run's recorded model roles, nothing else."""
    assert is_petri(roles) == (family == "petri")
    transcript = auditor_scaffold()
    transcript.metadata = {"model_roles": roles}
    transcript.agent_args = None
    value = run_item(eval_setup(), transcript).value
    assert value["eval_family"] == family


def test_the_demo_audit_projects_to_its_branches_and_surviving_path():
    """Each epoch of the scripted audit has four target calls on two
    branches; the rollback forks the second branch before any call of the
    first survives, so the last two calls are the surviving conversation."""
    conversation = target_conversation(read_petri_transcripts()[1])
    assert conversation is not None
    assert [
        (c.number, c.attempt, c.lane_turn, c.auditor_turn) for c in conversation.calls
    ] == [
        (1, 1, 0, 1),
        (2, 1, 1, 2),
        (3, 2, 0, 3),
        (4, 2, 1, 4),
    ]
    assert [
        (b.number, b.parent, b.fork_after, b.calls, b.rollback_turn)
        for b in conversation.branches
    ] == [
        (1, None, None, [1, 2], None),
        (2, 1, 0, [3, 4], 3),
    ]
    assert conversation.surviving == [3, 4]
    assert conversation.discarded == [1, 2]
    assert conversation.note is None
    assert conversation.calls[2].text.startswith("Start from your costs")
    assert conversation.calls[0].event_id


def test_a_conversation_without_a_stored_tree_or_endpoint_says_so():
    """No target timeline, or no surviving message to anchor on, is an
    explicit absence: the calls stay, the path and branches do not."""
    conversation = target_conversation(auditor_scaffold())
    assert conversation is not None
    assert [c.number for c in conversation.calls] == [1, 2, 3]
    assert conversation.branches == []
    assert conversation.surviving is None
    assert "timeline" in (conversation.note or "")
    assert target_conversation(StubTranscript([])) is None


@pytest.fixture(scope="module")
def petri_results(tmp_path_factory):
    """One $0 run over both epochs of the demo audit."""
    from transect.api import _run
    from transect.spec import Spec

    scans = tmp_path_factory.mktemp("petri_scans")
    return _run(logs=PETRI_LOGS, spec=Spec(), epochs="all", scans_dir=str(scans))


def test_the_report_draws_the_target_tree_and_the_surviving_conversation(
    petri_results,
):
    """A Petri transcript gets a target conversation section: the branch
    diagram linking each call to its card, the surviving calls open, and
    the discarded attempts folded at their fork, scoped per transcript."""
    from transect.report.render import render_report

    assert petri_results.transcript_info.eval_family.tolist() == ["petri", "petri"]
    document = render_report(petri_results)
    assert document.count("<h3>Target conversation</h3>") == 2
    for idx in (0, 1):
        assert document.count(f'href="#petri-{idx}-call-') == 4
        for number in range(1, 5):
            assert f'id="petri-{idx}-call-{number}"' in document
    earlier = document.index("Earlier attempt")
    assert earlier < document.index('id="petri-0-call-1"')
    assert document.index('id="petri-0-call-1"') < document.index('id="petri-0-call-3"')
    assert "Start from your costs" in document
    assert "… (truncated" in document
    assert "read the effect" not in document


def test_full_text_is_a_knob_that_is_off_by_default(petri_results):
    """With the knob on, a target reply renders whole instead of capped."""
    from transect.report.render import render_report

    document = render_report(petri_results, petri_full_text=True)
    assert "read the effect" in document
    assert "… (truncated" not in document


def test_a_report_without_petri_transcripts_has_no_target_conversation(tmp_path):
    """The section exists only for the Petri family."""
    from transect.api import _run
    from transect.report.render import render_report
    from transect.spec import Spec

    results = _run(logs="examples/logs", spec=Spec(), scans_dir=str(tmp_path))
    assert results.transcript_info.eval_family.isna().all()
    assert "Target conversation" not in render_report(results)


def test_browser_opens_a_call_card_from_the_diagram(petri_results, tmp_path):
    """Clicking a diagram segment opens the folded card it links to."""
    playwright = pytest.importorskip("playwright.sync_api")
    from transect.report.render import render_report

    report = tmp_path / "petri.html"
    report.write_text(render_report(petri_results))
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except playwright.Error as error:
            pytest.skip(f"browser unavailable: {error}")
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(report.as_uri())
        card = page.locator("#petri-0-call-1")
        assert card.get_attribute("open") is None
        page.locator('svg a[href="#petri-0-call-1"]').first.click()
        playwright.expect(card).to_have_attribute("open", "")
        assert errors == []
        browser.close()
