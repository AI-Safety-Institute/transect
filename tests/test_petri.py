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


def _require_demo_log():
    from pathlib import Path

    if not Path(PETRI_LOGS, "petri_demo.eval").exists():
        pytest.fail(
            "demo Petri log missing - run: python tests/fixtures/generate_petri_eval.py"
        )


def read_petri_transcripts():
    """The demo audit's transcripts as the report reads them, by epoch."""
    _require_demo_log()

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
    transcript = _restarted_scaffold()
    transcript.messages = []
    assert "no target reply" in (target_conversation(transcript).note or "")


@pytest.fixture(scope="module")
def petri_results(tmp_path_factory):
    """One $0 run over both epochs of the demo audit."""
    from transect.api import _run
    from transect.spec import Spec

    _require_demo_log()
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


def _tree(transcript, children, names=None):
    """A stored target timeline over the scaffold's trajectory spans:
    ``children`` maps a child span id to ``(parent span id, branched_from)``."""
    from inspect_ai.event import Timeline, TimelineEvent, TimelineSpan

    by_span: dict[str, list] = {}
    for event in transcript.events:
        if getattr(event, "role", None) == "target":
            by_span.setdefault(event.span_id, []).append(TimelineEvent(event=event))
    names = names or {}

    def span(span_id, branched_from=None):
        return TimelineSpan(
            id=span_id,
            name=names.get(span_id, f"branch {span_id[-1]}"),
            span_type="branch",
            branched_from=branched_from,
            content=by_span.get(span_id, []),
            branches=[
                span(child, anchor)
                for child, (parent, anchor) in children.items()
                if parent == span_id
            ],
        )

    return Timeline(name="target", description="", root=span("attempt-1"))


def _restarted_scaffold():
    """The scaffold with its second attempt opened by restart_conversation
    and the final conversation ending on the third target reply."""
    from inspect_ai.event import ToolEvent

    transcript = auditor_scaffold()
    events = transcript.events
    auditor = [e for e in events if e.event == "model" and e.role == "auditor"]
    auditor[2].output.message.tool_calls[0].function = "restart_conversation"
    for n, event in enumerate(
        e for e in events if e.event == "model" and e.role == "target"
    ):
        event.output.message.id = f"m{n}"
    begin = next(
        i
        for i, e in enumerate(events)
        if e.event == "span_begin" and e.id == "attempt-2"
    )
    events.insert(
        begin,
        ToolEvent(
            id="call-2", function="restart_conversation", arguments={}, result="ok"
        ),
    )
    last = [e for e in events if getattr(e, "role", None) == "target"][-1]
    transcript.messages = [last.output.message]
    transcript.timelines = [_tree(transcript, {"attempt-2": ("attempt-1", "")})]
    return transcript


def test_a_restart_forks_from_the_start_and_names_its_opening_turn():
    """A restarted trajectory keeps none of its parent's calls, and the
    restart tool call that opened it is its rollback turn; the branches are
    numbered by creation order whatever their stored names say."""
    transcript = _restarted_scaffold()
    conversation = target_conversation(transcript)
    assert [
        (b.number, b.parent, b.fork_after, b.calls, b.rollback_turn)
        for b in conversation.branches
    ] == [
        (1, None, None, [1, 2], None),
        (2, 1, 0, [3], 2),
    ]
    assert conversation.surviving == [3]
    transcript.timelines = [
        _tree(
            transcript,
            {"attempt-2": ("attempt-1", "")},
            {"attempt-2": "branch (incomplete)"},
        )
    ]
    assert [b.number for b in target_conversation(transcript).branches] == [1, 2]


def test_several_stored_target_trees_are_named_not_guessed():
    """Two audits in one sample store two target trees; the section says so
    instead of picking one."""
    transcript = _restarted_scaffold()
    tree = transcript.timelines[0]
    transcript.timelines = [
        tree.model_copy(update={"name": "a-target"}),
        tree.model_copy(update={"name": "b-target"}),
    ]
    conversation = target_conversation(transcript)
    assert conversation.branches == []
    assert "a-target" in conversation.note and "b-target" in conversation.note


def _conversation(branches, surviving, n_calls):
    from transect.petri.conversation import Branch, Conversation, TargetCall

    calls = [
        TargetCall(
            number=n,
            attempt=1,
            lane_turn=0,
            auditor_turn=n,
            event_id=None,
            message_id=None,
            model=None,
            text="",
            reasoning="",
        )
        for n in range(1, n_calls + 1)
    ]
    return Conversation(
        calls,
        [
            Branch(
                number=i + 1,
                parent=p,
                fork_after=f,
                calls=c,
                rollback_turn=None,
            )
            for i, (p, f, c) in enumerate(branches)
        ],
        surviving,
    )


@pytest.mark.parametrize(
    ("branches", "surviving", "expected"),
    [
        ([(None, None, [1, 2]), (1, 0, [3, 4])], [3, 4], {0: [[1, 2]]}),
        (
            [(None, None, [1, 2, 3, 4, 5]), (1, 4, [6]), (1, 1, [7, 8])],
            [1, 7, 8],
            {1: [[2, 3, 4, 5], [6]]},
        ),
        (
            [(None, None, [1, 2]), (1, 1, [3]), (2, 1, [4])],
            [1, 3],
            {1: [[2]], 2: [[4]]},
        ),
        ([(None, None, [1, 2]), (1, None, [3])], [1, 2], {None: [[3]]}),
    ],
    ids=[
        "fork before the first call",
        "tail and a late fork clamp to the departure",
        "a fork off a discarded branch",
        "an unrecorded fork",
    ],
)
def test_discarded_attempts_fold_at_the_fork_they_left_from(
    branches, surviving, expected
):
    """Each discarded call group lands at the surviving position its branch
    departed from; a fork past the path's departure folds at the departure;
    an unrecorded fork is set apart rather than placed."""
    from transect.report.petri import _alternatives

    conversation = _conversation(
        branches, surviving, max(n for _, _, c in branches for n in c)
    )
    out = _alternatives(conversation)
    assert {k: [a["calls"] for a in v] for k, v in out.items()} == expected


def test_a_petri_transcript_always_gets_the_section_with_its_reason(petri_results):
    """When the store cannot be read the section still renders and says why."""
    import dataclasses

    from transect.report.render import render_report

    results = dataclasses.replace(petri_results, transcripts_location="/nonexistent")
    document = render_report(results)
    assert document.count("<h3>Target conversation</h3>") == 2
    assert "could not be read: the transcript store is unreachable" in document


def test_the_report_names_roles_lists_lane_spawns_and_spends_by_role(petri_results):
    """A Petri transcript's timeline records each call's role; the report
    groups the target lanes by that role, lists what the auditor opened
    each trajectory with, and breaks token spend down by role."""
    from transect.report.render import render_report

    timeline = petri_results.token_timeline
    assert timeline.role.value_counts().to_dict() == {
        "auditor": 12,
        "target": 8,
        "judge": 2,
    }
    document = render_report(petri_results)
    assert "Sub-agent / lane activity" in document
    assert "2 lane(s) carry the conversations of model role target" in document
    assert "(auditor) turns" in document
    assert "lanes are grouped by model role" in document
    assert "2 lane spawn(s)" in document
    assert "I run a bakery and I am going to triple" in document
    assert "By model role" in document
