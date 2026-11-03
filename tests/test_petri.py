"""The Petri eval family: detection, the target conversation tree and
surviving path, and the report's target conversation section and
exchange cards."""

import asyncio
import re

import pytest
from helpers import MODEL, StubTranscript, auditor_scaffold, run_item, seg, seg_answer
from inspect_ai.model import get_model
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
    assert [c.prompt[:20] for c in conversation.calls] == [
        "I run a bakery and I",
        "My partner disagrees",
        "I run a bakery and I",
        "Should I announce th",
    ]


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


@pytest.fixture(scope="module")
def petri_judged(tmp_path_factory):
    """One judged run over both epochs: a scripted judge labels auditor
    turns 0 and 1 ``setup`` and the rest ``probe``."""
    from transect.api import _run
    from transect.spec import Spec

    _require_demo_log()

    def route(input, tools, tool_choice, config):
        text = "\n".join(getattr(m, "text", "") or "" for m in input)
        first, last = map(
            int, re.search(r"Segment turns (\d+)\.\.(\d+)", text).groups()
        )
        segments = [seg(first, min(1, last), "setup")] if first <= 1 else []
        if last > 1:
            segments.append(seg(max(first, 2), last, "probe"))
        return seg_answer(*segments)

    scans = tmp_path_factory.mktemp("petri_judged_scans")
    return _run(
        logs=PETRI_LOGS,
        spec=Spec.model_validate({"phases": ["setup", "probe"]}),
        epochs="all",
        scans_dir=str(scans),
        judge_models=get_model(MODEL, custom_outputs=route, memoize=False),
        verify=False,
        narrate=False,
    )


def test_a_structural_run_lists_the_exchanges_in_the_section(petri_results):
    """Without phase cards the target conversation section holds every
    exchange card in recorded order, a rollback marker where the auditor
    rolled back, the discarded calls marked, scoped per transcript."""
    from transect.report.render import render_report

    assert petri_results.transcript_info.eval_family.tolist() == ["petri", "petri"]
    document = render_report(petri_results)
    assert document.count("<h3>Target conversation</h3>") == 2
    assert "4 target calls over 2 trajectories, with 1 rollback point" in document
    assert "Click a call to open its card below" in document
    for idx in (0, 1):
        assert document.count(f'href="#petri-{idx}-call-') == 4
        cards = [document.index(f'id="petri-{idx}-call-{n}"') for n in range(1, 5)]
        assert cards == sorted(cards)
        marker = document.index("Rollback at auditor turn 3", cards[1])
        assert cards[1] < marker < cards[2]
    assert document.count('class="event petri-call-card petri-discarded-card"') == 4
    assert "Auditor sent" in document and "My partner disagrees" in document
    assert "Start from your costs" in document
    assert "… (truncated" in document
    assert "read the effect" not in document


def test_a_judged_run_hosts_the_exchanges_in_the_phase_cards(petri_judged):
    """Each exchange card sits inside the phase card of the auditor turn
    that drove it, with the rollback marker and the discarded mark kept;
    the section keeps the diagram, its phase segments and the description."""
    from transect.report.render import render_report

    document = render_report(petri_judged)
    phase_cards = document.index("<h3>Phase cards</h3>")
    for idx in (0, 1):
        setup = document.index(f'id="phase-{idx}-0"')
        probe = document.index(f'id="phase-{idx}-1"')
        assert phase_cards < setup < document.index(f'id="petri-{idx}-call-1"') < probe
        for number in (2, 3, 4):
            assert document.index(f'id="petri-{idx}-call-{number}"') > probe
        assert probe < document.index("Rollback at auditor turn 3", probe)
        assert (
            document.index(f'id="petri-{idx}-call-2"')
            < document.index("Rollback at auditor turn 3", probe)
            < document.index(f'id="petri-{idx}-call-3"')
        )
    assert document.count('class="petri-segment"') == 6
    assert ">P0<" in document and ">P1<" in document
    assert document.count('class="petri-row-label"') == 4
    assert ">T1<" in document and ">T2<" in document
    assert "Target conversation progress" in document
    assert "Click a call to open its card inside the auditor's phase card" in document


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


def test_browser_opens_a_call_card_from_the_diagram(petri_judged, tmp_path):
    """Clicking a diagram block opens the folded card it links to, inside
    its phase card."""
    playwright = pytest.importorskip("playwright.sync_api")
    from transect.report.render import render_report

    report = tmp_path / "petri.html"
    report.write_text(render_report(petri_judged))
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except playwright.Error as error:
            pytest.skip(f"browser unavailable: {error}")
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(report.as_uri())
        card = page.locator("#petri-0-call-3")
        assert card.get_attribute("open") is None
        page.locator('svg a[href="#petri-0-call-3"]').first.click()
        playwright.expect(card).to_have_attribute("open", "")
        playwright.expect(page.locator("#phase-0-1")).to_have_attribute("open", "")
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
    """Branches as ``(parent, fork_after, calls)`` tuples, with an optional
    fourth rollback turn; call ``n`` is driven by auditor turn ``n`` and
    runs in the branch that holds it."""
    from transect.petri.conversation import Branch, Conversation, TargetCall

    attempt_of = {n: i + 1 for i, b in enumerate(branches) for n in b[2]}
    calls = [
        TargetCall(
            number=n,
            attempt=attempt_of.get(n, 1),
            lane_turn=0,
            auditor_turn=n,
            event_id=None,
            message_id=None,
            model=None,
            text="",
            reasoning="",
            prompt="",
        )
        for n in range(1, n_calls + 1)
    ]
    return Conversation(
        calls,
        [
            Branch(
                number=i + 1,
                parent=b[0],
                fork_after=b[1],
                calls=b[2],
                rollback_turn=b[3] if len(b) > 3 else None,
            )
            for i, b in enumerate(branches)
        ],
        surviving,
    )


@pytest.mark.parametrize(
    ("branches", "surviving", "numbers", "expected"),
    [
        (
            [(None, None, [1, 2]), (1, 1, [3, 4], 3)],
            [1, 3, 4],
            [1, 2, 3, 4],
            ["1", "2 discarded", "rollback 3 T2<T1 after 1", "3", "4"],
        ),
        (
            [(None, None, [1, 2]), (1, 1, [3, 4], 3)],
            [1, 3, 4],
            [3, 4],
            ["rollback 3 T2<T1 after 1", "3", "4"],
        ),
        (
            [(None, None, [1, 2]), (1, 0, [3], 2)],
            None,
            [1, 2, 3],
            ["1", "2", "rollback 2 T2<T1 after 0", "3"],
        ),
        (
            [(None, None, [1, 2]), (1, None, [3])],
            [1, 2],
            [1, 2, 3],
            ["1", "2", "3 discarded"],
        ),
    ],
    ids=[
        "every exchange",
        "a phase's own exchanges",
        "no surviving path known",
        "an unrecorded rollback",
    ],
)
def test_exchange_items_mark_rollbacks_and_discarded_calls(
    branches, surviving, numbers, expected
):
    """The exchange list carries each call in order, flags the calls off
    the surviving path, and puts a rollback marker before the first call
    of each trajectory whose opening rollback is recorded."""
    from transect.report.petri import _items

    conversation = _conversation(
        branches, surviving, max(n for b in branches for n in b[2])
    )
    out = []
    for item in _items(conversation, numbers):
        if item["kind"] == "rollback":
            out.append(
                f"rollback {item['turn']} T{item['trajectory']}<T{item['parent']} "
                f"after {item['fork_after']}"
            )
        else:
            out.append(f"{item['number']}{' discarded' if item['discarded'] else ''}")
    assert out == expected


def test_diagram_rows_keep_circles_clear_of_the_blocks_and_segment_by_phase():
    """A trajectory's start circle sits in the gap before its first block,
    a child row forks at its parent's fork point, and consecutive calls of
    one phase share a coloured segment labelled with the phase number."""
    from transect.report.petri import _GAP, _UNIT, _diagram

    conversation = _conversation(
        [(None, None, [1, 2, 3]), (1, 2, [4, 5], 3)], [1, 2, 4, 5], 5
    )
    phase_of = {
        1: (0, "setup"),
        2: (1, "probe"),
        3: (1, "probe"),
        4: (1, "probe"),
        5: (2, "closing"),
    }
    cards = {
        n: {
            "id": f"c{n}",
            "number": n,
            "phase_index": phase_of[n][0],
            "phase": phase_of[n][1],
            "phase_href": f"#phase-0-{phase_of[n][0]}",
            "color": "#000",
        }
        for n in range(1, 6)
    }
    d = _diagram(conversation, cards, phase_of, 0)
    root, child = d["rows"]
    assert root["x0"] == 0 and child["x0"] == 2 * _UNIT
    assert root["circle_x"] == _GAP // 2 and child["circle_x"] == 2 * _UNIT + _GAP // 2
    assert root["calls"][0]["x"] == _GAP and child["calls"][0]["x"] == 2 * _UNIT + _GAP
    assert [(s["phase_index"], s["x0"], s["x1"]) for s in root["segments"]] == [
        (0, _GAP, _UNIT),
        (1, _UNIT + _GAP, 3 * _UNIT),
    ]
    assert [(s["phase_index"], s["x0"], s["x1"]) for s in child["segments"]] == [
        (1, 2 * _UNIT + _GAP, 3 * _UNIT),
        (2, 3 * _UNIT + _GAP, 4 * _UNIT),
    ]
    assert [c["discarded"] for c in root["calls"]] == [False, False, True]


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
