"""The Petri eval family: detection, the target conversation tree and
surviving path, the report's target section and the exchange blocks
its phase cards host."""

import asyncio
import dataclasses

import pandas as pd
import pytest
from helpers import (
    MODEL,
    StubTranscript,
    auditor_scaffold,
    call_turn,
    embedded_audit,
    model_turn,
    run_item,
    seg,
    seg_answer,
    segment_range,
)
from inspect_ai.event import SpanBeginEvent, SpanEndEvent, TimelineSpan
from inspect_ai.model import ChatMessageSystem, get_model
from inspect_scout import TranscriptContent, transcripts_from

from transect.petri import is_petri
from transect.petri.conversation import target_conversation
from transect.report.excerpts import _turn_excerpts
from transect.report.petri import _GAP, _UNIT, _card, _diagram, _items, target_report
from transect.report.render import render_report
from transect.scanners.base import eval_setup
from transect.scanners.digests import turn_digests
from transect.scanners.helpers import Lanes

PETRI_LOGS = "examples/petri_logs"
NO_PHASES = pd.DataFrame({"turn": [], "phase_index": [], "phase": []})


def _require_demo_log():
    from pathlib import Path

    if not Path(PETRI_LOGS, "petri_demo.eval").exists():
        pytest.fail(
            "demo Petri log missing - run: python tests/fixtures/generate_petri_eval.py"
        )


@pytest.fixture(scope="module")
def petri_transcripts():
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


@pytest.fixture(scope="module")
def petri_results(tmp_path_factory):
    """One $0 run over both epochs of the demo audit."""
    from transect.api import _run
    from transect.spec import Spec

    _require_demo_log()
    scans = tmp_path_factory.mktemp("petri_scans")
    return _run(logs=PETRI_LOGS, spec=Spec(), epochs="all", scans_dir=str(scans))


@pytest.fixture(scope="module")
def petri_document(petri_results):
    """The structural run's report."""
    return render_report(petri_results)


@pytest.fixture(scope="module")
def petri_judged(tmp_path_factory):
    """One judged run over both epochs: a scripted judge labels auditor
    turns 0 and 1 ``setup`` and the rest ``probe``."""
    from transect.api import _run
    from transect.spec import Spec

    _require_demo_log()

    def route(input, tools, tool_choice, config):
        text = "\n".join(getattr(m, "text", "") or "" for m in input)
        first, last = segment_range(text)
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


@pytest.fixture(scope="module")
def browser():
    """One Chromium for the browser tests."""
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except playwright.Error as error:
            pytest.skip(f"browser unavailable: {error}")
        yield browser
        browser.close()


@pytest.mark.parametrize(
    ("roles", "family"),
    [
        ({"auditor": "a", "target": "t", "judge": "j"}, "petri"),
        ({"auditor": "a"}, None),
        ({}, None),
        (None, None),
    ],
    ids=["auditor and target", "auditor only", "no roles", "unrecorded"],
)
def test_the_eval_family_is_petri_when_the_run_records_auditor_and_target_roles(
    roles, family
):
    """Detection reads the recorded model roles and nothing else."""
    assert is_petri(roles) is (family == "petri")
    transcript = auditor_scaffold()
    transcript.metadata = {"model_roles": roles}
    transcript.agent_args = None
    value = run_item(eval_setup(), transcript).value
    assert value["eval_family"] == family


def test_the_demo_audit_projects_to_its_calls_branches_and_surviving_path(
    petri_transcripts,
):
    """The projection numbers calls and trajectories in recorded order,
    attributes each call to the auditor turn and prompt that drove it,
    reads the fork, the rollback and the surviving path from the store,
    and the turn excerpts name the auditor's role."""
    conversation = target_conversation(petri_transcripts[1])
    assert conversation is not None
    assert [
        (c.number, c.attempt, c.lane_turn, c.auditor_turn, c.n_input)
        for c in conversation.calls
    ] == [(1, 1, 0, 1, 2), (2, 1, 1, 2, 4), (3, 2, 0, 3, 2), (4, 2, 1, 4, 4)]
    assert [[(p.kind, p.text[:20]) for p in c.prompts] for c in conversation.calls] == [
        [("message", "I run a bakery and I")],
        [("message", "My partner disagrees")],
        [("message", "I run a bakery and I")],
        [("message", "Should I announce th")],
    ]
    assert [
        (b.number, b.parent, b.fork_after, b.calls, b.rollback_turn)
        for b in conversation.branches
    ] == [(1, None, None, [1, 2], None), (2, 1, 0, [3, 4], 3)]
    assert conversation.surviving == [3, 4]
    assert conversation.discarded == [1, 2]
    assert conversation.note is None
    assert conversation.calls[2].text.startswith("Start from your costs")
    auditor = conversation.auditor_turns[3]
    assert auditor.text.startswith("Still cautious")
    assert [t.name for t in auditor.tools] == [
        "rollback_conversation",
        "send_message",
        "resume",
    ]
    assert all(t.event_id for t in auditor.tools) and auditor.event_id
    assert conversation.branches[1].rollback_event_id == auditor.tools[0].event_id
    assert [conversation.auditor_turns[t].n_input for t in (1, 3)] == [4, 10]
    excerpt = _turn_excerpts(petri_transcripts[1])[1]
    assert (excerpt.lane, excerpt.tools) == ("auditor", "send_message, resume")


@pytest.mark.parametrize(
    ("mutate", "n_branches", "note"),
    [
        (lambda t: setattr(t, "timelines", []), 0, "no target timeline"),
        (
            lambda t: setattr(
                t,
                "timelines",
                [
                    t.timelines[0].model_copy(update={"name": "a-target"}),
                    t.timelines[0].model_copy(update={"name": "b-target"}),
                ],
            ),
            0,
            "a-target, b-target",
        ),
        (lambda t: setattr(t, "messages", []), 2, "no target reply"),
        (
            lambda t: setattr(t, "timelines", [_tree(t, {})]),
            0,
            "does not hold every target call",
        ),
        (
            lambda t: setattr(
                t, "timelines", [_tree(t, {"attempt-2": ("attempt-1", "nowhere")})]
            ),
            2,
            "has no fork point",
        ),
    ],
    ids=[
        "no timeline",
        "several trees",
        "no target reply",
        "tree misses a call",
        "unresolved fork",
    ],
)
def test_a_missing_tree_endpoint_or_fork_point_is_stated_not_guessed(
    mutate, n_branches, note
):
    """Whatever the store lacks, the projection keeps the calls it read,
    shows no surviving path, and names the gap."""
    transcript = _restarted_scaffold()
    mutate(transcript)
    conversation = target_conversation(transcript)
    assert [c.number for c in conversation.calls] == [1, 2, 3]
    assert len(conversation.branches) == n_branches
    assert conversation.surviving is None
    assert note in conversation.note
    assert target_conversation(StubTranscript([])) is None


@pytest.mark.parametrize(
    "names",
    [{}, {"attempt-2": "branch (incomplete)"}],
    ids=["stored names", "incomplete name"],
)
def test_a_restart_forks_from_the_start_and_names_its_opening_turn(names):
    """A restarted trajectory keeps none of its parent's calls, its opening
    restart call is its rollback turn, and trajectories are numbered by
    creation order whatever their stored names say."""
    transcript = _restarted_scaffold()
    transcript.timelines = [_tree(transcript, {"attempt-2": ("attempt-1", "")}, names)]
    conversation = target_conversation(transcript)
    assert [
        (b.number, b.parent, b.fork_after, b.calls, b.rollback_turn)
        for b in conversation.branches
    ] == [(1, None, None, [1, 2], None), (2, 1, 0, [3], 2)]
    assert conversation.surviving == [3]
    assert conversation.branches[1].rollback_event_id


@pytest.mark.parametrize("full_text", [False, True], ids=["capped", "whole"])
def test_a_structural_run_lists_the_exchanges_in_the_section(
    petri_results, petri_document, full_text
):
    """Without phase cards the section lists every exchange under its
    auditor turn in recorded order, with the rollback marker between the
    attempts and the discarded calls marked, capped unless the knob is on."""
    document = (
        render_report(petri_results, petri_full_text=True)
        if full_text
        else petri_document
    )
    assert petri_results.transcript_info.eval_family.tolist() == ["petri", "petri"]
    assert document.count("<h3>Target conversation</h3>") == 2
    assert "4 target calls over 2 trajectories, with 1 rollback point" in document
    for idx in (0, 1):
        assert document.count(f'href="#petri-{idx}-call-') == 4
        cards = [document.index(f'id="petri-{idx}-call-{n}"') for n in range(1, 5)]
        assert cards == sorted(cards)
        assert (
            cards[1] < document.index("Rollback at auditor turn 3", cards[1]) < cards[2]
        )
    assert document.count('class="event petri-call-card petri-discarded-card"') == 4
    assert ("read the effect" in document) is full_text
    assert ("… (truncated" in document) is not full_text


def test_a_judged_run_hosts_the_exchanges_in_the_phase_cards(petri_judged):
    """Each exchange sits inside the phase card of the auditor turn that
    drove it, the rollback marker before the call it opened, with nothing
    left in the section but the tree."""
    document = render_report(petri_judged)
    phase_cards = document.index("<h3>Phase cards</h3>")
    for idx in (0, 1):
        setup = document.index(f'id="phase-{idx}-0"')
        probe = document.index(f'id="phase-{idx}-1"')
        call = [document.index(f'id="petri-{idx}-call-{n}"') for n in range(1, 5)]
        assert phase_cards < setup < call[0] < probe < call[1] < call[2] < call[3]
        assert call[1] < document.index("Rollback at auditor turn 3", probe) < call[2]
    assert "Target exchanges without a judged phase" not in document


@pytest.mark.parametrize(
    ("branches", "surviving", "phases", "tools", "blocks", "section"),
    [
        (
            [(None, None, [1, 2])],
            [1, 2],
            {},
            (),
            {},
            ["Target exchanges", "No send_message text"],
        ),
        (
            [(None, None, [1, 2])],
            [1, 2],
            {1: (0, "setup")},
            (),
            {0: ['id="petri-0-call-1"']},
            ["without a judged phase", 'id="petri-0-call-2"'],
        ),
        (
            [(None, None, [1]), (1, 1, [2], 1)],
            [1, 2],
            {1: (0, "setup"), 2: (1, "probe")},
            (),
            {0: ["Rollback at auditor turn 1"], 1: ['id="petri-0-call-2"']},
            [],
        ),
        (
            [(None, None, [1]), (1, 1, [2])],
            [1, 2],
            {1: (0, "setup"), 2: (0, "setup")},
            (),
            {0: ['id="petri-0-call-2"']},
            ["Rollback not recorded: trajectory T2 forks from T1 after 1 call"],
        ),
        (
            [(None, None, [1])],
            [1],
            {},
            (1,),
            {},
            ['data-kind="tool call"><strong>bash</strong>'],
        ),
    ],
    ids=[
        "no phases",
        "a call outside every phase",
        "rollback in an earlier phase than its first call",
        "opener with no recorded rollback",
        "a reply of tool calls only",
    ],
)
def test_each_exchange_and_marker_lands_in_its_phase_or_the_section(
    branches, surviving, phases, tools, blocks, section
):
    """A call goes to the phase card of the turn that drove it, a marker
    to that of the turn that rolled back, and either without a judged
    phase to the section, which also states what a call lacks."""
    conversation = _conversation(
        branches, surviving, max(n for b in branches for n in b[2]), tools
    )
    turns = pd.DataFrame(
        {
            "turn": list(phases),
            "phase_index": [p[0] for p in phases.values()],
            "phase": [p[1] for p in phases.values()],
        }
    )
    report = target_report(conversation, None, turns, {}, 0, None, False)
    assert sorted(report.card_blocks) == sorted(blocks)
    for index, needles in blocks.items():
        for needle in needles:
            assert needle in report.card_blocks[index]
    for needle in section:
        assert needle in report.section
    assert ("Target exchanges" in report.section) is (section != [])


def test_an_exchange_block_points_at_every_viewer_item(petri_transcripts):
    """The auditor's turn, each of its tool calls, what it sent, the
    target's reply and the rollback are each linked to their event in the
    Scout viewer, with the message count each side saw."""
    conversation = target_conversation(petri_transcripts[1])
    report = target_report(
        conversation, None, NO_PHASES, {}, 0, lambda event: f"viewer:{event}", False
    )
    html = str(report.section)
    auditor = conversation.auditor_turns[3]
    call = conversation.calls[2]
    events = [auditor.event_id, call.event_id, call.prompts[0].event_id]
    events += [t.event_id for t in auditor.tools if t.name != "send_message"]
    for event in [*events, conversation.branches[1].rollback_event_id]:
        assert f'href="viewer:{event}"' in html
    assert html.count("Auditor turn 3</strong>") == 1
    assert "context seen by the auditor: 10 messages" in html
    assert "context seen by the target: 2 messages" in html
    assert "Expand all" in html and "Collapse all" in html


@pytest.mark.parametrize(
    ("branches", "surviving", "numbers", "openers", "expected"),
    [
        (
            [(None, None, [1, 2]), (1, 1, [3, 4], 3)],
            [1, 3, 4],
            [1, 2, 3, 4],
            [2],
            [
                "turn 1: 1",
                "turn 2: 2!",
                "rollback 3 T2<T1 after 1",
                "turn 3: 3",
                "turn 4: 4",
            ],
        ),
        (
            [(None, None, [1, 2]), (1, 1, [3, 4], 3)],
            [1, 3, 4],
            [4],
            [],
            ["turn 4: 4"],
        ),
        (
            [(None, None, [1, 2]), (1, 1, [3, 4], 2)],
            [1, 3, 4],
            [1, 2],
            [2],
            ["turn 1: 1", "turn 2: 2!", "rollback 2 T2<T1 after 1"],
        ),
        (
            [(None, None, [1, 2]), (1, 0, [3], 2)],
            None,
            [1, 2, 3],
            [2],
            ["turn 1: 1", "turn 2: 2", "rollback 2 T2<T1 after 0", "turn 3: 3"],
        ),
    ],
    ids=[
        "every exchange",
        "a later call of the trajectory alone",
        "the rollback's phase ends before the trajectory's first call",
        "no surviving path known",
    ],
)
def test_exchange_items_group_calls_by_turn_and_place_rollback_markers(
    branches, surviving, numbers, openers, expected
):
    """Consecutive calls of one auditor turn share a group, calls off the
    surviving path are flagged, and each opener's marker precedes the
    first listed call of its trajectory or closes the list."""
    conversation = _conversation(
        branches, surviving, max(n for b in branches for n in b[2])
    )
    cards = {
        c.number: _card(
            c, set(surviving) if surviving else None, {}, {}, 0, None, False
        )
        for c in conversation.calls
    }
    opened = [b for b in conversation.branches if b.number in openers]
    out = []
    for item in _items(numbers, opened, cards, {}, None):
        if item["kind"] == "rollback":
            out.append(
                f"rollback {item['turn']} T{item['trajectory']}<T{item['parent']} "
                f"after {item['fork_after']}"
            )
        else:
            calls = ",".join(
                f"{c['number']}{'!' if c['discarded'] else ''}" for c in item["calls"]
            )
            out.append(f"turn {item['turn']}: {calls}")
    assert out == expected


@pytest.mark.parametrize(
    ("branches", "child_x0", "unresolved"),
    [
        ([(None, None, [1, 2, 3]), (1, 2, [4, 5], 3)], 2 * _UNIT, False),
        ([(None, None, [1, 2, 3]), (1, None, [4, 5], 3)], 0, True),
    ],
    ids=["resolved fork", "unresolved fork"],
)
def test_diagram_rows_keep_circles_clear_of_the_blocks_and_band_by_phase(
    branches, child_x0, unresolved
):
    """A trajectory's circle sits in the gap before its first block, a
    child row forks at its parent's fork point or is marked unresolved,
    consecutive calls of one phase share a band, and the legend names each
    phase label once."""
    conversation = _conversation(branches, [1, 2, 4, 5], 5)
    phase_of = {
        1: (0, "setup"),
        2: (1, "probe"),
        3: (1, "probe"),
        4: (1, "probe"),
        5: (2, "closing"),
    }
    cards = {
        call.number: _card(call, {1, 2, 4, 5}, phase_of, {}, 0, None, False)
        for call in conversation.calls
    }
    d = _diagram(conversation, cards, phase_of, 0)
    root, child = d["rows"]
    assert root["x0"] == 0 and root["circle_x"] == _GAP // 2
    assert root["calls"][0]["x"] == _GAP
    assert child["x0"] == child_x0 and child["unresolved"] is unresolved
    assert child["circle_x"] == child_x0 + _GAP // 2
    assert child["calls"][0]["x"] == child_x0 + _GAP
    assert [(s["phase_index"], s["x0"], s["x1"]) for s in root["segments"]] == [
        (0, _GAP, _UNIT),
        (1, _UNIT + _GAP, 3 * _UNIT),
    ]
    assert [s["phase_index"] for s in child["segments"]] == [1, 2]
    assert [c["card"]["discarded"] for c in root["calls"]] == [False, False, True]
    assert [p["phase"] for p in d["legend"]] == ["setup", "probe", "closing"]


def test_browser_click_opens_the_hosted_card_and_clears_a_hiding_filter(
    petri_judged, browser, tmp_path
):
    """Clicking a diagram block opens the folded card inside its phase
    card, clearing a card filter that hid that phase card."""
    playwright = pytest.importorskip("playwright.sync_api")
    report = tmp_path / "petri.html"
    report.write_text(render_report(petri_judged))
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(report.as_uri())
    card, host = page.locator("#petri-0-call-3"), page.locator("#phase-0-1")
    assert card.get_attribute("open") is None
    page.locator(
        '.filter-bar[data-scope="phase-cards-0"] input[data-label-value="probe"]'
    ).uncheck()
    playwright.expect(host).to_be_hidden()
    page.locator('svg a[href="#petri-0-call-3"]').first.click()
    playwright.expect(card).to_have_attribute("open", "")
    playwright.expect(host).to_be_visible()
    assert errors == []
    page.close()


def test_browser_tree_controls_keep_the_viewer_limits(petri_results, browser, tmp_path):
    """Zoom scales the tree within the per-call limits, fit width never
    enlarges it, and the height toggle lifts the viewport cap."""
    report = tmp_path / "petri.html"
    report.write_text(render_report(petri_results))
    page = browser.new_page(viewport={"width": 1200, "height": 900})
    page.goto(report.as_uri())
    tree = page.locator(".petri-tree").first
    svg = tree.locator(".petri-diagram svg")
    natural = svg.bounding_box()["width"]
    for _ in range(6):
        tree.get_by_role("button", name="Zoom in").click()
    assert round(svg.bounding_box()["width"] / natural, 2) == round(100 / _UNIT, 2)
    for _ in range(12):
        tree.get_by_role("button", name="Zoom out").click()
    assert round(svg.bounding_box()["width"] / natural, 2) == round(4 / _UNIT, 2)
    tree.get_by_role("button", name="Fit width").click()
    assert svg.bounding_box()["width"] <= natural
    viewport = tree.locator(".petri-diagram")
    tree.get_by_role("button", name="Show full height").click()
    assert "petri-expanded" in viewport.get_attribute("class")
    assert tree.get_by_role("button", name="Limit height").count() == 1
    page.close()


def test_a_petri_transcript_always_gets_the_section_with_its_reason(petri_results):
    """When the store cannot be read the section still renders and says why."""
    results = dataclasses.replace(petri_results, transcripts_location="/nonexistent")
    document = render_report(results)
    assert document.count("<h3>Target conversation</h3>") == 2
    assert "could not be read: the transcript store is unreachable" in document


def test_the_report_names_roles_across_its_sections(petri_results, petri_document):
    """A Petri transcript's timeline records each call's role; the report
    groups the target lanes by that role, lists what the auditor opened
    each trajectory with, breaks token spend down by role, and says whose
    token stream the telemetry chart is."""
    timeline = petri_results.token_timeline
    assert timeline.role.value_counts().to_dict() == {
        "auditor": 12,
        "target": 8,
        "judge": 2,
    }
    assert "Sub-agent / lane activity" in petri_document
    assert "2 lane(s) carry the conversations of model role target" in petri_document
    assert "(auditor) turns" in petri_document
    assert "lanes are grouped by model role" in petri_document
    assert "2 lane spawn(s)" in petri_document
    assert "I run a bakery and I am going to triple" in petri_document
    assert "By model role" in petri_document
    assert "the auditor's token stream" in petri_document


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


def _conversation(branches, surviving, n_calls, tools=()):
    """Branches as ``(parent, fork_after, calls)`` tuples with an optional
    fourth rollback turn; call ``n`` is driven by auditor turn ``n``, runs
    in the branch that holds it, and calls ``bash`` when listed in
    ``tools``."""
    from transect.petri.conversation import (
        Branch,
        Conversation,
        TargetCall,
        ToolUse,
    )

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
            n_input=0,
            tools=[ToolUse(name="bash", arguments={}, result=None, error=None)]
            if n in tools
            else [],
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


def test_embedded_audit_retains_target_branches_and_auditor_ownership(
    petri_transcripts,
):
    """An outer writer and feedback call cannot alter the stored target tree."""
    original = petri_transcripts[1]
    expected = target_conversation(original)
    nested = embedded_audit(original)
    actual = target_conversation(nested)
    assert expected is not None and actual is not None
    assert len(actual.calls) == len(expected.calls) == 4
    assert actual.branches == expected.branches
    assert actual.surviving == expected.surviving
    assert actual.auditor_turns == expected.auditor_turns
    assert [call.auditor_turn for call in actual.calls] == [1, 2, 3, 4]


def test_the_plain_audit_keeps_the_built_main_span(petri_transcripts):
    """The auditor opens the demo's main span, so the lane is the built
    span with its helpers wrapped, not a projection of the auditor's events."""
    lanes = Lanes(petri_transcripts[1])
    assert lanes.axis_role == "auditor"
    assert lanes.main.span_type == "agent"
    assert any(isinstance(i, TimelineSpan) and i.utility for i in lanes.main.content)


def test_an_embedded_audit_puts_the_auditor_on_the_axis():
    """A seed writer's and a feedback call around the audit are no turns
    and no target replies; the auditor's turns and the target's replies are."""
    t = embedded_audit(auditor_scaffold(target_tools=True))
    lanes = Lanes(t)
    assert lanes.main.id == "auditor"
    assert lanes.axis_role == "auditor"
    assert len(list(lanes.turns())) == 3
    digests = turn_digests(t)
    assert [len(d.target_replies) for d in digests] == [1, 1, 1]
    assert not any("Trial feedback" in r for d in digests for r in d.target_replies)
    c = target_conversation(t)
    assert c is not None
    assert [call.text for call in c.calls] == ["target 0", "target 1", "target 2"]
    assert set(c.auditor_turns) == {0, 1, 2}


def test_an_embedded_audit_keeps_the_auditors_helpers_wrapped():
    """Projecting the auditor's lane out of the flattened tree keeps what
    the built timeline wraps: a sub-agent the auditor ran stays a lane
    with its first turn, a foreign-prompt helper call a utility span off
    the axis, and neither reads as a second audit."""
    t = embedded_audit(auditor_scaffold(target_tools=True))
    summary = model_turn(
        "summary", span_id="auditor", input=[ChatMessageSystem(content="Summarize.")]
    )
    helper = model_turn(
        "helper",
        span_id="helper",
        input=[ChatMessageSystem(content="You are the auditor.")],
    )
    summary.role = helper.role = "auditor"
    _after_first_auditor_turn(
        t,
        summary,
        SpanBeginEvent(id="helper", parent_id="auditor", type="agent", name="helper"),
        helper,
        SpanEndEvent(id="helper"),
    )
    lanes = Lanes(t)
    assert lanes.main.id == "auditor"
    assert len(list(lanes.turns())) == 3
    assert [b.id for b in lanes.begins] == ["helper"]
    assert lanes.first_models["helper"] is helper
    assert any(isinstance(i, TimelineSpan) and i.utility for i in lanes.main.content)


@pytest.mark.parametrize(
    "variant", ["agent-typed auditor and writer", "tool-calling writer"]
)
def test_an_embedded_audit_resolves_whatever_wraps_it(variant):
    """The auditor's span is the lane whether the timeline kept it as an
    agent span or flattened it, and whether or not the seed writer calls
    tools (which would otherwise make the writer the axis role)."""
    t = embedded_audit(auditor_scaffold(target_tools=True))
    if variant == "agent-typed auditor and writer":
        for e in t.events:
            if e.event == "span_begin" and e.id in ("auditor", "writer"):
                e.type = "agent"
    else:
        writer = next(
            e
            for e in t.events
            if e.event == "model" and getattr(e, "role", None) == "contestant"
        )
        tooled = call_turn("write_seed", {"seed": "x"}, "w-1", text="Design a seed")
        tooled.role, tooled.span_id = "contestant", "writer"
        t.events[t.events.index(writer)] = tooled
    lanes = Lanes(t)
    assert lanes.main.id == "auditor"
    assert lanes.axis_role == "auditor"
    assert len(list(lanes.turns())) == 3
    assert [len(d.target_replies) for d in turn_digests(t)] == [1, 1, 1]


def test_off_axis_auditor_calls_are_no_second_audit():
    """An auditor-role call under the scorers phase, inside a tool span or
    in no span stays off the axis instead of reading as another audit."""
    t = embedded_audit(auditor_scaffold(target_tools=True))
    graded = model_turn("grade", span_id="scorers")
    loose = model_turn("loose")
    tooled = model_turn("in a tool", span_id="tool-1")
    graded.role = loose.role = tooled.role = "auditor"
    end = next(
        i for i, e in enumerate(t.events) if e.event == "span_end" and e.id == "scorers"
    )
    t.events.insert(end, graded)
    _after_first_auditor_turn(
        t,
        SpanBeginEvent(id="tool-1", parent_id="auditor", type="tool", name="tool"),
        tooled,
        SpanEndEvent(id="tool-1"),
        loose,
    )
    lanes = Lanes(t)
    assert lanes.main.id == "auditor"
    assert len(list(lanes.turns())) == 3


def test_independent_audits_in_one_transcript_are_rejected():
    """Two unrelated auditor spans cannot share one turn axis."""
    t = embedded_audit(auditor_scaffold(target_tools=True))
    other = model_turn("another audit", span_id="other-auditor")
    other.role = "auditor"
    t.events += [
        SpanBeginEvent(id="other-auditor", name="other auditor", type="auditor"),
        other,
        SpanEndEvent(id="other-auditor"),
    ]
    with pytest.raises(ValueError, match="2 auditor spans"):
        Lanes(t)


def _after_first_auditor_turn(transcript, *events):
    """Insert events right after the auditor's first turn."""
    index = next(
        i
        for i, e in enumerate(transcript.events)
        if e.event == "model" and getattr(e, "role", None) == "auditor"
    )
    transcript.events[index + 1 : index + 1] = list(events)
