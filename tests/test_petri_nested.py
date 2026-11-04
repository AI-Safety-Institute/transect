"""Embedded Petri audits must use the auditor axis, not the seed writer."""

import pytest
from helpers import auditor_scaffold, embedded_audit, model_turn
from inspect_ai.event import SpanBeginEvent, SpanEndEvent

from transect.petri.conversation import target_conversation
from transect.scanners.digests import turn_digests
from transect.scanners.helpers import Lanes


def test_nested_axis_and_target_only_replies():
    """Only auditor turns and actual target replies enter the Petri surfaces."""
    t = embedded_audit(auditor_scaffold(target_tools=True))
    lanes = Lanes(t)
    assert lanes.main.id == "auditor"
    assert lanes.axis_role == "auditor"
    assert len(list(lanes.turns())) == 3
    digests = turn_digests(t)
    assert len(digests) == 3
    assert sum(len(d.target_replies) for d in digests) == 3
    assert all(
        "Trial feedback" not in reply for d in digests for reply in d.target_replies
    )
    c = target_conversation(t)
    assert len(c.calls) == 3
    assert set(c.auditor_turns) == {0, 1, 2}
    assert [call.text for call in c.calls] == ["target 0", "target 1", "target 2"]


def test_multiple_embedded_audits_rejected():
    """Independent auditor spans cannot silently share one turn axis."""
    t = embedded_audit(auditor_scaffold(target_tools=True))
    other = model_turn("another audit", span_id="other-auditor")
    other.role = "auditor"
    t.events += [
        SpanBeginEvent(id="other-auditor", name="other auditor", type="auditor"),
        other,
        SpanEndEvent(id="other-auditor"),
    ]
    with pytest.raises(ValueError, match="multiple auditor spans"):
        Lanes(t)


def test_existing_scaffold_without_petri_metadata_keeps_auditor_axis():
    """The generic resolver keeps its existing behaviour without Petri metadata."""
    t = auditor_scaffold(target_tools=True)
    assert Lanes(t).axis_role == "auditor"
    assert len(list(Lanes(t).turns())) == 3
