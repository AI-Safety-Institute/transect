"""Cohort / k-roll voting for decision_phases.

The scanner-agnostic machinery lives in ``transect.scanners.cohort``.
"""

from typing import Literal

from pydantic import BaseModel

from transect.scanners.cohort import Ballot, vote_ballots
from transect.scanners.phases_common import DigestJudgement


class TurnVote(BaseModel):
    """One turn's consensus vote record (cohort/k-roll)."""

    turn: int
    label: str
    agreement: float  # voters on the modal label / n_voting
    n_voting: int


class ConsensusJudgement(DigestJudgement):
    """A per-turn judgement plus the consensus annotations."""

    source: Literal["judge", "verifier"] = "judge"
    agreement: float | None = None
    confidence_pm: float | None = None


class CohortMember(BaseModel):
    """One member's complete per-turn record."""

    model: str
    roll: int
    turns: list[DigestJudgement] = []


class CohortAgreement(BaseModel):
    """Run-level cohort shape. (Per-turn agreement lives on the vote
    records; run-level means are exactly derivable from them, so they
    are not duplicated here.)"""

    n_members: int = 0
    n_models: int = 0
    k_rolls: int = 0


class Cohort(BaseModel):
    """Solo runs carry ``ran=False`` with empty defaults."""

    ran: bool = False
    members: list[CohortMember] = []
    vote: list[TurnVote] = []
    agreement: CohortAgreement = CohortAgreement()


def vote_turns(
    member_rows: dict[tuple[str, int], list[DigestJudgement]],
    order: list[tuple[str, int]],
    carry: str | None = None,
) -> tuple[list[ConsensusJudgement], list[TurnVote]]:
    """Consense one chunk's member judgements into per-turn rows.

    The modal label wins; a tie goes to the tied side with the
    higher mean voter confidence; a residual tie (equal counts and
    equal means) goes to the earliest member in ``order``.
    Agreement = voters-on-modal / n_voting.

    Args:
        member_rows: (model, roll) -> that member's rows for one
            chunk. All lists cover the same digest turns, in order.
        order: Member precedence (the residual tie-break).
        carry: The consensus label left open by the previous chunk.

    Returns:
        ``(consensus_rows, votes)`` - one consensus ``DigestJudgement``
        per turn (judged rows carry ``agreement``/``confidence_pm``:
        the modal side's 95%-CI half-width), and one ``TurnVote`` per
        voted turn.
    """
    consensus: list[ConsensusJudgement] = []
    votes: list[TurnVote] = []
    running = carry
    for rows in zip(*(member_rows[m] for m in order), strict=True):
        turn = rows[0].turn
        voters = [r for r in rows if r.basis == "judged"]
        if voters:
            outcome = vote_ballots(
                [
                    Ballot(
                        label=r.phase or "",
                        confidence=r.confidence,
                        explanation=r.explanation,
                    )
                    for r in voters
                ]
            )
            assert outcome is not None  # voters is non-empty
            consensus.append(
                ConsensusJudgement(
                    turn=turn,
                    phase=outcome.label,
                    confidence=outcome.confidence,
                    explanation=outcome.explanation,
                    basis="judged",
                    agreement=(outcome.agreement if outcome.n_voting > 1 else None),
                    confidence_pm=outcome.confidence_pm,
                )
            )
            votes.append(
                TurnVote(
                    turn=turn,
                    label=outcome.label,
                    agreement=outcome.agreement,
                    n_voting=outcome.n_voting,
                )
            )
            running = outcome.label
            continue
        if any(r.basis == "filled" for r in rows):
            if running is not None:
                consensus.append(
                    ConsensusJudgement(
                        turn=turn,
                        phase=running,
                        confidence=0.3,
                        basis="filled",
                        confidence_pm=0.0,
                    )
                )
            else:
                consensus.append(ConsensusJudgement(turn=turn, basis="missing_turn"))
            continue
        statuses = [r.basis for r in rows]
        status = max(
            set(statuses), key=lambda s: (statuses.count(s), -statuses.index(s))
        )
        consensus.append(ConsensusJudgement(turn=turn, basis=status))
    return consensus, votes
