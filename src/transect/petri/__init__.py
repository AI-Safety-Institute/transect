"""The Petri eval family: an auditor model driving a target model through
tools, with rollbacks that branch the target's conversation.

A run is Petri when its recorded model roles include ``auditor`` and
``target`` (`is_petri`); the eval_setup scanner stamps that as
``eval_family`` on `transcript_info`. On such a transcript the auditor's
turns are the axis and each target trajectory is a role lane
(`scanners.helpers.Lanes`); `conversation.target_conversation` reads
the target's calls, its stored branch tree and the surviving
conversation for the report's target conversation section.
"""

from typing import Any

EVAL_FAMILY = "petri"


def is_petri(model_roles: Any) -> bool:
    """Whether a run's recorded model roles name an auditor and a target."""
    return isinstance(model_roles, dict) and {"auditor", "target"} <= set(model_roles)
