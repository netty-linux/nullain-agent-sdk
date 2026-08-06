"""Nullain Agent SDK — Workflow Orchestrator (P4.27).

A workflow is a Python function that orchestrates subagents deterministically:
which subagents run, in what order, with what fan-out and pipeline stages is
fixed by the script, never decided by an LLM. The orchestration is
deterministic; only the subagent outputs are non-deterministic (LLM calls).

The DSL surface (``agent``, ``parallel``, ``pipeline``, ``phase``, ``log``) is
exposed on :class:`WorkflowContext`; :class:`Workflow` binds a
:class:`WorkflowSpawner` (e.g. :func:`loop_spawner` over an ``AgentLoop``) and
runs the function. Subagents inherit the P4.24 authority-intersection law and
the P4.25/P4.26 tool surface.
"""

from nullain.workflow.context import PipelineStage, WorkflowContext
from nullain.workflow.spawner import WorkflowSpawner, loop_spawner
from nullain.workflow.workflow import Workflow, WorkflowFn

__all__ = [
    "PipelineStage",
    "Workflow",
    "WorkflowContext",
    "WorkflowFn",
    "WorkflowSpawner",
    "loop_spawner",
]
