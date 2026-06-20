"""Drug Discovery fsub specialist."""

from __future__ import annotations

from agent_core.agents.base import DoerBase


class SubstructFeatureDoer(DoerBase):
    specialist = "fsub"
