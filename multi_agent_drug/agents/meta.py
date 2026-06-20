"""Drug Discovery meta specialist."""

from __future__ import annotations

from agent_core.agents.base import DoerBase


class MetaDoer(DoerBase):
    specialist = "meta"
