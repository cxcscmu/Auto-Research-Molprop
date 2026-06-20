"""Drug Discovery daugm specialist — data augmentation via external datasets."""

from __future__ import annotations

from agent_core.agents.base import DoerBase


class DataAugDoer(DoerBase):
    specialist = "daugm"
