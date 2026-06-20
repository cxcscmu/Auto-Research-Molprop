"""Drug Discovery calib specialist."""

from __future__ import annotations

from agent_core.agents.base import DoerBase


class CalibrationDoer(DoerBase):
    specialist = "calib"
