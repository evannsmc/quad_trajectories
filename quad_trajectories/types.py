"""Type definitions for trajectory generation."""

import enum
from dataclasses import dataclass
from typing import Optional

try:
    # Python 3.11+
    from enum import StrEnum  # type: ignore[attr-defined]
except ImportError:
    # Python 3.10 fallback
    class StrEnum(str, enum.Enum):
        """Minimal StrEnum compatible with stdlib's StrEnum."""
        pass


@dataclass(frozen=True)
class TrajContext:
    """Context object containing trajectory parameters.

    Attributes:
        sim: Whether running in simulation (True) or on hardware (False).
             Affects default heights and available modes.
        hover_mode: Hover position selection (1-8). Modes 5+ only available in sim.
        spin: Enable yaw rotation during trajectory.
        double_speed: Execute trajectory at 2x speed.
        short: Use short variant (e.g., for fig8_vertical).
    """
    sim: bool
    hover_mode: Optional[int] = None
    spin: Optional[bool] = None
    double_speed: Optional[bool] = None
    short: Optional[bool] = None


class TrajectoryType(StrEnum):
    """Enumeration of available trajectories."""
    HOVER = "hover"
    HOVER_CONTRACTION = "hover_contraction"
    YAW_ONLY = "yaw_only"
    CIRCLE_HORIZONTAL = "circle_horz"
    CIRCLE_VERTICAL = "circle_vert"
    FIG8_HORIZONTAL = "fig8_horz"
    FIG8_VERTICAL = "fig8_vert"
    HELIX = "helix"
    SAWTOOTH = "sawtooth"
    TRIANGLE = "triangle"
    FIG8_CONTRACTION = "fig8_contraction"
    FIG8_HEADING_CONTRACTION = "fig8_heading_contraction"
    SPIRAL_CONTRACTION = "spiral_contraction"
    TREFOIL_CONTRACTION = "trefoil_contraction"
