"""Trajectory registry mapping TrajectoryType to trajectory functions."""

from typing import Callable, Dict
import jax.numpy as jnp

from .types import TrajectoryType, TrajContext
from .core import (
    hover,
    hover_contraction,
    yaw_only,
    circle_horizontal,
    circle_vertical,
    fig8_horizontal,
    fig8_vertical,
    helix,
    sawtooth,
    triangle,
    fig8_contraction,
    fig8_heading_contraction,
    spiral_contraction,
    trefoil_contraction,
)

# Type alias for trajectory functions
TrajectoryFunc = Callable[[float, TrajContext], jnp.ndarray]

# Registry maps trajectory types to their position-only functions
TRAJ_REGISTRY: Dict[TrajectoryType, TrajectoryFunc] = {
    TrajectoryType.HOVER: hover,
    TrajectoryType.HOVER_CONTRACTION: hover_contraction,
    TrajectoryType.YAW_ONLY: yaw_only,
    TrajectoryType.CIRCLE_HORIZONTAL: circle_horizontal,
    TrajectoryType.CIRCLE_VERTICAL: circle_vertical,
    TrajectoryType.FIG8_HORIZONTAL: fig8_horizontal,
    TrajectoryType.FIG8_VERTICAL: fig8_vertical,
    TrajectoryType.HELIX: helix,
    TrajectoryType.SAWTOOTH: sawtooth,
    TrajectoryType.TRIANGLE: triangle,
    TrajectoryType.FIG8_CONTRACTION: fig8_contraction,
    TrajectoryType.FIG8_HEADING_CONTRACTION: fig8_heading_contraction,
    TrajectoryType.SPIRAL_CONTRACTION: spiral_contraction,
    TrajectoryType.TREFOIL_CONTRACTION: trefoil_contraction,
}
