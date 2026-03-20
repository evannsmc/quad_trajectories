"""
Quad Trajectories Package
=========================

Standalone trajectory definitions for quadrotor control. Provides position-only
(x, y, z, yaw) trajectory functions; controllers compute derivatives via JAX autodiff.

Basic Usage:
    from quad_trajectories import TRAJ_REGISTRY, TrajectoryType, TrajContext

    # Get trajectory function
    traj_fn = TRAJ_REGISTRY[TrajectoryType.CIRCLE_HORIZONTAL]
    ctx = TrajContext(sim=True, spin=False)

    # Get position at time t
    pos = traj_fn(t, ctx)  # Returns [x, y, z, yaw]

Computing Derivatives (in your controller):
    from jax import jacfwd

    # Velocity
    vel = jacfwd(lambda t: traj_fn(t, ctx))(t)

    # Acceleration
    accel = jacfwd(jacfwd(lambda t: traj_fn(t, ctx)))(t)

Using Utility Functions:
    from quad_trajectories import get_pos_vel_fn, generate_horizon_with_velocity

    # Get combined position/velocity function
    pos_vel = get_pos_vel_fn(traj_fn, ctx)
    pos, vel = pos_vel(t)

    # Generate samples over a horizon
    positions, velocities = generate_horizon_with_velocity(traj_fn, ctx, t_start, horizon, num_steps)
"""

# Types
from .types import TrajContext, TrajectoryType

# Registry
from .registry import TRAJ_REGISTRY, TrajectoryFunc

# Individual trajectory functions (for direct import if needed)
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
    SIM_HEIGHT,
    HARDWARE_HEIGHT,
)

# Utility functions
from .utils import (
    get_velocity_fn,
    get_acceleration_fn,
    get_pos_vel_fn,
    generate_horizon_positions,
    generate_horizon_with_velocity,
    generate_reference_trajectory,
    GRAVITY,
    flat_to_x_u,
    flat_to_x,
    generate_feedforward_trajectory,
)

# JAX utilities
from .jax_utils import jit

__all__ = [
    # Types
    "TrajContext",
    "TrajectoryType",
    "TrajectoryFunc",
    # Registry
    "TRAJ_REGISTRY",
    # Trajectory functions
    "hover",
    "hover_contraction",
    "yaw_only",
    "circle_horizontal",
    "circle_vertical",
    "fig8_horizontal",
    "fig8_vertical",
    "helix",
    "sawtooth",
    "triangle",
    "fig8_contraction",
    "fig8_heading_contraction",
    "spiral_contraction",
    "trefoil_contraction",
    # Constants
    "SIM_HEIGHT",
    "HARDWARE_HEIGHT",
    "GRAVITY",
    # Utilities
    "get_velocity_fn",
    "get_acceleration_fn",
    "get_pos_vel_fn",
    "generate_horizon_positions",
    "generate_horizon_with_velocity",
    "generate_reference_trajectory",
    "flat_to_x_u",
    "flat_to_x",
    "generate_feedforward_trajectory",
    "jit",
]
