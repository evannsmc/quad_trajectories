"""
Position-only trajectory definitions for quadrotor control.
============================================================

This module provides JIT-compiled trajectory generation functions that return
POSITION ONLY: [x, y, z, yaw]. Controllers compute derivatives (velocity,
acceleration, jerk) via JAX autodiff as needed.

Design Philosophy:
- Single source of truth: trajectory defines position, derivatives computed on demand
- Controllers import trajectory function and apply jacfwd() for needed derivatives
- All trajectories use consistent interface: (t, ctx) -> jnp.ndarray[4]

Example Usage in Controller:
    from jax import jacfwd
    from quad_trajectories import TRAJ_REGISTRY, TrajContext

    traj_fn = TRAJ_REGISTRY[TrajectoryType.CIRCLE_HORIZONTAL]
    ctx = TrajContext(sim=True, spin=False)

    # Position only
    pos = traj_fn(t, ctx)  # [x, y, z, yaw]

    # Velocity via autodiff
    vel = jacfwd(lambda t: traj_fn(t, ctx))(t)  # [vx, vy, vz, yaw_rate]

    # Acceleration if needed
    accel = jacfwd(jacfwd(lambda t: traj_fn(t, ctx)))(t)
"""

import jax.numpy as jnp
from .jax_utils import jit
from .types import TrajContext

# Default heights
SIM_HEIGHT = 3.0
HARDWARE_HEIGHT = 0.85


@jit(static_argnames=("ctx",))
def hover(t: float, ctx: TrajContext) -> jnp.ndarray:
    """Returns constant hover position.

    Args:
        t: Time (unused for hover, but required for consistent interface)
        ctx: Trajectory context with hover_mode selection

    Returns:
        Position array [x, y, z, yaw]
    """
    mode = ctx.hover_mode if ctx.hover_mode is not None else 1
    sim = ctx.sim

    hover_dict = {
        1: jnp.array([0.0, 0.0, -0.9, 0.0]),
        2: jnp.array([0.0, 0.8, -0.9, 0.0]),
        3: jnp.array([0.8, 0.0, -0.8, 0.0]),
        4: jnp.array([0.8, 0.8, -0.8, 0.0]),
        5: jnp.array([0.0, 0.0, -10.0, 0.0]),
        6: jnp.array([1.0, 1.0, -4.0, 0.0]),
        7: jnp.array([0.0, 10.0, -5.0, 0.0]),
        8: jnp.array([1.0, 1.0, -3.0, 0.0]),
    }

    if mode not in hover_dict:
        raise ValueError(f"hover_dict #{mode} not found")

    if not sim and mode > 4:
        raise RuntimeError("hover modes 5+ not available for hardware")

    return hover_dict[mode]


@jit(static_argnames=("ctx",))
def hover_contraction(t: float, ctx: TrajContext) -> jnp.ndarray:
    """Returns the legacy contraction-workspace hover positions."""
    del t

    height = SIM_HEIGHT if ctx.sim else HARDWARE_HEIGHT
    mode = ctx.hover_mode if ctx.hover_mode is not None else 1

    hover_dict = {
        1: jnp.array([0.0, 0.0, -height, 0.0]),
        2: jnp.array([0.0, 0.8, -height, 0.0]),
        3: jnp.array([0.8, 0.0, -height, 0.0]),
        4: jnp.array([0.8, 0.8, -height, 0.0]),
        5: jnp.array([0.0, 0.0, -10.0, 0.0]),
        6: jnp.array([1.0, 1.0, -4.0, 0.0]),
        7: jnp.array([0.0, 10.0, -5.0, 0.0]),
        8: jnp.array([1.0, 1.0, -3.0, 0.0]),
    }

    if mode not in hover_dict:
        raise ValueError(f"hover_dict #{mode} not found")

    if not ctx.sim and mode > 4:
        raise RuntimeError("hover modes 5+ not available for hardware")

    return hover_dict[mode]


@jit(static_argnames=("ctx",))
def yaw_only(t: float, ctx: TrajContext) -> jnp.ndarray:
    """Returns stationary position with yawing motion.

    Args:
        t: Time in seconds
        ctx: Trajectory context

    Returns:
        Position array [x, y, z, yaw]
    """
    height = HARDWARE_HEIGHT if not ctx.sim else SIM_HEIGHT
    spin_period = 20.0

    if ctx.double_speed:
        spin_period /= 2.0

    x = 0.0
    y = 0.0
    z = -height
    yaw = t / (spin_period / (2 * jnp.pi))

    return jnp.array([x, y, z, yaw], dtype=jnp.float64)


@jit(static_argnames=("ctx",))
def circle_horizontal(t: float, ctx: TrajContext) -> jnp.ndarray:
    """Returns horizontal circle trajectory position.

    Args:
        t: Time in seconds
        ctx: Trajectory context

    Returns:
        Position array [x, y, z, yaw]
    """
    radius = 0.6
    period_pos = 13.0
    height = HARDWARE_HEIGHT if not ctx.sim else SIM_HEIGHT
    period_spin = 20.0

    omega_spin = 2 * jnp.pi / period_spin if ctx.spin else 0.0
    if ctx.double_speed:
        period_pos /= 2.0
    omega_pos = 2 * jnp.pi / period_pos

    x = radius * jnp.cos(omega_pos * t)
    y = radius * jnp.sin(omega_pos * t)
    z = -height
    yaw = omega_spin * t

    return jnp.array([x, y, z, yaw], dtype=jnp.float64)


@jit(static_argnames=("ctx",))
def circle_vertical(t: float, ctx: TrajContext) -> jnp.ndarray:
    """Returns vertical circle trajectory position.

    Args:
        t: Time in seconds
        ctx: Trajectory context

    Returns:
        Position array [x, y, z, yaw]
    """
    radius = 0.35
    period_pos = 13.0
    height = HARDWARE_HEIGHT if not ctx.sim else SIM_HEIGHT
    period_spin = 20.0

    omega_spin = 2 * jnp.pi / period_spin if ctx.spin else 0.0
    if ctx.double_speed:
        period_pos /= 2.0
    omega_pos = 2 * jnp.pi / period_pos

    x = radius * jnp.cos(omega_pos * t)
    y = 0.0
    z = -radius * jnp.sin(omega_pos * t) - height
    yaw = omega_spin * t

    return jnp.array([x, y, z, yaw], dtype=jnp.float64)


@jit(static_argnames=("ctx",))
def fig8_horizontal(t: float, ctx: TrajContext) -> jnp.ndarray:
    """Returns horizontal figure-8 trajectory position.

    Args:
        t: Time in seconds
        ctx: Trajectory context

    Returns:
        Position array [x, y, z, yaw]
    """
    radius = 0.35
    period_pos = 13.0
    height = HARDWARE_HEIGHT if not ctx.sim else SIM_HEIGHT
    period_spin = 20.0

    omega_spin = 2 * jnp.pi / period_spin if ctx.spin else 0.0
    if ctx.double_speed:
        period_pos /= 2.0
    omega_pos = 2 * jnp.pi / period_pos

    x = radius * jnp.sin(2 * omega_pos * t)
    y = radius * jnp.sin(omega_pos * t)
    z = -height
    yaw = omega_spin * t

    return jnp.array([x, y, z, yaw], dtype=jnp.float64)


@jit(static_argnames=("ctx",))
def fig8_vertical(t: float, ctx: TrajContext) -> jnp.ndarray:
    """Returns vertical figure-8 trajectory position.

    Args:
        t: Time in seconds
        ctx: Trajectory context (short=True for short variant)

    Returns:
        Position array [x, y, z, yaw]
    """
    radius = 0.35
    period_pos = 13.0
    height = HARDWARE_HEIGHT if not ctx.sim else SIM_HEIGHT
    period_spin = 20.0

    omega_spin = 2 * jnp.pi / period_spin if ctx.spin else 0.0
    if ctx.double_speed:
        period_pos /= 2.0
    omega_pos = 2 * jnp.pi / period_pos

    x = 0.0

    yz1 = radius * jnp.sin(omega_pos * t)
    yz2 = radius * jnp.sin(2 * omega_pos * t)
    y = jnp.where(ctx.short, yz1, yz2)  # type: ignore
    z = jnp.where(ctx.short, -yz2 - height, -yz1 - height)  # type: ignore

    yaw = omega_spin * t

    return jnp.array([x, y, z, yaw], dtype=jnp.float64)


@jit(static_argnames=("ctx",))
def helix(t: float, ctx: TrajContext) -> jnp.ndarray:
    """Returns helix trajectory position (spirals up and down).

    Args:
        t: Time in seconds
        ctx: Trajectory context

    Returns:
        Position array [x, y, z, yaw]
    """
    z0 = HARDWARE_HEIGHT if not ctx.sim else 2.0
    z_max = 2.6 if not ctx.sim else SIM_HEIGHT
    radius = 0.6
    num_turns = 3
    cycle_time = 50.0
    period_spin = 35.0

    omega_spin = 2 * jnp.pi / period_spin if ctx.spin else 0.0

    if ctx.double_speed:
        cycle_time /= 2.0

    t_cycle = t % cycle_time
    T_half = cycle_time / 2.0

    # Use jnp.where for differentiability instead of if/else
    # Going up branch
    z_up = z0 + (z_max - z0) * (t_cycle / T_half)
    progress_up = (z_up - z0) / (z_max - z0)

    # Going down branch
    t_down = t_cycle - T_half
    z_down = z_max - (z_max - z0) * (t_down / T_half)
    progress_down = (z_down - z0) / (z_max - z0)

    # Select based on condition using jnp.where for differentiability
    z = jnp.where(t_cycle <= T_half, z_up, z_down)
    progress = jnp.where(t_cycle <= T_half, progress_up, progress_down)

    # Angle is based on progress through the cycle
    theta = 2 * jnp.pi * num_turns * progress
    x = radius * jnp.cos(theta)
    y = radius * jnp.sin(theta)
    yaw = omega_spin * t

    return jnp.array([x, y, -z, yaw], dtype=jnp.float64)


@jit(static_argnames=("ctx",))
def sawtooth(t: float, ctx: TrajContext) -> jnp.ndarray:
    """Returns sawtooth pattern trajectory position (waypoint-based).

    Args:
        t: Time in seconds
        ctx: Trajectory context

    Returns:
        Position array [x, y, z, yaw]
    """
    height = HARDWARE_HEIGHT if not ctx.sim else SIM_HEIGHT
    flight_time = 120.0
    num_repeats = 2 if ctx.double_speed else 1
    period_spin = 30.0

    omega_spin = 2 * jnp.pi / period_spin if ctx.spin else 0.0

    # Waypoints
    points = jnp.array([
        [0.0, 0.0], [0.0, 0.4], [0.4, -0.4], [0.4, 0.4], [0.4, -0.4],
        [0.0, 0.4], [0.0, -0.4], [-0.4, 0.4], [-0.4, -0.4],
        [-0.4, 0.4], [0.0, -0.4], [0.0, 0.0]
    ], dtype=jnp.float64)

    # Adjust flight time based on number of repetitions
    adjusted_flight_time = flight_time / num_repeats
    num_segments = len(points) - 1
    T_seg = adjusted_flight_time / num_segments

    # Calculate time within current cycle
    cycle_time = t % (num_segments * T_seg)

    # Determine segment index (continuous)
    segment_idx = jnp.floor(cycle_time / T_seg)
    segment_idx = jnp.clip(segment_idx, 0, num_segments - 1).astype(int)

    # Time within the current segment
    local_time = cycle_time - segment_idx * T_seg

    # Linear interpolation
    start_point = points[segment_idx]
    end_point = points[(segment_idx + 1) % len(points)]

    alpha = local_time / T_seg
    x = start_point[0] + (end_point[0] - start_point[0]) * alpha
    y = start_point[1] + (end_point[1] - start_point[1]) * alpha
    z = -height
    yaw = omega_spin * t

    return jnp.array([x, y, z, yaw], dtype=jnp.float64)


@jit(static_argnames=("ctx",))
def triangle(t: float, ctx: TrajContext) -> jnp.ndarray:
    """Returns equilateral triangle trajectory position (waypoint-based).

    Args:
        t: Time in seconds
        ctx: Trajectory context

    Returns:
        Position array [x, y, z, yaw]
    """
    height = HARDWARE_HEIGHT if not ctx.sim else SIM_HEIGHT
    side_length = 0.8
    flight_time = 60.0
    num_repeats = 2 if ctx.double_speed else 1
    period_spin = 20.0

    omega_spin = 2 * jnp.pi / period_spin if ctx.spin else 0.0

    # Triangle vertices
    h = jnp.sqrt(side_length**2 - (side_length/2)**2)
    points = jnp.array([
        [0.0, h/2],
        [side_length/2, -h/2],
        [-side_length/2, -h/2]
    ], dtype=jnp.float64)

    # Calculate segment time
    T_seg = flight_time / (3 * num_repeats)

    # Calculate time within current cycle
    cycle_time = t % (3 * T_seg)

    # Determine segment index
    segment_idx = jnp.floor(cycle_time / T_seg)
    segment_idx = jnp.clip(segment_idx, 0, 2).astype(int)

    # Time within the current segment
    local_time = cycle_time - segment_idx * T_seg

    # Linear interpolation
    start_point = points[segment_idx]
    end_point = points[(segment_idx + 1) % 3]

    alpha = local_time / T_seg
    x = start_point[0] + (end_point[0] - start_point[0]) * alpha
    y = start_point[1] + (end_point[1] - start_point[1]) * alpha
    z = -height
    yaw = omega_spin * t

    return jnp.array([x, y, z, yaw], dtype=jnp.float64)

@jit(static_argnames=("ctx",))
def fig8_contraction(t: float, ctx: TrajContext) -> jnp.ndarray:
    """Returns the set3 contraction figure-eight."""
    del ctx

    height = 3.0
    radius = 2.0
    period = 15.0

    px = radius * jnp.sin(2 * jnp.pi * t / period)
    py = radius * jnp.sin(4 * jnp.pi * t / period) / 2.0
    pz = -height
    psi = 0.0

    return jnp.array([px, py, pz, psi], dtype=jnp.float64)


@jit(static_argnames=("ctx",))
def fig8_heading_contraction(t: float, ctx: TrajContext) -> jnp.ndarray:
    """Returns the set3 contraction figure-eight with heading tracking."""
    del ctx

    height = 3.0
    radius = 1.5
    period = 20.0
    s = 2 * jnp.pi * t / period

    px = radius * jnp.sin(s)
    py = radius * jnp.sin(2 * s) / 2.0
    pz = -height
    psi = jnp.arctan2(jnp.cos(2 * s), jnp.cos(s))

    return jnp.array([px, py, pz, psi], dtype=jnp.float64)


@jit(static_argnames=("ctx",))
def spiral_contraction(t: float, ctx: TrajContext) -> jnp.ndarray:
    """Returns the set3 contraction retracing spiral."""
    del ctx

    h_low = 1.5
    h_high = 3.0
    radius = 1.5
    cycle_time = 55.0
    num_turns = 3.0

    # Use a smooth 0 -> 1 -> 0 progress law so the retrace happens with zero
    # velocity at the turnaround instead of a discontinuous branch flip.
    phase = 2.0 * jnp.pi * jnp.mod(t, cycle_time) / cycle_time
    progress = 0.5 * (1.0 - jnp.cos(phase))

    z_height = h_low + (h_high - h_low) * progress
    theta = 2.0 * jnp.pi * num_turns * progress

    px = radius * jnp.cos(theta)
    py = radius * jnp.sin(theta)
    pz = -z_height
    psi = 0.0

    return jnp.array([px, py, pz, psi], dtype=jnp.float64)


@jit(static_argnames=("ctx",))
def trefoil_contraction(t: float, ctx: TrajContext) -> jnp.ndarray:
    """Returns the set3 contraction trefoil trajectory."""
    del ctx

    period = 25.0
    radius = 0.6
    s = 2 * jnp.pi * t / period

    h_low = 1.5
    h_high = 3.0
    h_mid = 0.5 * (h_low + h_high)
    h_amp = 0.5 * (h_high - h_low)

    px = radius * (jnp.sin(s) + 2 * jnp.sin(2 * s))
    py = radius * (jnp.cos(s) - 2 * jnp.cos(2 * s))
    pz = -(h_mid + h_amp * jnp.sin(3 * s))
    psi = 0.0

    return jnp.array([px, py, pz, psi], dtype=jnp.float64)
