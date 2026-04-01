"""Utility functions for working with trajectories.

These utilities help controllers compute derivatives and generate
trajectory samples over prediction horizons.
"""

from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from jax import jacfwd, vmap

from .types import TrajContext
from .jax_utils import jit

GRAVITY: float = 9.8  # Match Gazebo world (Tools/simulation/gz/worlds/default.sdf)


def get_velocity_fn(
    traj_fn: Callable[[float, TrajContext], jnp.ndarray],
    ctx: TrajContext
) -> Callable[[float], jnp.ndarray]:
    """Returns a JIT-compiled velocity function for the given trajectory.

    Args:
        traj_fn: Position-only trajectory function (t, ctx) -> [x, y, z, yaw]
        ctx: Trajectory context

    Returns:
        Function (t) -> [vx, vy, vz, yaw_rate]
    """
    @jit
    def vel_fn(t: float) -> jnp.ndarray:
        return jacfwd(lambda t_: traj_fn(t_, ctx))(t)
    return vel_fn


def get_acceleration_fn(
    traj_fn: Callable[[float, TrajContext], jnp.ndarray],
    ctx: TrajContext
) -> Callable[[float], jnp.ndarray]:
    """Returns a JIT-compiled acceleration function for the given trajectory.

    Args:
        traj_fn: Position-only trajectory function (t, ctx) -> [x, y, z, yaw]
        ctx: Trajectory context

    Returns:
        Function (t) -> [ax, ay, az, yaw_accel]
    """
    @jit
    def accel_fn(t: float) -> jnp.ndarray:
        return jacfwd(jacfwd(lambda t_: traj_fn(t_, ctx)))(t)
    return accel_fn


def get_pos_vel_fn(
    traj_fn: Callable[[float, TrajContext], jnp.ndarray],
    ctx: TrajContext
) -> Callable[[float], Tuple[jnp.ndarray, jnp.ndarray]]:
    """Returns a JIT-compiled function that returns both position and velocity.

    Args:
        traj_fn: Position-only trajectory function (t, ctx) -> [x, y, z, yaw]
        ctx: Trajectory context

    Returns:
        Function (t) -> (pos, vel) where each is [x, y, z, yaw] / [vx, vy, vz, yaw_rate]
    """
    @jit
    def pos_vel_fn(t: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pos = traj_fn(t, ctx)
        vel = jacfwd(lambda t_: traj_fn(t_, ctx))(t)
        return pos, vel
    return pos_vel_fn


@jit
def generate_horizon_positions(
    traj_fn: Callable[[float, TrajContext], jnp.ndarray],
    ctx: TrajContext,
    t_start: float,
    horizon: float,
    num_steps: int
) -> jnp.ndarray:
    """Generate position samples over a prediction horizon.

    Args:
        traj_fn: Position-only trajectory function
        ctx: Trajectory context
        t_start: Starting time (seconds)
        horizon: Prediction horizon length (seconds)
        num_steps: Number of discretization steps (>=1)

    Returns:
        Array of shape (num_steps, 4) with positions [x, y, z, yaw]
    """
    if num_steps == 1:
        t_samples = jnp.array([t_start], dtype=jnp.float64)
    else:
        t_samples = jnp.linspace(t_start, t_start + horizon, num_steps, dtype=jnp.float64)

    return vmap(lambda t: traj_fn(t, ctx))(t_samples)


def generate_horizon_with_velocity(
    traj_fn: Callable[[float, TrajContext], jnp.ndarray],
    ctx: TrajContext,
    t_start: float,
    horizon: float,
    num_steps: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate position and velocity samples over a prediction horizon.

    Args:
        traj_fn: Position-only trajectory function
        ctx: Trajectory context
        t_start: Starting time (seconds)
        horizon: Prediction horizon length (seconds)
        num_steps: Number of discretization steps (>=1)

    Returns:
        Tuple of (positions, velocities), each of shape (num_steps, 4)
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be >= 1")

    if num_steps == 1:
        t_samples = jnp.array([t_start], dtype=jnp.float64)
    else:
        t_samples = jnp.linspace(t_start, t_start + horizon, num_steps, dtype=jnp.float64)

    def one_sample(t: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pos = traj_fn(t, ctx)
        vel = jacfwd(lambda t_: traj_fn(t_, ctx))(t)
        return pos, vel

    positions, velocities = vmap(one_sample)(t_samples)
    return positions, velocities


def generate_reference_trajectory(
    traj_func: Callable[[float, TrajContext], jnp.ndarray],
    t_start: float,
    horizon: float,
    num_steps: int,
    ctx: TrajContext
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate reference trajectory for a prediction horizon.

    This function provides compatibility with the original newton_raphson_enhanced
    interface. It is equivalent to generate_horizon_with_velocity but with
    arguments in the original order.

    Args:
        traj_func: Trajectory function that returns position [x, y, z, yaw]
        t_start: Starting time for trajectory (seconds)
        horizon: Prediction horizon length (seconds)
        num_steps: Number of discretization steps (>=1)
        ctx: Trajectory context

    Returns:
        Tuple of (positions, velocities), each of shape (num_steps, 4)
    """
    return generate_horizon_with_velocity(traj_func, ctx, t_start, horizon, num_steps)


def flat_to_x_u(
    t: float,
    flat_output: Callable[[float], jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute feedforward state x_ff and control u_ff from flat output via autodiff.

    Identical to the contraction controller's _flat_to_x_u.

    flat_output(t) returns [px, py, pz, psi].

    State:   x_ff = [px, py, pz, vx, vy, vz, f, phi, th, psi]
               where f = specific thrust (m/s^2), phi = roll, th = pitch
    Control: u_ff = [df, dphi, dth, dpsi]  (rates of f, roll, pitch, yaw)
    """
    g = GRAVITY
    px, py, pz, psi = flat_output(t)
    vx, vy, vz, dpsi = jacfwd(flat_output)(t)

    def f_th_phi(t_):
        ax, ay, az = jacfwd(jacfwd(flat_output))(t_)[:3]
        f = jnp.sqrt(ax**2 + ay**2 + (az - g) ** 2)
        th = jnp.arcsin(-ax / f)
        phi = jnp.arctan2(ay, g - az)
        return jnp.array([f, th, phi])

    f, th, phi = f_th_phi(t)
    df, dth, dphi = jacfwd(f_th_phi)(t)
    x_ff = jnp.array([px, py, pz, vx, vy, vz, f, phi, th, psi])
    u_ff = jnp.array([df, dphi, dth, dpsi])
    return x_ff, u_ff


def flat_to_x(
    t: float,
    flat_output: Callable[[float], jnp.ndarray],
) -> jnp.ndarray:
    """Compute the feedforward state only, without third-order derivatives."""
    g = GRAVITY
    px, py, pz, psi = flat_output(t)
    vx, vy, vz, _ = jacfwd(flat_output)(t)
    ax, ay, az = jacfwd(jacfwd(flat_output))(t)[:3]

    f = jnp.sqrt(ax**2 + ay**2 + (az - g) ** 2)
    th = jnp.arcsin(-ax / f)
    phi = jnp.arctan2(ay, g - az)

    return jnp.array([px, py, pz, vx, vy, vz, f, phi, th, psi])


def generate_feedforward_trajectory(
    traj_fn: Callable[[float, TrajContext], jnp.ndarray],
    ctx: TrajContext,
    t_start: float,
    horizon: float,
    num_steps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate feedforward state and control trajectories over a prediction horizon.

    Uses flat_to_x_u at each time step (same autodiff approach as contraction controller).

    Returns:
        x_ff_traj: shape (num_steps, 10) – [px, py, pz, vx, vy, vz, f, phi, th, psi]
        u_ff_traj: shape (num_steps,  4) – [df, dphi, dth, dpsi]
    """
    if num_steps == 1:
        t_samples = jnp.array([t_start], dtype=jnp.float64)
    else:
        t_samples = jnp.linspace(t_start, t_start + horizon, num_steps, dtype=jnp.float64)

    flat_output = lambda t: traj_fn(t, ctx)
    one_sample = lambda t: flat_to_x_u(t, flat_output)
    x_ff_traj, u_ff_traj = vmap(one_sample)(t_samples)
    return x_ff_traj, u_ff_traj
