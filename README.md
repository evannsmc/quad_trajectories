# JAX Trajectory Definitions for Quadrotors

[![Part of: PX4-ROS2 Control Stack](https://img.shields.io/badge/Part_of-PX4--ROS2_Control_Stack-blue)](https://www.evannsmc.com/projects)
![Algorithms](https://img.shields.io/badge/Algorithms-NMPC_%7C_NR_%7C_Geometric-brightgreen)
![Languages](https://img.shields.io/badge/Languages-Python_%7C_C%2B%2B-blue)
[![Docker: PX4-ROS2-Docker](https://img.shields.io/badge/Docker-PX4--ROS2--Docker-2496ED?logo=docker&logoColor=white)](https://github.com/evannsmc/PX4-ROS2-Docker)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

A **support library** for the [evannsmc PX4-ROS2 control stack](https://www.evannsmc.com/projects) — a JAX-based library of quadrotor trajectory definitions with automatic, on-demand derivative generation. It is consumed by the stack's **Python and C++ controllers** (NMPC, Newton-Raphson, geometric), so the same trajectory math drives every controller; the easiest way to build and integrate the whole stack is the [PX4-ROS2-Docker](https://github.com/evannsmc/PX4-ROS2-Docker) container.

Trajectories return position-level outputs `[x, y, z, yaw]` — all higher-order derivatives are computed on demand using JAX's forward-mode autodiff (`jacfwd`).

<details>
<summary><b>📖 Table of Contents</b></summary>

- [Available Trajectories](#available-trajectories)
- [Key Features](#key-features)
- [Usage](#usage)
- [Feedforward for `fig8_akash`](#feedforward-for-fig8_akash)
- [Package Structure](#package-structure)
- [Installation](#installation)
- [Used by](#used-by)
- [License](#license)

</details>

## Available Trajectories

| Type | CLI value | Description |
|------|-----------|-------------|
| Hover | `hover` | Stationary hover with 8 sub-modes (altitude, position combos) |
| Yaw Only | `yaw_only` | Hold position while spinning in yaw |
| Circle (Horizontal) | `circle_horz` | Circular path in the XY plane |
| Circle (Vertical) | `circle_vert` | Circular path in the XZ plane |
| Figure-8 (Horizontal) | `fig8_horz` | Lemniscate in the XY plane |
| Figure-8 (Vertical) | `fig8_vert` | Lemniscate in the XZ plane (supports `--short` variant) |
| Figure-8 (Contraction) | `fig8_akash` | Lemniscate identical to the contraction controller's `figure_eight`; supports feedforward via `flat_to_x_u` |
| Helix | `helix` | Spiral ascending and descending |
| Sawtooth | `sawtooth` | Waypoint-based sawtooth pattern |
| Triangle | `triangle` | Waypoint-based triangular pattern |

## Key Features

- **Position-only design** — trajectories define only `[x, y, z, yaw]`; controllers compute velocity, acceleration, jerk, etc. via `jacfwd()`
- **JAX JIT-compiled** — all trajectory functions are JIT-compiled for real-time performance
- **Registry pattern** — `TRAJ_REGISTRY` maps `TrajectoryType` enum values to trajectory callables
- **Context-aware** — `TrajContext` controls sim/hardware mode, hover mode, spin enable, double speed, and short variants

## Usage

```python
from quad_trajectories import TRAJ_REGISTRY, TrajectoryType, TrajContext

ctx = TrajContext(sim=True, spin=True, double_speed=False)
traj_fn = TRAJ_REGISTRY[TrajectoryType.HELIX]

# Get [x, y, z, yaw] at time t
pos = traj_fn(t, ctx)
```

Derivatives are typically computed by the controller using utility functions in `quad_trajectories.utils`, which wrap `jax.jacfwd` to produce velocity, acceleration, and lookahead horizons.

## Feedforward for `fig8_akash`

The `fig8_akash` trajectory supports differential-flatness feedforward using the same approach as the contraction controller. Given a trajectory function `traj_fn(t, ctx) → [px, py, pz, psi]`, two levels of `jax.jacfwd` are applied to recover the full feedforward state and control:

```
x_ff = [px, py, pz, vx, vy, vz, f, phi, th, psi]
         position     velocity  thrust  euler angles

u_ff = [df, dphi, dth, dpsi]
         thrust-rate   angular rates
```

where `f` is specific thrust (m/s²), `phi`/`th`/`psi` are roll/pitch/yaw, and `u_ff` gives the rates of each.

**Single time step (e.g. Newton-Raphson):**

```python
from quad_trajectories import flat_to_x_u, TRAJ_REGISTRY, TrajContext, TrajectoryType

ctx = TrajContext(sim=True, ...)
flat_output = lambda t: TRAJ_REGISTRY[TrajectoryType.FIG8_AKASH](t, ctx)
x_ff, u_ff = flat_to_x_u(t, flat_output)
# u_ff[1:4] = [roll_rate, pitch_rate, yaw_rate] — add to NR output rates
```

**Over a prediction horizon (e.g. NMPC):**

```python
from quad_trajectories import generate_feedforward_trajectory, TRAJ_REGISTRY, TrajContext, TrajectoryType

ctx = TrajContext(sim=True, ...)
x_ff_traj, u_ff_traj = generate_feedforward_trajectory(
    TRAJ_REGISTRY[TrajectoryType.FIG8_AKASH], ctx,
    t_start, horizon, num_steps)
# x_ff_traj[:, 7:10] = [phi, th, psi] per step  → use as euler_ref in NMPC
# u_ff_traj           = [df, dphi, dth, dpsi]    → use as u_ref in NMPC
```

## Package Structure

```
quad_trajectories/
├── __init__.py          # Public API exports
├── core.py              # All trajectory implementations
├── registry.py          # TrajectoryType enum → function mapping
├── types.py             # TrajContext dataclass, TrajectoryType enum
├── utils.py             # Derivative helpers and horizon generation
└── jax_utils.py         # JAX JIT configuration
```

## Installation

```bash
# Inside a ROS 2 workspace src/ directory
git clone git@github.com:evannsmc/quad_trajectories.git
cd .. && colcon build --symlink-install
```

> This is a support package — it is meant to sit alongside one of the controllers in the same workspace `src/`. For a ready-to-build layout of the whole stack, use the [PX4-ROS2-Docker](https://github.com/evannsmc/PX4-ROS2-Docker) container.

## Used by

This is a support package for the controllers in the evannsmc PX4-ROS2 control stack:

- **Python controllers** — [nmpc_acados_px4](https://github.com/evannsmc/nmpc_acados_px4) · [newton_raphson_px4](https://github.com/evannsmc/newton_raphson_px4) · [geometric_px4](https://github.com/evannsmc/geometric_px4)
- **C++ controllers** — [nmpc_acados_px4_cpp](https://github.com/evannsmc/nmpc_acados_px4_cpp) · [newton_raphson_px4_cpp](https://github.com/evannsmc/newton_raphson_px4_cpp) · [geometric_px4_cpp](https://github.com/evannsmc/geometric_px4_cpp)
- **C++ counterpart of this package** — [quad_trajectories_cpp](https://github.com/evannsmc/quad_trajectories_cpp)
- **Docker integration** — [PX4-ROS2-Docker](https://github.com/evannsmc/PX4-ROS2-Docker)

Part of the [evannsmc open-source portfolio](https://www.evannsmc.com/projects).

## License

MIT
