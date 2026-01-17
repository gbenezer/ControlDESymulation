# Copyright (C) 2025 Gil Benezer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Trajectory and Sequence Types

Defines types for time series data in control and dynamical systems:
- State trajectories (time series of states)
- Control sequences (time series of control inputs)
- Output sequences (time series of measurements)
- Noise sequences (time series of stochastic disturbances)
- Time arrays and spans
- Trajectory analysis types

System execution result types (IntegrationResult, SimulationResult, etc.)
are now defined in the system_results module and re-exported here for
backward compatibility.

Mathematical Context
-------------------
Trajectories represent solutions to differential/difference equations:

Continuous: x(t) where dx/dt = f(x(t), u(t), t)
Discrete: {x[k]} where x[k+1] = f(x[k], u[k])

Shape Conventions:
- Single trajectory: (n_steps, nx)
- Batched trajectories: (n_steps, batch, nx)
- Multiple trials: (n_trials, n_steps, nx)

Usage
-----
>>> from controldesymulation.types.trajectories import (
...     StateTrajectory,
...     ControlSequence,
...     TimePoints,
...     IntegrationResult,
...     SimulationResult,
... )
>>>
>>> # Simulate system
>>> trajectory: StateTrajectory = system.simulate(x0, u_seq, steps=100)
>>> print(trajectory.shape)  # (101, nx) - includes t=0
>>>
>>> # Analyze trajectory
>>> time: TimePoints = np.linspace(0, 10, 101)
>>> for t, x in zip(time, trajectory):
...     print(f"t={t:.2f}, x={x}")
"""

from typing import Any, Dict, Optional, Tuple

from typing_extensions import TypedDict

from .core import ArrayLike

# Import system execution result types from system_results module
from .system_results import (
    # Base types
    IntegrationResultBase,
    SimulationResultBase,
    RolloutResultBase,
    DiscreteSimulationResultBase,
    # Continuous deterministic
    IntegrationResult,
    SimulationResult,
    RolloutResult,
    # Continuous stochastic
    SDEIntegrationResult,
    SDESimulationResult,
    SDERolloutResult,
    # Discrete deterministic
    DiscreteSimulationResult,
    DiscreteRolloutResult,
    # Discrete stochastic
    DiscreteStochasticSimulationResult,
    DiscreteStochasticRolloutResult,
    # Union types
    ContinuousIntegrationResultUnion,
    ContinuousSimulationResultUnion,
    ContinuousRolloutResultUnion,
    DiscreteSimulationResultUnion,
    DiscreteRolloutResultUnion,
    SystemResult,
)

# ============================================================================
# Trajectory and Sequence Types
# ============================================================================

StateTrajectory = ArrayLike
"""
State trajectory over time.

Time series of state vectors representing system evolution.

Shapes:
- Single trajectory: (n_steps, nx)
  Each row is x[k] or x(t_k)
  
- Batched trajectories: (n_steps, batch, nx)
  Multiple trajectories simulated in parallel
  
- Multiple trials: (n_trials, n_steps, nx)
  Independent simulation runs

Indexing:
- trajectory[k] -> state at step k (nx,)
- trajectory[:, i] -> i-th state component over time (n_steps,)
- trajectory[k, b] -> state at step k, batch b (nx,)

Examples
--------
>>> # Single trajectory from simulation
>>> x0 = np.array([1.0, 0.0])
>>> trajectory: StateTrajectory = system.simulate(x0, u_seq, steps=100)
>>> print(trajectory.shape)  # (101, 2) - includes initial state
>>> 
>>> # Extract position and velocity
>>> position = trajectory[:, 0]  # First state component
>>> velocity = trajectory[:, 1]  # Second state component
>>> 
>>> # Batched simulation (Monte Carlo)
>>> x0_batch = np.random.randn(1000, 2)  # 1000 initial conditions
>>> trajectories: StateTrajectory = system.simulate_batch(x0_batch, u_seq)
>>> print(trajectories.shape)  # (101, 1000, 2)
>>> 
>>> # Multiple independent trials
>>> trials: StateTrajectory = np.array([
...     system.simulate(x0, u_seq, steps=100)
...     for _ in range(10)
... ])
>>> print(trials.shape)  # (10, 101, 2)
"""

ControlSequence = ArrayLike
"""
Control input sequence over time.

Time series of control vectors applied to system.

Shapes:
- Single sequence: (n_steps, nu)
  Control at each time step
  
- Batched sequences: (n_steps, batch, nu)
  Different controls for each batch
  
- Open-loop: (n_steps, nu)
  Pre-computed control sequence

Note: Length is typically n_steps (not n_steps+1) since u[k] affects
x[k] → x[k+1], and we don't need u at final state.

Examples
--------
>>> # Zero control
>>> u_seq: ControlSequence = np.zeros((100, 1))
>>> 
>>> # Sinusoidal control
>>> t = np.linspace(0, 10, 100)
>>> u_seq: ControlSequence = np.sin(t).reshape(-1, 1)
>>> 
>>> # MPC generates control sequence
>>> u_optimal: ControlSequence = mpc.solve(x0, horizon=20)
>>> print(u_optimal.shape)  # (20, nu)
>>> 
>>> # Feedback control sequence (computed online)
>>> trajectory = [x0]
>>> controls = []
>>> x = x0
>>> for k in range(100):
...     u = controller(x)  # Feedback policy
...     controls.append(u)
...     x = system.step(x, u)
...     trajectory.append(x)
>>> u_seq: ControlSequence = np.array(controls)
>>> 
>>> # Batched control for multiple systems
>>> u_batch: ControlSequence = np.random.randn(100, 50, 2)  # 50 systems
"""

OutputSequence = ArrayLike
"""
Output/measurement sequence over time.

Time series of sensor measurements or system outputs.

Shapes:
- Single sequence: (n_steps, ny)
  Measurements at each time
  
- Batched sequences: (n_steps, batch, ny)
  Multiple measurement streams

Common uses:
- Sensor data for state estimation
- System identification datasets
- Validation data for learned models

Examples
--------
>>> # Simulate and observe
>>> trajectory: StateTrajectory = system.simulate(x0, u_seq, steps=100)
>>> observations: OutputSequence = system.observe(trajectory)
>>> print(observations.shape)  # (101, ny)
>>> 
>>> # Noisy measurements
>>> y_clean: OutputSequence = C @ trajectory.T  # (ny, n_steps)
>>> y_noisy: OutputSequence = y_clean + np.random.randn(*y_clean.shape) * 0.1
>>> y_noisy = y_noisy.T  # (n_steps, ny)
>>> 
>>> # Kalman filter with measurements
>>> x_estimates = []
>>> for k in range(len(observations)):
...     y_k = observations[k]
...     x_hat = kalman_filter.update(y_k, u_seq[k])
...     x_estimates.append(x_hat)
>>> 
>>> # System identification dataset
>>> dataset = {
...     'inputs': u_seq,      # (n_steps, nu)
...     'outputs': y_seq,     # (n_steps, ny)
...     'time': t_points,     # (n_steps,)
... }
"""

NoiseSequence = ArrayLike
"""
Noise/disturbance sequence for stochastic simulation.

Time series of random disturbances in stochastic systems.

Shapes:
- Single sequence: (n_steps, nw)
  IID noise samples
  
- Batched sequences: (n_steps, batch, nw)
  Independent noise for each trajectory

Distribution:
- Discrete stochastic: w[k] ~ N(0, I)
- Continuous SDE: dW[k] ~ N(0, dt*I) (Brownian increments)

Examples
--------
>>> # Standard normal noise for discrete system
>>> w_seq: NoiseSequence = np.random.randn(100, 2)
>>> trajectory = system.simulate_stochastic(x0, u_seq, w_seq)
>>> 
>>> # Brownian increments for continuous SDE
>>> dt = 0.01
>>> n_steps = 1000
>>> dW: NoiseSequence = np.random.randn(n_steps, 2) * np.sqrt(dt)
>>> 
>>> # Reproducible simulation with seed
>>> np.random.seed(42)
>>> w_seq: NoiseSequence = np.random.randn(100, 3)
>>> 
>>> # Batched Monte Carlo simulation
>>> n_trials = 1000
>>> w_batch: NoiseSequence = np.random.randn(100, n_trials, 2)
>>> trajectories_mc = system.simulate_stochastic_batch(x0, u_seq, w_batch)
>>> 
>>> # Colored noise (correlated)
>>> # First generate white noise, then filter
>>> w_white = np.random.randn(1000, 2)
>>> # Apply low-pass filter for colored noise
>>> from scipy.signal import lfilter
>>> b, a = [0.1, 0.9], [1.0]
>>> w_colored: NoiseSequence = lfilter(b, a, w_white, axis=0)
"""


# ============================================================================
# Time Array Types
# ============================================================================

TimePoints = ArrayLike
"""
Array of time points for simulation or evaluation.

Discrete time instants at which system is evaluated.

Shape: (n_points,)

Types:
- Regular grid: t = [0, dt, 2*dt, ..., T]
- Irregular grid: t = [0, 0.1, 0.15, 0.3, ...]
- Adaptive: From adaptive integrator (irregular)

Examples
--------
>>> # Regular time grid
>>> t: TimePoints = np.linspace(0, 10, 101)
>>> dt = t[1] - t[0]
>>> print(f"dt = {dt:.3f}")  # 0.100
>>> 
>>> # Irregular time grid
>>> t: TimePoints = np.array([0, 0.1, 0.15, 0.5, 1.0, 2.0, 5.0, 10.0])
>>> 
>>> # Logarithmic spacing (for stiff systems)
>>> t: TimePoints = np.logspace(-3, 1, 100)  # 0.001 to 10
>>> 
>>> # From simulation result
>>> result: IntegrationResult = integrator.solve(x0, u, t_span)
>>> t: TimePoints = result['t']
>>> trajectory: StateTrajectory = result['x']
>>> 
>>> # Discrete-time steps (as floats)
>>> k = np.arange(0, 100)
>>> t: TimePoints = k * dt  # Convert to continuous time
"""

TimeSpan = Tuple[float, float]
"""
Time interval for continuous integration: (t_start, t_end).

Defines initial and final times for ODE/SDE integration.

Format: (t_start, t_end) where t_start < t_end

Examples
--------
>>> # Standard interval [0, T]
>>> t_span: TimeSpan = (0.0, 10.0)
>>> 
>>> # Non-zero start time
>>> t_span: TimeSpan = (5.0, 15.0)
>>> 
>>> # Short interval for testing
>>> t_span: TimeSpan = (0.0, 0.1)
>>> 
>>> # Use in integration
>>> from scipy.integrate import solve_ivp
>>> result = solve_ivp(
...     fun=dynamics,
...     t_span=t_span,
...     y0=x0,
...     method='RK45'
... )
>>> 
>>> # Extract t_eval from t_span
>>> t_start, t_end = t_span
>>> t_eval: TimePoints = np.linspace(t_start, t_end, 1000)
"""


# ============================================================================
# Trajectory Analysis Types
# ============================================================================


class TrajectoryStatistics(TypedDict, total=False):
    """
    Statistical summary of trajectory.

    Computed statistics over time series data.

    Attributes
    ----------
    mean : ArrayLike
        Mean state over trajectory (nx,)
    std : ArrayLike
        Standard deviation over trajectory (nx,)
    min : ArrayLike
        Minimum values (nx,)
    max : ArrayLike
        Maximum values (nx,)
    initial : ArrayLike
        Initial state x[0] (nx,)
    final : ArrayLike
        Final state x[-1] (nx,)
    length : int
        Number of time steps
    duration : float
        Time duration (t_end - t_start)

    Examples
    --------
    >>> def compute_trajectory_stats(trajectory: StateTrajectory) -> TrajectoryStatistics:
    ...     '''Compute statistics of trajectory.'''
    ...     return TrajectoryStatistics(
    ...         mean=np.mean(trajectory, axis=0),
    ...         std=np.std(trajectory, axis=0),
    ...         min=np.min(trajectory, axis=0),
    ...         max=np.max(trajectory, axis=0),
    ...         initial=trajectory[0],
    ...         final=trajectory[-1],
    ...         length=len(trajectory),
    ...     )
    >>>
    >>> stats: TrajectoryStatistics = compute_trajectory_stats(trajectory)
    >>> print(f"Mean state: {stats['mean']}")
    >>> print(f"Final state: {stats['final']}")
    >>> print(f"Max deviation: {np.max(stats['std'])}")
    """

    mean: ArrayLike
    std: ArrayLike
    min: ArrayLike
    max: ArrayLike
    initial: ArrayLike
    final: ArrayLike
    length: int
    duration: float


class TrajectorySegment(TypedDict):
    """
    Segment of trajectory between two time points.

    Extracted portion of full trajectory for analysis.

    Attributes
    ----------
    states : StateTrajectory
        State trajectory segment
    controls : Optional[ControlSequence]
        Control sequence segment
    time : TimePoints
        Time points for segment
    start_index : int
        Index in original trajectory where segment starts
    end_index : int
        Index in original trajectory where segment ends

    Examples
    --------
    >>> def extract_segment(
    ...     result: SimulationResult,
    ...     t_start: float,
    ...     t_end: float
    ... ) -> TrajectorySegment:
    ...     '''Extract trajectory segment.'''
    ...     time = result['t']
    ...     mask = (time >= t_start) & (time <= t_end)
    ...     indices = np.where(mask)[0]
    ...
    ...     return TrajectorySegment(
    ...         states=result['x'][mask],
    ...         controls=result['u'][mask[:-1]] if 'u' in result else None,
    ...         time=time[mask],
    ...         start_index=indices[0],
    ...         end_index=indices[-1],
    ...     )
    >>>
    >>> # Extract transient response (first 2 seconds)
    >>> segment: TrajectorySegment = extract_segment(result, 0.0, 2.0)
    >>> transient_states = segment['states']
    """

    states: ArrayLike
    controls: Optional[ArrayLike]
    time: ArrayLike
    start_index: int
    end_index: int


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # Trajectory and sequence types
    "StateTrajectory",
    "ControlSequence",
    "OutputSequence",
    "NoiseSequence",
    # Time types
    "TimePoints",
    "TimeSpan",
    # Result types (re-exported from system_results)
    # Base types
    "IntegrationResultBase",
    "SimulationResultBase",
    "RolloutResultBase",
    "DiscreteSimulationResultBase",
    # Continuous deterministic
    "IntegrationResult",
    "SimulationResult",
    "RolloutResult",
    # Continuous stochastic
    "SDEIntegrationResult",
    "SDESimulationResult",
    "SDERolloutResult",
    # Discrete deterministic
    "DiscreteSimulationResult",
    "DiscreteRolloutResult",
    # Discrete stochastic
    "DiscreteStochasticSimulationResult",
    "DiscreteStochasticRolloutResult",
    # Union types
    "ContinuousIntegrationResultUnion",
    "ContinuousSimulationResultUnion",
    "ContinuousRolloutResultUnion",
    "DiscreteSimulationResultUnion",
    "DiscreteRolloutResultUnion",
    "SystemResult",
    # Analysis types
    "TrajectoryStatistics",
    "TrajectorySegment",
]
