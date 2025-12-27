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
Discrete-Time Trajectory Simulator

Simulates discrete-time trajectories for both pure discrete systems and
discretized continuous systems.

This simulator handles:
- Pure discrete systems (DiscreteSymbolicSystem)
- Discretized continuous systems (SymbolicDynamicalSystem + Discretizer)
- Controller types (sequence, function, nn.Module, None)
- Output feedback with observers
- Batched simulation
- Return options (all states vs final, with/without controls)

Architecture
-----------
DiscreteSimulator is a THIN wrapper that delegates to:
- DiscreteSymbolicSystem.forward() for pure discrete systems
- Discretizer.step() for discretized continuous systems
- Observer.update() for output feedback (when provided)

The simulator's job is orchestration, not computation.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union

import numpy as np

# Conditional imports for optional backends
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    # Create dummy nn for type hints
    class nn:
        class Module:
            pass


try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

if TYPE_CHECKING:
    from src.observers.observer_base import Observer
    from src.systems.base.discrete_symbolic_system import DiscreteSymbolicSystem
    from src.systems.base.discretization.discretizer import Discretizer
    from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem

# Type alias
from src.types import ArrayLike


class DiscreteSimulator:
    """
    Simulates discrete-time trajectories with flexible control and observation.

    Handles both:
    - Pure discrete systems (x[k+1] = f(x[k], u[k]) defined symbolically)
    - Discretized continuous systems (using numerical integrator)

    Key Features
    -----------
    - Multiple controller types (sequence, function, neural network, None)
    - Output feedback control with observers
    - Batched trajectory simulation
    - Flexible return options (full trajectory vs final state only)
    - Optional control recording
    - Backend-agnostic (NumPy, PyTorch, JAX)

    Controller Types Supported
    --------------------------
    1. **None**: Autonomous system (no control)
       >>> sim.simulate(x0, steps=100, controller=None)

    2. **Tensor/Array**: Pre-computed control sequence
       >>> u_seq = np.array([[0.1], [0.2], ..., [0.0]])  # (steps, nu)
       >>> sim.simulate(x0, steps=100, controller=u_seq)

    3. **Function**: State-feedback controller
       >>> controller = lambda x, k: -K @ x  # LQR-like
       >>> sim.simulate(x0, steps=100, controller=controller)

    4. **nn.Module**: Neural network policy (PyTorch)
       >>> policy = torch.nn.Sequential(...)
       >>> sim.simulate(x0, steps=100, controller=policy)

    Parameters
    ----------
    system : Union[DiscreteSymbolicSystem, SymbolicDynamicalSystem]
        The discrete-time system (or continuous system, requires discretizer)
    discretizer : Optional[Discretizer]
        Discretizer for continuous systems (None for pure discrete)
    observer : Optional[Any]
        Optional observer for output feedback control
        Note: Observer base class not yet implemented.
        When implemented, observer should have:
        - initialize(x0) -> x_hat: Initialize observer state
        - update(x_hat, u, y) -> x_hat_next: Update from measurement

    Raises
    ------
    TypeError
        If system type is invalid or discretizer missing for continuous system

    Examples
    --------
    Pure discrete system:
    >>> from src.systems.base.discrete_symbolic_system import DiscreteSymbolicSystem
    >>> discrete_system = DiscreteLinearSystem()
    >>> sim = DiscreteSimulator(discrete_system)
    >>>
    >>> # Simulate with LQR controller
    >>> K = np.array([[1.0, 2.0]])
    >>> controller = lambda x, k: -K @ x
    >>> result = sim.simulate(x0, steps=100, controller=controller)

    Discretized continuous system:
    >>> from src.discretization.discretizer import Discretizer
    >>> continuous_system = Pendulum()
    >>> discretizer = Discretizer(continuous_system, dt=0.01, method='rk4')
    >>> sim = DiscreteSimulator(continuous_system, discretizer=discretizer)
    >>>
    >>> # Simulate with feedforward control
    >>> u_sequence = np.sin(np.linspace(0, 10, 1000)).reshape(-1, 1)
    >>> result = sim.simulate(x0, steps=1000, controller=u_sequence)

    With observer (output feedback):
    >>> from src.observers.kalman_filter import KalmanFilter
    >>> observer = KalmanFilter(system, Q, R)
    >>> sim = DiscreteSimulator(system, observer=observer)
    >>>
    >>> # Observer estimates state from noisy measurements
    >>> result = sim.simulate(x0, steps=100, controller=controller)
    """

    def __init__(
        self,
        system: Union["DiscreteSymbolicSystem", "SymbolicDynamicalSystem"],
        discretizer: Optional["Discretizer"] = None,
        observer: Optional[Any] = None,
    ):
        """
        Initialize discrete-time simulator.

        Parameters
        ----------
        system : Union[DiscreteSymbolicSystem, SymbolicDynamicalSystem]
            System to simulate
        discretizer : Optional[Discretizer]
            Required if system is continuous, None if system is discrete
        observer : Optional[Any]
            Observer for output feedback control
            Note: Observer base class not yet implemented.
            When provided, should implement:
            - initialize(x0) -> x_hat
            - update(x_hat, u, y) -> x_hat_next

        Raises
        ------
        TypeError
            If system is continuous but no discretizer provided
        """
        self.system = system
        self.discretizer = discretizer
        self.observer = observer

        # Validate system/discretizer compatibility
        from src.systems.base.discrete_symbolic_system import DiscreteSymbolicSystem

        is_discrete = isinstance(system, DiscreteSymbolicSystem) or getattr(
            system, "_is_discrete", False
        )

        if not is_discrete and discretizer is None:
            raise TypeError(
                f"Continuous system {system.__class__.__name__} requires a Discretizer. "
                f"Either provide discretizer parameter or use DiscreteSymbolicSystem."
            )

        # Store system properties
        self.nx = system.nx
        self.nu = system.nu
        self.ny = system.ny

        # Determine backend from system
        self.backend = system._default_backend

    def simulate(
        self,
        x0: ArrayLike,
        steps: int,
        controller: Optional[Union[ArrayLike, Callable, "nn.Module"]] = None,
        dt: Optional[float] = None,
        return_controls: bool = False,
        return_final_only: bool = False,
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
        """
        Simulate discrete-time trajectory.

        Parameters
        ----------
        x0 : ArrayLike
            Initial state(s). Shape: (nx,) or (batch_size, nx)
        steps : int
            Number of discrete time steps to simulate
        controller : Optional[Union[ArrayLike, Callable, nn.Module]]
            Control inputs. Can be:
            - None: No control (autonomous system)
            - ArrayLike: Pre-computed control sequence
              - Shape: (steps, nu) for single trajectory
              - Shape: (batch_size, steps, nu) for batched
            - Callable: Function u = f(x, k) returning control at step k
            - nn.Module: Neural network policy u = π(x) (PyTorch only)
        dt : Optional[float]
            Time step size (required if discretizer is used, ignored for pure discrete)
        return_controls : bool
            If True, return tuple (states, controls)
        return_final_only : bool
            If True, return only final state instead of full trajectory
            Memory-efficient for long rollouts when only endpoint matters

        Returns
        -------
        states : ArrayLike
            If return_final_only=False:
              - Single: shape (steps+1, nx)
              - Batched: shape (batch_size, steps+1, nx)
            If return_final_only=True:
              - Single: shape (nx,)
              - Batched: shape (batch_size, nx)
        controls : ArrayLike (optional)
            Returned if return_controls=True
            - Single: shape (steps, nu)
            - Batched: shape (batch_size, steps, nu)

        Raises
        ------
        ValueError
            If steps <= 0
            If controller dimensions don't match
            If dt is None when discretizer is used

        Examples
        --------
        Autonomous system:
        >>> states = sim.simulate(
        ...     x0=np.array([1.0, 0.0]),
        ...     steps=100
        ... )
        >>> states.shape
        (101, 2)  # steps+1 includes initial state

        With pre-computed control:
        >>> u_seq = np.random.randn(100, 1)
        >>> states = sim.simulate(x0, steps=100, controller=u_seq)

        With state feedback:
        >>> K = np.array([[1.0, 2.0]])
        >>> controller = lambda x, k: -K @ x
        >>> states = sim.simulate(x0, steps=100, controller=controller)

        With neural network policy (PyTorch):
        >>> policy = torch.nn.Sequential(
        ...     torch.nn.Linear(2, 64),
        ...     torch.nn.ReLU(),
        ...     torch.nn.Linear(64, 1)
        ... )
        >>> states = sim.simulate(x0_torch, steps=100, controller=policy)

        Return only final state (memory efficient):
        >>> x_final = sim.simulate(
        ...     x0, steps=10000, controller=controller, return_final_only=True
        ... )
        >>> x_final.shape
        (2,)  # Just the final state

        Return controls too:
        >>> states, controls = sim.simulate(
        ...     x0, steps=100, controller=controller, return_controls=True
        ... )
        >>> controls.shape
        (100, 1)

        Batched simulation:
        >>> x0_batch = np.random.randn(32, 2)  # 32 initial conditions
        >>> states_batch = sim.simulate(x0_batch, steps=100)
        >>> states_batch.shape
        (32, 101, 2)  # 32 trajectories

        With observer (output feedback):
        >>> sim_with_obs = DiscreteSimulator(system, observer=kalman_filter)
        >>> states = sim_with_obs.simulate(x0, steps=100, controller=controller)
        >>> # Controller receives observer's state estimate, not true state
        """
        # Validate inputs
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")

        if self.discretizer is not None and dt is None:
            raise ValueError("dt must be provided when using a discretizer for continuous systems")

        # Handle batched vs single trajectory
        if self._get_ndim(x0) == 1:
            x0 = self._expand_batch(x0, 0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        batch_size, state_dim = self._get_shape(x0)

        if state_dim != self.nx:
            raise ValueError(
                f"Initial state dimension {state_dim} doesn't match system dimension {self.nx}"
            )

        # Initialize storage
        if not return_final_only:
            states = self._zeros(batch_size, steps + 1, state_dim)
            states = self._set_initial(states, x0)
        else:
            x_current = self._clone(x0)

        if return_controls:
            controls = self._zeros(batch_size, steps, self.nu)

        # Initialize observer state if needed
        if self.observer is not None:
            x_hat = self.observer.initialize(x0)
        else:
            x_hat = self._clone(x0) if return_final_only else self._get_state(states, 0)

        # Main simulation loop
        for k in range(steps):
            # Get current state for control computation
            if return_final_only:
                x_ctrl = x_hat if self.observer is not None else x_current
            else:
                x_ctrl = x_hat if self.observer is not None else self._get_state(states, k)

            # Compute control
            u = self._compute_control(controller, x_ctrl, k, batch_size)

            if return_controls:
                controls = self._set_control(controls, k, u)

            # Propagate dynamics
            if return_final_only:
                x_next = self._step(x_current, u, dt)
                x_current = x_next
            else:
                x_current = self._get_state(states, k)
                x_next = self._step(x_current, u, dt)
                states = self._set_state(states, k + 1, x_next)

            # Update observer if needed
            if self.observer is not None:
                y = self.system.h(x_next, backend=self.backend)
                x_hat = self.observer.update(x_hat, u, y)

        # Prepare output
        if return_final_only:
            result_states = self._squeeze_if_needed(x_current, squeeze_batch)
        else:
            result_states = self._squeeze_if_needed(states, squeeze_batch)

        if return_controls:
            result_controls = self._squeeze_if_needed(controls, squeeze_batch)
            return result_states, result_controls
        else:
            return result_states

    # ========================================================================
    # Control Computation
    # ========================================================================

    def _compute_control(
        self,
        controller: Optional[Union[ArrayLike, Callable, "nn.Module"]],
        x: ArrayLike,
        k: int,
        batch_size: int,
    ) -> ArrayLike:
        """
        Compute control input based on controller type.

        Parameters
        ----------
        controller : Optional[Union[ArrayLike, Callable, nn.Module]]
            Controller specification
        x : ArrayLike
            Current state (batch_size, nx) - ALWAYS 2D internally
        k : int
            Current time step
        batch_size : int
            Number of trajectories

        Returns
        -------
        ArrayLike
            Control input (batch_size, nu)

        Notes
        -----
        State x is always 2D internally (batch_size, nx), even for batch_size=1.
        For single-trajectory controllers that expect 1D input, we squeeze
        before calling and expand result.
        """
        if controller is None:
            # Autonomous system
            return self._zeros(batch_size, self.nu)

        elif self._is_array(controller):
            # Pre-computed sequence
            ndim = self._get_ndim(controller)

            if ndim == 2:
                # Single trajectory: (steps, nu)
                u_k = self._get_item(controller, k)
                # Expand to batch if needed
                if batch_size > 1:
                    return self._expand_to_batch(u_k, batch_size)
                else:
                    return self._expand_batch(u_k, 0)  # Add batch dim
            else:
                # Batched: (batch_size, steps, nu)
                return self._get_batch_item(controller, k)

        elif TORCH_AVAILABLE and isinstance(controller, nn.Module):
            # Neural network policy
            if self.backend != "torch":
                raise TypeError(
                    f"Neural network controllers require PyTorch backend, "
                    f"but system uses {self.backend}"
                )
            # x is already (batch_size, nx)
            u = controller(x)

            # Ensure output has correct shape (batch_size, nu)
            if u.ndim == 1:
                u = u.unsqueeze(1)  # (batch_size,) -> (batch_size, 1)
            elif u.ndim == 2 and u.shape[1] != self.nu:
                # Handle transposed output
                if u.shape[0] == self.nu and u.shape[1] == batch_size:
                    u = u.T

            return u

        elif callable(controller):
            # Function controller: u = f(x, k)
            # x has shape (batch_size, nx)

            # For single trajectory (batch_size=1), try calling with 1D state
            if batch_size == 1:
                # Squeeze to 1D for controller
                x_1d = self._squeeze_if_needed(x, True)

                try:
                    u = controller(x_1d, k)
                    # Ensure u is array-like
                    u = np.atleast_1d(np.asarray(u))
                    # Add batch dimension
                    return self._expand_batch(u, 0)
                except (IndexError, ValueError, TypeError):
                    # Controller expects 2D, try with original x
                    u = controller(x, k)
                    return self._ensure_control_shape(u, batch_size)
            else:
                # Batched: call with 2D state
                u = controller(x, k)
                return self._ensure_control_shape(u, batch_size)

        else:
            raise TypeError(
                f"Unsupported controller type: {type(controller).__name__}. "
                f"Must be None, array/tensor, callable, or nn.Module"
            )

    def _ensure_control_shape(self, u: ArrayLike, batch_size: int) -> ArrayLike:
        """Ensure control has shape (batch_size, nu)."""
        # Convert to numpy-like for inspection
        if TORCH_AVAILABLE and isinstance(u, torch.Tensor):
            u_ndim = len(u.shape)
            u_shape = tuple(u.shape)
        elif JAX_AVAILABLE and isinstance(u, jnp.ndarray):
            u_ndim = u.ndim
            u_shape = tuple(u.shape)
        else:
            u_arr = np.atleast_1d(np.asarray(u))
            u_ndim = u_arr.ndim
            u_shape = u_arr.shape
            u = u_arr

        # Handle scalar control
        if u_ndim == 0:
            u = np.atleast_1d(np.asarray(u))
            return self._expand_batch(u, 0)

        # Handle 1D control
        if u_ndim == 1:
            # Could be (nu,) or (batch_size,)
            if len(u) == self.nu:
                # Single control (nu,), expand to batch
                return self._expand_to_batch(u, batch_size)
            elif len(u) == batch_size and self.nu == 1:
                # Batch of scalar controls (batch_size,), reshape to (batch_size, 1)
                if isinstance(u, np.ndarray):
                    return u.reshape(batch_size, 1)
                elif TORCH_AVAILABLE and isinstance(u, torch.Tensor):
                    return u.reshape(batch_size, 1)
                elif JAX_AVAILABLE and isinstance(u, jnp.ndarray):
                    return u.reshape(batch_size, 1)
            else:
                # Assume (nu,), expand to batch
                return self._expand_to_batch(u, batch_size)

        # Already 2D - verify shape
        if u_shape[0] == batch_size and u_shape[1] == self.nu:
            return u
        elif u_shape[0] == self.nu and u_shape[1] == batch_size:
            # Transposed
            if isinstance(u, np.ndarray):
                return u.T
            elif TORCH_AVAILABLE and isinstance(u, torch.Tensor):
                return u.T
            elif JAX_AVAILABLE and isinstance(u, jnp.ndarray):
                return u.T
        else:
            raise ValueError(
                f"Control shape {u_shape} doesn't match expected "
                f"({batch_size}, {self.nu}) or ({self.nu}, {batch_size})"
            )

        return u

    def _validate_control_output(self, u, expected_batch_size, k):
        """Validate control output shape and values."""
        shape = self._get_shape(u)

        if len(shape) != 2:
            raise ValueError(
                f"Controller at step {k} returned {len(shape)}D array. "
                f"Expected 2D with shape ({expected_batch_size}, {self.nu})"
            )

        if shape[0] != expected_batch_size:
            raise ValueError(
                f"Controller at step {k} returned batch_size={shape[0]}, "
                f"expected {expected_batch_size}"
            )

        if shape[1] != self.nu:
            raise ValueError(f"Controller at step {k} returned nu={shape[1]}, expected {self.nu}")

    def _validate_control_output(self, u, expected_batch_size, k):
        """Validate control output shape and values."""
        shape = self._get_shape(u)

        if len(shape) != 2:
            raise ValueError(
                f"Controller at step {k} returned {len(shape)}D array. "
                f"Expected 2D with shape ({expected_batch_size}, {self.nu})"
            )

        if shape[0] != expected_batch_size:
            raise ValueError(
                f"Controller at step {k} returned batch_size={shape[0]}, "
                f"expected {expected_batch_size}"
            )

        if shape[1] != self.nu:
            raise ValueError(f"Controller at step {k} returned nu={shape[1]}, expected {self.nu}")

    # ========================================================================
    # Dynamics Propagation
    # ========================================================================

    def _step(
        self,
        x: ArrayLike,
        u: ArrayLike,
        dt: Optional[float],
    ) -> ArrayLike:
        """
        Single discrete-time step.

        Parameters
        ----------
        x : ArrayLike
            Current state (batch_size, nx)
        u : ArrayLike
            Control input (batch_size, nu) or (batch_size, 0) for autonomous
        dt : Optional[float]
            Time step (only used if discretizer present)

        Returns
        -------
        ArrayLike
            Next state (batch_size, nx)
        """
        if self.discretizer is not None:
            # Use discretizer for continuous system
            # Discretizer handles autonomous systems internally
            if self.nu == 0:
                x_next = self.discretizer.step(x, u=None, dt=dt)
            else:
                x_next = self.discretizer.step(x, u, dt)
        else:
            # Direct discrete-time update
            # Handle autonomous systems - don't pass u at all
            if self.nu == 0:
                x_next = self.system(x, backend=self.backend)
            else:
                x_next = self.system(x, u, backend=self.backend)

        # Ensure output has correct shape (batch_size, nx)
        # Some backends may return flattened or wrong shape
        x_next = self._ensure_state_shape(x_next, x)

        return x_next

    def _ensure_state_shape(self, x_next: ArrayLike, x_template: ArrayLike) -> ArrayLike:
        """
        Ensure x_next has same shape as x_template.

        Parameters
        ----------
        x_next : ArrayLike
            State from dynamics evaluation (might be wrong shape)
        x_template : ArrayLike
            Template state with correct shape (batch_size, nx)

        Returns
        -------
        ArrayLike
            x_next reshaped to match template
        """
        template_shape = self._get_shape(x_template)
        next_shape = self._get_shape(x_next)

        # If shapes match, nothing to do
        if next_shape == template_shape:
            return x_next

        # If x_next is flattened, reshape it
        expected_size = template_shape[0] * template_shape[1]

        if len(next_shape) == 1 and next_shape[0] == expected_size:
            # Flattened: reshape to (batch_size, nx)
            if isinstance(x_next, np.ndarray):
                return x_next.reshape(template_shape)
            elif TORCH_AVAILABLE and isinstance(x_next, torch.Tensor):
                return x_next.reshape(template_shape)
            elif JAX_AVAILABLE and isinstance(x_next, jnp.ndarray):
                return x_next.reshape(template_shape)

        # If x_next has wrong number of dimensions, try to fix
        if len(next_shape) == 1 and next_shape[0] == template_shape[1]:
            # Missing batch dimension: (nx,) → (1, nx)
            return self._expand_batch(x_next, 0)

        # Otherwise return as-is and let numpy/torch raise error
        return x_next

    # ========================================================================
    # Backend-Agnostic Array Operations
    # ========================================================================

    def _get_ndim(self, arr: ArrayLike) -> int:
        """Get number of dimensions."""
        if isinstance(arr, np.ndarray):
            return arr.ndim
        elif TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
            return len(arr.shape)
        elif JAX_AVAILABLE and isinstance(arr, jnp.ndarray):
            return arr.ndim
        else:
            raise TypeError(f"Unsupported array type: {type(arr)}")

    def _get_shape(self, arr: ArrayLike) -> Tuple[int, ...]:
        """Get array shape as tuple."""
        if isinstance(arr, np.ndarray):
            return arr.shape
        elif TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
            return tuple(arr.shape)
        elif JAX_AVAILABLE and isinstance(arr, jnp.ndarray):
            return tuple(arr.shape)
        else:
            raise TypeError(f"Unsupported array type: {type(arr)}")

    def _is_array(self, obj) -> bool:
        """Check if object is array-like."""
        if isinstance(obj, np.ndarray):
            return True
        elif TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
            return True
        elif JAX_AVAILABLE and isinstance(obj, jnp.ndarray):
            return True
        return False

    def _zeros(self, *shape) -> ArrayLike:
        """Create zeros array in system's backend."""
        if self.backend == "numpy":
            return np.zeros(shape)
        elif self.backend == "torch":
            import torch

            return torch.zeros(shape)
        elif self.backend == "jax":
            import jax.numpy as jnp

            return jnp.zeros(shape)

    def _clone(self, arr: ArrayLike) -> ArrayLike:
        """Clone array."""
        if isinstance(arr, np.ndarray):
            return arr.copy()
        elif TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
            return arr.clone()
        elif JAX_AVAILABLE and isinstance(arr, jnp.ndarray):
            return jnp.array(arr)
        else:
            return np.array(arr)

    def _expand_batch(self, arr: ArrayLike, axis: int) -> ArrayLike:
        """Add batch dimension."""
        if isinstance(arr, np.ndarray):
            return np.expand_dims(arr, axis)
        elif TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
            return arr.unsqueeze(axis)
        elif JAX_AVAILABLE and isinstance(arr, jnp.ndarray):
            return jnp.expand_dims(arr, axis)

    def _squeeze_if_needed(self, arr: ArrayLike, should_squeeze: bool) -> ArrayLike:
        """Squeeze batch dimension if needed."""
        if not should_squeeze:
            return arr

        if isinstance(arr, np.ndarray):
            return np.squeeze(arr, 0)
        elif TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
            return arr.squeeze(0)
        elif JAX_AVAILABLE and isinstance(arr, jnp.ndarray):
            return jnp.squeeze(arr, 0)

    def _get_item(self, arr: ArrayLike, idx: int) -> ArrayLike:
        """Get item from array."""
        return arr[idx]

    def _get_batch_item(self, arr: ArrayLike, idx: int) -> ArrayLike:
        """Get item from batched array: arr[:, idx, ...]"""
        return arr[:, idx]

    def _expand_to_batch(self, arr: ArrayLike, batch_size: int) -> ArrayLike:
        """Expand single item to batch."""
        if batch_size == 1:
            # Just add batch dimension
            return self._expand_batch(arr, 0)

        if isinstance(arr, np.ndarray):
            return np.tile(arr, (batch_size, 1))
        elif TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
            return arr.unsqueeze(0).expand(batch_size, -1)
        elif JAX_AVAILABLE and isinstance(arr, jnp.ndarray):
            return jnp.tile(arr, (batch_size, 1))
        else:
            # Convert to numpy and tile
            arr_np = np.asarray(arr)
            return np.tile(arr_np, (batch_size, 1))

    def _set_initial(self, arr: ArrayLike, x0: ArrayLike) -> ArrayLike:
        """Set initial state in trajectory array."""
        if isinstance(arr, np.ndarray):
            arr[:, 0] = x0
            return arr
        elif TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
            arr[:, 0] = x0
            return arr
        elif JAX_AVAILABLE and isinstance(arr, jnp.ndarray):
            # JAX arrays are immutable
            return arr.at[:, 0].set(x0)
        else:
            arr[:, 0] = x0
            return arr

    def _get_state(self, states: ArrayLike, k: int) -> ArrayLike:
        """Get state at time step k."""
        return states[:, k]

    def _set_state(self, states: ArrayLike, k: int, x: ArrayLike) -> ArrayLike:
        """Set state at time step k."""
        if isinstance(states, np.ndarray):
            states[:, k] = x
            return states
        elif TORCH_AVAILABLE and isinstance(states, torch.Tensor):
            states[:, k] = x
            return states
        elif JAX_AVAILABLE and isinstance(states, jnp.ndarray):
            # JAX arrays are immutable
            return states.at[:, k].set(x)
        else:
            states[:, k] = x
            return states

    def _set_control(self, controls: ArrayLike, k: int, u: ArrayLike) -> ArrayLike:
        """Set control at time step k."""
        if isinstance(controls, np.ndarray):
            controls[:, k] = u
            return controls
        elif TORCH_AVAILABLE and isinstance(controls, torch.Tensor):
            controls[:, k] = u
            return controls
        elif JAX_AVAILABLE and isinstance(controls, jnp.ndarray):
            # JAX arrays are immutable
            return controls.at[:, k].set(u)
        else:
            controls[:, k] = u
            return controls

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def simulate_with_disturbances(
        self,
        x0: ArrayLike,
        steps: int,
        controller: Optional[Union[ArrayLike, Callable, "nn.Module"]] = None,
        disturbance_func: Optional[Callable[[int], ArrayLike]] = None,
        dt: Optional[float] = None,
        return_controls: bool = False,
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
        """
        Simulate with additive state disturbances.

        Useful for:
        - Process noise simulation
        - Robustness testing
        - Uncertainty propagation

        Parameters
        ----------
        x0 : ArrayLike
            Initial state
        steps : int
            Number of steps
        controller : Optional
            Controller (same types as simulate)
        disturbance_func : Callable[[int], ArrayLike]
            Function that returns disturbance at step k: d[k] = f(k)
            Disturbance is added: x[k+1] = f(x[k], u[k]) + d[k]
        dt : Optional[float]
            Time step
        return_controls : bool
            Return controls

        Returns
        -------
        states : ArrayLike
            Trajectory with disturbances
        controls : ArrayLike (optional)
            Applied controls

        Examples
        --------
        >>> # Gaussian process noise
        >>> disturbance = lambda k: np.random.randn(2) * 0.01
        >>> states = sim.simulate_with_disturbances(
        ...     x0, steps=100, controller=controller, disturbance_func=disturbance
        ... )

        >>> # Deterministic disturbance (e.g., wind gust at specific time)
        >>> def wind_gust(k):
        ...     if 50 <= k <= 60:
        ...         return np.array([0.1, 0.0])
        ...     return np.array([0.0, 0.0])
        >>> states = sim.simulate_with_disturbances(x0, 100, disturbance_func=wind_gust)
        """
        if disturbance_func is None:
            # No disturbances - call regular simulate
            return self.simulate(
                x0, steps, controller=controller, dt=dt, return_controls=return_controls
            )

        # Handle batched vs single trajectory
        if self._get_ndim(x0) == 1:
            x0 = self._expand_batch(x0, 0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        batch_size, state_dim = self._get_shape(x0)

        # Initialize storage
        states = self._zeros(batch_size, steps + 1, state_dim)
        states = self._set_initial(states, x0)

        if return_controls:
            controls = self._zeros(batch_size, steps, self.nu)

        # Initialize observer if needed
        if self.observer is not None:
            x_hat = self.observer.initialize(x0)
        else:
            x_hat = None

        # Simulation loop with disturbances
        for k in range(steps):
            x_ctrl = x_hat if self.observer is not None else self._get_state(states, k)

            u = self._compute_control(controller, x_ctrl, k, batch_size)

            if return_controls:
                controls = self._set_control(controls, k, u)

            # Propagate dynamics
            x_current = self._get_state(states, k)
            x_next = self._step(x_current, u, dt)

            # Add disturbance
            d_k = disturbance_func(k)

            # Ensure disturbance has proper shape
            if self._get_ndim(d_k) == 1:
                d_k = self._expand_to_batch(d_k, batch_size)

            x_next = x_next + d_k

            states = self._set_state(states, k + 1, x_next)

            # Update observer
            if self.observer is not None:
                y = self.system.h(x_next, backend=self.backend)
                x_hat = self.observer.update(x_hat, u, y)

        # Prepare output
        result_states = self._squeeze_if_needed(states, squeeze_batch)

        if return_controls:
            result_controls = self._squeeze_if_needed(controls, squeeze_batch)
            return result_states, result_controls
        else:
            return result_states

    def get_info(self) -> dict:
        """
        Get simulator configuration information.

        Returns
        -------
        dict
            Simulator configuration

        Examples
        --------
        >>> info = sim.get_info()
        >>> print(info['system_type'])
        'DiscreteLinearSystem'
        >>> print(info['has_observer'])
        False
        """
        return {
            "system": self.system.__class__.__name__,
            "system_type": "discrete" if self.discretizer is None else "discretized_continuous",
            "discretizer": self.discretizer.method if self.discretizer else None,
            "has_observer": self.observer is not None,
            "observer_type": self.observer.__class__.__name__ if self.observer else None,
            "dimensions": {
                "nx": self.nx,
                "nu": self.nu,
                "ny": self.ny,
            },
            "backend": self.backend,
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        observer_str = f", observer={self.observer.__class__.__name__}" if self.observer else ""
        discretizer_str = f", discretizer={self.discretizer.method}" if self.discretizer else ""
        return (
            f"DiscreteSimulator("
            f"system={self.system.__class__.__name__}"
            f"{discretizer_str}{observer_str})"
        )

    def __str__(self) -> str:
        """Human-readable string."""
        sys_type = "discrete" if self.discretizer is None else "discretized"
        obs_str = " with observer" if self.observer else ""
        return f"DiscreteSimulator({self.system.__class__.__name__}, {sys_type}{obs_str})"


# ============================================================================
# Convenience Functions
# ============================================================================


def simulate_discrete(
    system: "DiscreteSymbolicSystem",
    x0: ArrayLike,
    steps: int,
    controller: Optional[Union[ArrayLike, Callable, "nn.Module"]] = None,
    **kwargs,
) -> ArrayLike:
    """
    Convenience function for quick discrete simulation.

    Parameters
    ----------
    system : DiscreteSymbolicSystem
        Discrete system
    x0 : ArrayLike
        Initial state
    steps : int
        Number of steps
    controller : Optional
        Controller
    **kwargs
        Additional options passed to simulate()

    Returns
    -------
    ArrayLike
        State trajectory

    Examples
    --------
    >>> states = simulate_discrete(discrete_system, x0, steps=100)
    >>>
    >>> # With controller
    >>> states = simulate_discrete(
    ...     discrete_system, x0, steps=100,
    ...     controller=lambda x, k: -K @ x
    ... )
    """
    sim = DiscreteSimulator(system)
    return sim.simulate(x0, steps, controller=controller, **kwargs)


def simulate_discretized(
    system: "SymbolicDynamicalSystem",
    x0: ArrayLike,
    steps: int,
    dt: float,
    method: str = "rk4",
    controller: Optional[Union[ArrayLike, Callable, "nn.Module"]] = None,
    **kwargs,
) -> ArrayLike:
    """
    Convenience function for simulating discretized continuous system.

    Parameters
    ----------
    system : SymbolicDynamicalSystem
        Continuous system
    x0 : ArrayLike
        Initial state
    steps : int
        Number of steps
    dt : float
        Time step
    method : str
        Discretization method ('euler', 'rk4', etc.)
    controller : Optional
        Controller
    **kwargs
        Additional options

    Returns
    -------
    ArrayLike
        State trajectory

    Examples
    --------
    >>> states = simulate_discretized(
    ...     pendulum, x0, steps=1000, dt=0.01, method='rk4'
    ... )
    """
    from src.discretization.discretizer import Discretizer

    discretizer = Discretizer(system, dt=dt, method=method)
    sim = DiscreteSimulator(system, discretizer=discretizer)
    return sim.simulate(x0, steps, controller=controller, dt=dt, **kwargs)
