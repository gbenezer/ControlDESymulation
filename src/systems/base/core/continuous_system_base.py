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
Continuous System Base Class (Layer 1)
======================================

Abstract base class for all continuous-time dynamical systems.

This module should be placed at:
    src/systems/base/continuous_system_base.py
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union
import numpy as np

from src.types.core import ControlVector, StateVector, ControlInput, FeedbackController, ScalarLike
from src.types.linearization import LinearizationResult
from src.types.backends import IntegrationMethod, Backend
from src.types.trajectories import IntegrationResult, SimulationResult, TimeSpan


class ContinuousSystemBase(ABC):
    """
    Abstract base class for all continuous-time dynamical systems.

    This class defines the fundamental interface that all continuous-time systems
    must implement. All continuous-time systems satisfy:
        dx/dt = f(x, u, t)

    Subclasses must implement:
    1. __call__(x, u, t): Evaluate dynamics at a point
    2. integrate(x0, u, t_span): Low-level numerical integration with solver diagnostics
    3. linearize(x_eq, u_eq): Compute linearization

    Additional concrete methods provided:
    - simulate(): High-level simulation with regular time grid (wraps integrate())

    Examples
    --------
    >>> class MyODESystem(ContinuousSystemBase):
    ...     def __call__(self, x, u=None, t=0.0):
    ...         return -x + (u if u is not None else 0.0)
    ...     
    ...     def integrate(self, x0, u, t_span, method="RK45", **kwargs):
    ...         # Use scipy.integrate.solve_ivp or similar
    ...         result = solve_ivp(...)
    ...         return {
    ...             "t": result.t,
    ...             "y": result.y,
    ...             "success": result.success,
    ...             "nfev": result.nfev,
    ...             ...
    ...         }
    ...     
    ...     def linearize(self, x_eq, u_eq):
    ...         A = -np.eye(self.nx)
    ...         B = np.eye(self.nx, self.nu)
    ...         return (A, B)
    """

    # =========================================================================
    # Abstract Methods (MUST be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def __call__(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        t: ScalarLike = 0.0,
        backend: Optional[Backend] = None
    ) -> StateVector:
        """
        Evaluate continuous-time dynamics: dx/dt = f(x, u, t).

        This is the core dynamics evaluation method. It computes the time derivative
        of the state vector at a given point in state space.

        Parameters
        ----------
        x : StateVector
            Current state vector (nx,) or (nx, n_batch)
        u : Optional[ControlVector]
            Control input vector (nu,) or (nu, n_batch)
            If None, assumes zero control or autonomous dynamics
        t : float
            Current time (default: 0.0)
            Used for time-varying systems

        Returns
        -------
        StateVector
            Time derivative dx/dt with same shape as x

        Notes
        -----
        - For autonomous systems, t is ignored
        - For time-invariant systems, t is typically ignored
        - For batch evaluation, x and u should have shape (n_dim, n_batch)
        - The returned derivative should be in the same backend as the input

        Examples
        --------
        Evaluate dynamics at a single point:

        >>> x = np.array([1.0, 2.0])
        >>> u = np.array([0.5])
        >>> dxdt = system(x, u)  # Returns dx/dt

        Batch evaluation:

        >>> x_batch = np.random.randn(2, 100)  # 100 states
        >>> u_batch = np.random.randn(1, 100)  # 100 controls
        >>> dxdt_batch = system(x_batch, u_batch)  # Returns (2, 100)
        """
        pass

    @abstractmethod
    def integrate(
        self,
        x0: StateVector,
        u: ControlInput = None,
        t_span: TimeSpan = (0.0, 10.0),
        method: IntegrationMethod = "RK45",
        **integrator_kwargs
    ) -> IntegrationResult:
        """
        Low-level numerical integration with ODE solver diagnostics.

        Numerically solve the initial value problem:
            dx/dt = f(x, u, t)
            x(t0) = x0

        This method exposes the raw solver output including adaptive time points,
        convergence information, and performance metrics. For typical use cases,
        consider using simulate() instead, which provides a cleaner interface.

        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        u : Union[ControlVector, Callable[[float], ControlVector], None]
            Control input, can be:
            - None: Zero control or autonomous
            - Array: Constant control u(t) = u for all t
            - Callable: Time-varying control u(t) = u_func(t)
        t_span : tuple[float, float]
            Time interval (t_start, t_end)
        method : str
            Integration method (e.g., 'RK45', 'RK23', 'LSODA', 'Radau', 'BDF')
        **integrator_kwargs
            Additional arguments passed to the ODE solver
            Common: rtol, atol, max_step, first_step, dense_output

        Returns
        -------
        IntegrationResult
            TypedDict (returns as dict) containing:
            - t: Time points (n_points,) - adaptive, chosen by solver
            - y: State trajectory (nx, n_points)
            - success: bool - whether integration succeeded
            - message: str - solver status message
            - nfev: int - number of function evaluations
            - njev: int - number of Jacobian evaluations (if applicable)
            - nlu: int - number of LU decompositions (implicit methods)
            - status: int - termination status code

        Notes
        -----
        The time points in the result are chosen adaptively by the solver
        based on error control, not on a regular grid. For a regular time
        grid, use simulate() instead.

        Solver selection guidelines:
        - 'RK45': Explicit Runge-Kutta (good default for non-stiff)
        - 'RK23': Faster but less accurate than RK45
        - 'DOP853': High-accuracy explicit method
        - 'Radau': Implicit method for stiff systems
        - 'BDF': Implicit multistep for very stiff systems
        - 'LSODA': Automatically switches between stiff/non-stiff

        Examples
        --------
        Basic integration with default solver:

        >>> x0 = np.array([1.0, 0.0])
        >>> result = system.integrate(x0, u=None, t_span=(0, 10))
        >>> print(f"Success: {result['success']}")
        >>> print(f"Function evaluations: {result['nfev']}")
        >>> plt.plot(result['t'], result['y'][0, :])

        Time-varying control:

        >>> def u_func(t):
        ...     return np.array([np.sin(t)])
        >>> result = system.integrate(x0, u_func, t_span=(0, 10))

        Stiff system with tight tolerances:

        >>> result = system.integrate(
        ...     x0, u=None, t_span=(0, 10),
        ...     method='Radau',
        ...     rtol=1e-8,
        ...     atol=1e-10
        ... )

        Check solver performance:

        >>> if result['nfev'] > 10000:
        ...     print("Warning: Many function evaluations - try stiff solver")
        """
        pass

    @abstractmethod
    def linearize(
        self,
        x_eq: StateVector,
        u_eq: Optional[ControlVector] = None
    ) -> LinearizationResult:
        """
        Compute linearized dynamics around an equilibrium point.

        For a continuous-time system dx/dt = f(x, u), compute the linearization:
            d(δx)/dt = A·δx + B·δu

        where:
            A = ∂f/∂x|(x_eq, u_eq)  (State Jacobian, nx × nx)
            B = ∂f/∂u|(x_eq, u_eq)  (Control Jacobian, nx × nu)

        Parameters
        ----------
        x_eq : StateVector
            Equilibrium state (nx,)
        u_eq : Optional[ControlVector]
            Equilibrium control (nu,)
            If None, uses zero control

        Returns
        -------
        LinearizationResult
            Tuple containing Jacobian matrices:
            - Deterministic systems: (A, B)
            - Stochastic systems: (A, B, G) where G is diffusion matrix

        Notes
        -----
        The linearization is valid for small deviations from the equilibrium:
            δx = x - x_eq
            δu = u - u_eq

        For symbolic systems, Jacobians are computed symbolically then evaluated.
        For data-driven systems, Jacobians may be computed via finite differences
        or automatic differentiation.

        The equilibrium point should satisfy f(x_eq, u_eq) ≈ 0 (within tolerance).

        Examples
        --------
        Linearize at origin:

        >>> x_eq = np.zeros(2)
        >>> u_eq = np.zeros(1)
        >>> A, B = system.linearize(x_eq, u_eq)
        >>> print(f"A matrix:\\n{A}")
        >>> print(f"B matrix:\\n{B}")

        Check stability (continuous-time):

        >>> eigenvalues = np.linalg.eigvals(A)
        >>> is_stable = np.all(np.real(eigenvalues) < 0)
        >>> print(f"System stable: {is_stable}")

        Design LQR controller:

        >>> from scipy.linalg import solve_continuous_are
        >>> P = solve_continuous_are(A, B, Q, R)
        >>> K = np.linalg.inv(R) @ B.T @ P
        """
        pass

    # =========================================================================
    # Concrete Methods (Provided by base class)
    # =========================================================================

    def simulate(
        self,
        x0: StateVector,
        controller: Optional[FeedbackController] = None,
        t_span: TimeSpan = (0.0, 10.0),
        dt: ScalarLike = 0.01,
        method: IntegrationMethod = "RK45",
        **kwargs
    ) -> SimulationResult:
        """
        High-level simulation interface with regular time grid.

        This method wraps integrate() and post-processes the result to provide
        a regular time grid and cleaner output. This is the recommended method
        for most use cases (currently, may be deprecated once DiscretizedSystem is
        developed).

        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        controller : Optional[Callable[[float, StateVector], ControlVector]]
            Feedback controller u = controller(t, x)
            Uses (t, x) signature matching scipy/ODE conventions
            If None, uses zero control (open-loop)
        t_span : tuple[float, float]
            Simulation time interval (t_start, t_end)
        dt : float
            Output time step for regular grid (default: 0.01)
        method : str
            Integration method passed to integrate()
        **kwargs
            Additional arguments passed to integrate()

        Returns
        -------
        SimulationResult
            TypedDict (returns as dict) containing:
            - time: Regular time points (n_steps,) with spacing dt
            - states: State trajectory (nx, n_steps) - **NOTE: (nx, T) for backward compat**
            - controls: Control trajectory (nu, n_steps) if controller provided
            - metadata: Additional information (method, dt, success, etc.)

        Notes
        -----
        Unlike integrate(), this method:
        - Returns states on a regular time grid (not adaptive)
        - Supports state-feedback controllers
        - Hides solver diagnostics (cleaner output)
        - Is easier to use for plotting and analysis

        **IMPORTANT**: This method returns states in (nx, T) shape for backward
        compatibility with existing code, even though integrate() returns (T, nx).
        The transpose happens internally.

        Controller Signature
        --------------------
        Controllers must have signature (t, x) -> u:
        - t: float - current time
        - x: StateVector - current state
        - Returns: ControlVector - control input

        This matches scipy.integrate conventions where time is the first argument.

        Examples
        --------
        Open-loop simulation:

        >>> result = system.simulate(x0, t_span=(0, 5), dt=0.01)
        >>> plt.plot(result["time"], result["states"][0, :])  # Note: states[0, :]
        >>> plt.xlabel("Time (s)")
        >>> plt.ylabel("State")

        Closed-loop with state feedback:

        >>> K = np.array([[-1.0, -2.0]])  # LQR gain
        >>> def controller(t, x):
        ...     return -K @ x
        >>> result = system.simulate(x0, controller, t_span=(0, 5))

        Time-varying reference tracking:

        >>> def controller(t, x):
        ...     x_ref = np.array([np.sin(t), np.cos(t)])
        ...     return K @ (x_ref - x)
        >>> result = system.simulate(x0, controller, t_span=(0, 10))

        Adaptive gain controller:

        >>> def controller(t, x):
        ...     K = 1.0 + 0.1 * t  # Gain increases with time
        ...     return np.array([-K * x[0]])
        >>> result = system.simulate(x0, controller, t_span=(0, 10))

        Saturated control:

        >>> def controller(t, x):
        ...     u = -2.0 * x[0] - 0.5 * x[1]
        ...     return np.array([np.clip(u, -1.0, 1.0)])
        >>> result = system.simulate(x0, controller, t_span=(0, 10))
        """
        
        # Create regular time grid
        t_regular = np.arange(t_span[0], t_span[1] + dt, dt)
        
        # Pass controller directly - it already has (t, x) signature
        # No wrapper needed, integrate() expects control functions as (t, x)
        u_func = controller
        
        # Call low-level integrate() with regular time grid
        int_result = self.integrate(
            x0=x0,
            u=u_func,
            t_span=t_span,
            t_eval=t_regular,  # Request specific times
            method=method,
            **kwargs
        )
        
        # Handle both 'x' and 'y' keys (different integrator conventions)
        if "x" in int_result:
            states_time_major = int_result["x"]  # (T, nx) - our convention
        elif "y" in int_result:
            # scipy convention: (nx, T) - transpose to (T, nx)
            states_time_major = int_result["y"].T
        else:
            raise KeyError("Integration result missing both 'x' and 'y' keys")
        
        # Transpose to (nx, T) for backward compatibility
        states_regular = states_time_major.T  # (nx, T)
        
        # If times don't exactly match requested grid, interpolate
        if not np.allclose(int_result["t"], t_regular, rtol=1e-10):
            # Need to interpolate to exact regular grid
            nx = states_regular.shape[0]
            states_interp = np.zeros((nx, len(t_regular)))
            
            # Interpolate each state dimension
            for i in range(nx):
                states_interp[i, :] = np.interp(
                    t_regular,
                    int_result["t"],
                    states_time_major[:, i]  # Use time-major for interpolation
                )
            
            states_regular = states_interp
        
        # Reconstruct control trajectory if controller provided
        controls = None
        if controller is not None:
            # Evaluate controller at each time point
            controls_list = []
            for i, t in enumerate(t_regular):
                # Extract state at this time (states_regular is (nx, T))
                x_t = states_regular[:, i]
                
                # Call controller with (t, x) signature
                u_t = controller(t, x_t)
                controls_list.append(u_t)
            
            controls = np.array(controls_list).T  # (nu, T)
        
        return {
            "time": t_regular,
            "states": states_regular,  # (nx, T) for backward compatibility
            "controls": controls,
            "metadata": {
                "method": method,
                "dt": dt,
                "success": int_result.get("success", True),
                "nfev": int_result.get("nfev", None)
            }
        }

    # =========================================================================
    # Properties (Optional, can be overridden by subclasses)
    # =========================================================================

    @property
    def is_continuous(self) -> bool:
        """Return True (this is a continuous-time system)."""
        return True

    @property
    def is_discrete(self) -> bool:
        """Return False (this is NOT a discrete-time system)."""
        return False

    @property
    def is_stochastic(self) -> bool:
        """
        Return True if system has stochastic dynamics.

        Default: False (deterministic)
        Override in stochastic subclasses.
        """
        return False

    @property
    def is_time_varying(self) -> bool:
        """
        Return True if system dynamics depend explicitly on time.

        Default: False (time-invariant)
        Override in time-varying subclasses.
        """
        return False

    def __repr__(self) -> str:
        """
        String representation of the system.

        Returns
        -------
        str
            Human-readable description

        Examples
        --------
        >>> print(system)
        ContinuousSymbolicSystem(nx=2, nu=1, ny=2)
        """
        class_name = self.__class__.__name__
        nx = getattr(self, 'nx', '?')
        nu = getattr(self, 'nu', '?')
        ny = getattr(self, 'ny', '?')
        return f"{class_name}(nx={nx}, nu={nu}, ny={ny})"
