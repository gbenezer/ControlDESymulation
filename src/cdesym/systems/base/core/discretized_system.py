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
Discretized System - Numerical Discretization of Continuous Systems
====================================================================

Provides discrete-time approximation via three modes: FIXED_STEP,
DENSE_OUTPUT, and BATCH_INTERPOLATION.

This module provides DiscretizedSystem, which wraps any ContinuousSystemBase
and provides a discrete interface through numerical integration.

Protocol Satisfaction
--------------------
DiscretizedSystem satisfies:
- ✓ DiscreteSystemProtocol (step, simulate)
- ✓ LinearizableDiscreteProtocol (linearize via ZOH)
- ✗ SymbolicDiscreteProtocol (no symbolic expressions - purely numerical)

This is CORRECT - discretization is numerical, not symbolic!

NOTE: DiscretizedSystem itself doesn't provide symbolic discrete-time expressions,
even when wrapping symbolic continuous systems.

See class docstring for complete documentation.
"""

import inspect
import time
from enum import Enum
from typing import Dict, List, Literal, Optional

import numpy as np
from scipy.interpolate import interp1d

from cdesym.systems.base.core.continuous_system_base import ContinuousSystemBase
from cdesym.systems.base.core.discrete_system_base import DiscreteSystemBase
from cdesym.systems.base.numerical_integration.integrator_factory import IntegratorFactory

# Import method registry functions
from cdesym.systems.base.numerical_integration.method_registry import (
    get_available_methods,
    is_fixed_step,
    is_sde_method,
    normalize_method_name,
    validate_method,
)
from cdesym.types.backends import Backend
from cdesym.types.core import ControlVector, DiscreteControlInput, StateVector
from cdesym.types.linearization import DiscreteLinearization
from cdesym.types.trajectories import DiscreteSimulationResult


class DiscretizationMode(Enum):
    """Three discretization modes balancing accuracy and efficiency."""

    FIXED_STEP = "fixed_step"
    DENSE_OUTPUT = "dense_output"
    BATCH_INTERPOLATION = "batch_interpolation"


class DiscretizedSystem(DiscreteSystemBase):
    """
    Pure wrapper providing discrete interface to continuous systems.

    Protocol Satisfaction
    --------------------
    This class satisfies:
    - DiscreteSystemProtocol: Has step(), simulate(), dt, nx, nu
    - LinearizableDiscreteProtocol: Has linearize() (wraps continuous)

    Does NOT satisfy:
    - SymbolicDiscreteProtocol: No symbolic machinery (purely numerical)

    This means it can be used in:
    - ✓ Any function expecting DiscreteSystemProtocol
    - ✓ Control design (LQR, MPC) expecting LinearizableDiscreteProtocol
    - ✗ Code generation expecting SymbolicDiscreteProtocol

    Examples
    --------
    >>> from controldesymulation.types.protocols import LinearizableDiscreteProtocol
    >>>
    >>> def lqr_design(system: LinearizableDiscreteProtocol, Q, R):
    ...     Ad, Bd = system.linearize(np.zeros(system.nx), np.zeros(system.nu))
    ...     # ... LQR computation
    >>>
    >>> # DiscretizedSystem works here:
    >>> continuous = Pendulum(m=1.0, l=0.5)
    >>> discrete = DiscretizedSystem(continuous, dt=0.01)
    >>> K = lqr_design(discrete, Q, R)  # ✓ Type checks pass!
    """

    @staticmethod
    def get_available_methods(
        backend: Backend = "numpy",
        method_type: Literal["all", "deterministic", "stochastic", "fixed_step", "adaptive"] = "all",
    ) -> Dict[str, List[str]]:
        """
        Get available integration methods for a backend.

        Delegates to method_registry.get_available_methods() for the
        canonical source of truth about method availability.

        Parameters
        ----------
        backend : Backend, default='numpy'
            Backend to query: 'numpy', 'torch', or 'jax'
        method_type : str, default='all'
            Filter by method type:
            - 'all': All methods (default)
            - 'deterministic': Only ODE methods
            - 'stochastic': Only SDE methods
            - 'fixed_step': Only fixed-step methods (both ODE and SDE)
            - 'adaptive': Only adaptive methods (both ODE and SDE)

        Returns
        -------
        dict
            Dictionary with method categories and their available methods

        Examples
        --------
        >>> methods = DiscretizedSystem.get_available_methods('torch', 'stochastic')
        >>> print(methods['sde_fixed_step'])
        ['euler', 'milstein', 'srk', 'midpoint']
        >>> print(methods['canonical_aliases'])
        ['euler_maruyama', 'milstein', 'sra1', 'reversible_heun', 'rk45', ...]

        See Also
        --------
        method_registry.get_available_methods : Implementation details
        """
        return get_available_methods(backend=backend, method_type=method_type)

    def __init__(
        self,
        continuous_system: ContinuousSystemBase,
        dt: float = 0.01,
        method: str = "rk4",
        mode: Optional[DiscretizationMode] = None,
        interpolation_kind: str = "linear",
        auto_detect_sde: bool = True,
        sde_method: Optional[str] = None,
        **integrator_kwargs,
    ):
        """
        Initialize discretized system wrapper.

        Parameters
        ----------
        continuous_system : ContinuousSystemBase
            Continuous system to discretize. Supports:
            - ContinuousSystemBase (deterministic)
            - ContinuousSymbolicSystem (symbolic)
            - ContinuousStochasticSystem (stochastic)
        dt : float, default=0.01
            Sampling time step (seconds)
        method : str, default='rk4'
            Integration method. Available methods depend on system type and backend.

            **Method names are automatically normalized** to match the backend's
            conventions. You can use either canonical names (recommended) or
            backend-specific names - both work seamlessly.

            **Deterministic Fixed-Step Methods:**
            - 'euler': Forward Euler (1st order)
            - 'midpoint': Midpoint/RK2 (2nd order)
            - 'rk4': Classic Runge-Kutta 4 (4th order) [DEFAULT]
            - 'heun': Heun's method (2nd order)

            **Deterministic Adaptive Methods:**

            *Canonical names (work on all backends):*
            - 'rk45': Dormand-Prince 5(4) - general purpose
            - 'rk23': Bogacki-Shampine 3(2) - lower accuracy
            - 'dopri8': Dormand-Prince 8 - high accuracy
            - 'tsit5': Tsitouras 5(4) - modern, efficient
            - 'bdf': Backward Differentiation - very stiff
            - 'lsoda': Auto stiffness detection (NumPy/Scipy only)

            **Stochastic (SDE) Methods:**

            *Canonical names (recommended - work on all backends):*
            - 'euler_maruyama': Euler-Maruyama (strong 0.5, weak 1.0)
            - 'milstein': Milstein method (strong 1.0)
            - 'stratonovich_milstein': Stratonovich interpretation
            - 'sra1': Stochastic Runge-Kutta (diagonal noise, order 1.5)
            - 'reversible_heun': Time-reversible integration

            **Note on method selection:**
            - For stochastic systems, prefer specifying `sde_method` explicitly
            - Canonical names ('euler_maruyama', 'milstein', 'rk45') are
              automatically converted to backend-appropriate names
            - Use `DiscretizedSystem.get_available_methods(backend)` to see
              all methods for a specific backend

        mode : DiscretizationMode, optional
            Discretization mode. If None, auto-selected based on method:
            - FIXED_STEP for fixed-step methods (euler, rk4, euler_maruyama, etc.)
            - DENSE_OUTPUT for adaptive methods (rk45, lsoda, etc.)
            - BATCH_INTERPOLATION must be explicitly requested
        interpolation_kind : str, default='linear'
            Interpolation method for BATCH mode:
            - 'linear': Fast, robust (default)
            - 'cubic': Smoother, requires ≥4 adaptive points
        auto_detect_sde : bool, default=True
            For stochastic systems: warn if using deterministic method.
            Set to False to suppress warnings when intentionally using
            deterministic integration on stochastic systems (ignores noise).
        sde_method : str, optional
            Explicitly specify SDE method for stochastic systems.
            Overrides method parameter. Recommended options:
            - 'euler_maruyama': General purpose, fast (canonical name)
            - 'milstein': Higher accuracy (canonical name)
            - Backend-specific names also supported (e.g., 'EM', 'ItoMilstein')

            If specified, uses this method regardless of auto_detect_sde.
            Method name is automatically normalized to backend conventions.
        **integrator_kwargs
            Additional arguments passed to integrator:
            - rtol : float (default: 1e-6) - Relative tolerance (adaptive only)
            - atol : float (default: 1e-8) - Absolute tolerance (adaptive only)
            - max_steps : int - Maximum integration steps
            - seed : int - Random seed (SDE methods only)
            - adjoint : bool - Use adjoint method (JAX/PyTorch only)

        Raises
        ------
        TypeError
            If continuous_system is not ContinuousSystemBase
        ValueError
            If dt <= 0, or invalid mode/method combination, or invalid method
            for backend/system type combination

        Examples
        --------
        **Deterministic Systems:**

        >>> # Basic deterministic system with RK4
        >>> from controldesymulation.systems.examples import Pendulum
        >>> continuous = Pendulum(m=1.0, l=0.5)
        >>> discrete = DiscretizedSystem(continuous, dt=0.01, method='rk4')
        >>> result = discrete.simulate(x0=np.array([0.1, 0.0]), n_steps=1000)

        **Stochastic Systems:**

        >>> # Stochastic system with canonical SDE method (recommended)
        >>> from controldesymulation.systems.examples import StochasticPendulum
        >>> stochastic = StochasticPendulum(m=1.0, l=0.5, sigma=0.1)
        >>>
        >>> discrete = DiscretizedSystem(
        ...     stochastic,
        ...     dt=0.01,
        ...     sde_method='euler_maruyama'  # Canonical name
        ... )
        >>> # Automatically becomes 'EM' (NumPy), 'euler' (PyTorch), or 'Euler' (JAX)

        Notes
        -----
        **Method Name Normalization:**
        Method names are automatically converted to backend-appropriate forms using
        the method_registry module. This provides:
        - Portable canonical names ('euler_maruyama', 'rk45')
        - Automatic backend-specific conversion
        - Validation before creating integrators
        - Helpful error messages with suggestions

        **Backend Detection:**
        The backend is detected from `continuous_system._default_backend`.
        If not available, defaults to 'numpy'.
        """

        # ========================================================================
        # Validation
        # ========================================================================

        if not isinstance(continuous_system, ContinuousSystemBase):
            raise TypeError(
                f"Expected ContinuousSystemBase (or subclass), "
                f"got {type(continuous_system).__name__}",
            )

        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        # ========================================================================
        # Store basic attributes
        # ========================================================================

        self._continuous_system = continuous_system
        self._dt = float(dt)
        self._interpolation_kind = interpolation_kind
        self._integrator_kwargs = integrator_kwargs
        self._original_method = method  # Store original before any modifications

        # ========================================================================
        # Detect backend and normalize method names
        # ========================================================================

        # Detect backend from system
        if hasattr(continuous_system, "_default_backend"):
            backend = continuous_system._default_backend
        else:
            backend = "numpy"

        # Normalize method names using method_registry
        method = normalize_method_name(method, backend)

        # Also normalize sde_method if provided
        if sde_method is not None:
            sde_method = normalize_method_name(sde_method, backend)

        # ========================================================================
        # Detect system type
        # ========================================================================

        self._is_stochastic = continuous_system.is_stochastic
        self._is_symbolic = hasattr(continuous_system, "_f_sym")

        # Check if SDE integration is available
        try:
            from cdesym.systems.base.numerical_integration.stochastic.sde_integrator_factory import (
                SDEIntegratorFactory,
            )

            self._has_sde_integrator = True
        except ImportError:
            self._has_sde_integrator = False

        # ========================================================================
        # Method Selection for Stochastic Systems
        # ========================================================================

        if self._is_stochastic:
            # ====================================================================
            # Method Selection Decision Tree
            # ====================================================================
            #
            # 1. sde_method explicitly provided?
            #    ├─ YES: SDE integrator available?
            #    │   ├─ YES → Use sde_method (source: 'explicit')
            #    │   └─ NO  → Warn, use deterministic method (source: 'explicit_unavailable')
            #    │
            #    └─ NO: Continue to step 2
            #
            # 2. auto_detect_sde=True AND method is deterministic?
            #    ├─ YES: Warn, recommend SDE method, keep deterministic
            #    │       (source: 'deterministic_fallback')
            #    │
            #    └─ NO: Continue to step 3
            #
            # 3. Either:
            #    - auto_detect_sde=False (user explicitly disabled detection)
            #    - method is already an SDE method
            #
            #    Action: Use method as-is (source: 'user_specified')
            #
            # ====================================================================

            if sde_method is not None:
                # User explicitly requested SDE method
                if self._has_sde_integrator:
                    # SDE integrator available - use it
                    method = sde_method
                    self._method_source = "explicit"
                else:
                    # SDE integrator not available - warn and keep deterministic
                    import warnings

                    warnings.warn(
                        f"Explicitly requested SDE method '{sde_method}', but SDE integration "
                        f"not available. Using deterministic method '{method}' - noise IGNORED.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._method_source = "explicit_unavailable"

            elif auto_detect_sde and not is_sde_method(method):
                # User gave deterministic method for stochastic system
                recommended_method = self._detect_sde_method(continuous_system)

                if self._has_sde_integrator:
                    import warnings

                    warnings.warn(
                        f"Stochastic system with deterministic method '{method}'. "
                        f"Noise will be IGNORED. Consider using sde_method='{recommended_method}' "
                        f"for proper noise handling.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    import warnings

                    warnings.warn(
                        f"Stochastic system with deterministic method '{method}'. "
                        f"SDEIntegratorFactory not available - noise will be IGNORED. "
                        f"Recommended: '{recommended_method}'.",
                        UserWarning,
                        stacklevel=2,
                    )
                self._method_source = "deterministic_fallback"

            else:
                # User explicitly chose deterministic method or turned off auto-detect,
                # or already provided an SDE method
                if not is_sde_method(method) and not auto_detect_sde:
                    import warnings

                    warnings.warn(
                        f"Using deterministic method '{method}' on stochastic system. "
                        f"Noise will be IGNORED.",
                        UserWarning,
                        stacklevel=2,
                    )
                self._method_source = "user_specified"

        else:
            # Deterministic system
            self._has_sde_integrator = False
            self._method_source = "deterministic_system"

        # CRITICAL: Set self._method AFTER all the logic above
        self._method = method

        # ========================================================================
        # Classify method using method_registry
        # ========================================================================

        self._is_fixed_step = is_fixed_step(method)

        # ========================================================================
        # Determine discretization mode
        # ========================================================================

        self._mode = (
            mode
            if mode
            else (
                DiscretizationMode.FIXED_STEP
                if self._is_fixed_step
                else DiscretizationMode.DENSE_OUTPUT
            )
        )

        # Validate mode/method compatibility
        if self._mode == DiscretizationMode.FIXED_STEP and not self._is_fixed_step:
            raise ValueError(
                f"Cannot use adaptive method '{method}' with FIXED_STEP mode. "
                f"Use mode=DiscretizationMode.DENSE_OUTPUT or choose a fixed-step method.",
            )

        # Warn about BATCH mode with stochastic systems
        if self._mode == DiscretizationMode.BATCH_INTERPOLATION and self._is_stochastic:
            import warnings

            warnings.warn(
                "BATCH_INTERPOLATION mode with stochastic system. "
                "Each call generates different trajectory due to randomness.",
                UserWarning,
                stacklevel=2,
            )

        # ========================================================================
        # Handle symbolic systems (future optimization hooks)
        # ========================================================================

        if self._is_symbolic:
            self._has_symbolic_jacobian = hasattr(continuous_system, "get_jacobian_symbolic")

            if hasattr(continuous_system, "state_vars"):
                self._symbolic_state_vars = continuous_system.state_vars
            else:
                self._symbolic_state_vars = None

            if hasattr(continuous_system, "control_vars"):
                self._symbolic_control_vars = continuous_system.control_vars
            else:
                self._symbolic_control_vars = None
        else:
            self._has_symbolic_jacobian = False
            self._symbolic_state_vars = None
            self._symbolic_control_vars = None

        # ========================================================================
        # Store metadata for get_info()
        # ========================================================================

        self._system_metadata = {
            "is_stochastic": self._is_stochastic,
            "is_symbolic": self._is_symbolic,
            "has_sde_integrator": self._has_sde_integrator,
            "has_symbolic_jacobian": self._has_symbolic_jacobian,
            "method_source": self._method_source,
            "original_method": self._original_method,
            "final_method": self._method,
        }

    def _detect_sde_method(self, system) -> str:
        """
        Detect best SDE integration method for stochastic system.

        Parameters
        ----------
        system : ContinuousSystemBase
            System to analyze (should be stochastic)

        Returns
        -------
        str
            Recommended SDE method ('euler_maruyama' or 'milstein')

        Notes
        -----
        Decision logic:
        - Additive noise → 'euler_maruyama' (simplest, exact for additive)
        - Diagonal multiplicative noise → 'milstein' (better accuracy)
        - General multiplicative noise → 'euler_maruyama' (conservative)
        """
        if not system.is_stochastic:
            return self._method

        # Check for additive noise (Euler-Maruyama is exact)
        if hasattr(system, "is_additive_noise"):
            try:
                if system.is_additive_noise():
                    return "euler_maruyama"
            except:
                pass

        # Check for diagonal noise (Milstein is efficient)
        if hasattr(system, "is_diagonal_noise"):
            try:
                if system.is_diagonal_noise():
                    return "milstein"
            except:
                pass

        # Default: Euler-Maruyama (most general, conservative)
        return "euler_maruyama"

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def mode(self) -> DiscretizationMode:
        return self._mode

    @property
    def nx(self) -> int:
        return self._continuous_system.nx

    @property
    def nu(self) -> int:
        return self._continuous_system.nu

    @property
    def ny(self) -> int:
        return self._continuous_system.ny

    @property
    def is_stochastic(self) -> bool:
        return self._continuous_system.is_stochastic

    def step(self, x: StateVector, u: Optional[ControlVector] = None, k: int = 0) -> StateVector:
        if self._mode == DiscretizationMode.BATCH_INTERPOLATION:
            raise NotImplementedError("step() not supported in BATCH_INTERPOLATION mode")

        t_start, t_end = k * self._dt, (k + 1) * self._dt
        return (
            self._step_fixed(x, u, t_start, t_end)
            if self._mode == DiscretizationMode.FIXED_STEP
            else self._step_dense(x, u, t_start, t_end)
        )

    def _step_fixed(self, x, u, t_start, t_end):
        """Single fixed-step integration from t_start to t_end."""
        # Check if we should use SDE integrator (using method_registry)
        use_sde = (
            self._is_stochastic and is_sde_method(self._method) and self._has_sde_integrator
        )

        if use_sde:
            # Use SDE integrator for stochastic systems
            try:
                from cdesym.systems.base.numerical_integration.stochastic.sde_integrator_factory import (
                    SDEIntegratorFactory,
                )

                integrator = SDEIntegratorFactory.create(
                    sde_system=self._continuous_system,
                    backend=self._continuous_system._default_backend,
                    method=self._method,
                    dt=self._dt,
                    **self._integrator_kwargs,
                )
            except (ImportError, AttributeError, ValueError) as e:
                # SDE integrator creation failed - fall back to deterministic
                import warnings

                warnings.warn(
                    f"Failed to create SDE integrator: {e}. "
                    f"Falling back to deterministic integration (noise ignored).",
                    UserWarning,
                    stacklevel=2,
                )
                integrator = IntegratorFactory.create(
                    system=self._continuous_system,
                    backend=self._continuous_system._default_backend,
                    method="rk4",
                    dt=self._dt,
                    **self._integrator_kwargs,
                )
        else:
            # Use regular integrator for deterministic systems or deterministic methods
            integrator = IntegratorFactory.create(
                system=self._continuous_system,
                backend=self._continuous_system._default_backend,
                method=self._method,
                dt=self._dt,
                **self._integrator_kwargs,
            )

        result = integrator.integrate(x0=x, u_func=lambda t, xv: u, t_span=(t_start, t_end))
        return result["x"][-1, :] if "x" in result else result["y"][:, -1]

    def _step_dense(self, x, u, t_start, t_end):
        """Single step using dense output (adaptive methods)."""
        # Check if we should use SDE integrator (using method_registry)
        use_sde = (
            self._is_stochastic and is_sde_method(self._method) and self._has_sde_integrator
        )

        if use_sde:
            # Use SDE integrator (dense output not typically supported for SDEs)
            try:
                from cdesym.systems.base.numerical_integration.stochastic.sde_integrator_factory import (
                    SDEIntegratorFactory,
                )

                integrator = SDEIntegratorFactory.create(
                    sde_system=self._continuous_system,
                    backend=self._continuous_system._default_backend,
                    method=self._method,
                    dt=self._dt,
                    **self._integrator_kwargs,
                )

                result = integrator.integrate(
                    x0=x,
                    u_func=lambda t, xv: u,
                    t_span=(t_start, t_end),
                    dense_output=False,
                )

                return result["x"][-1, :] if "x" in result else result["y"][:, -1]

            except (ImportError, AttributeError, ValueError) as e:
                import warnings

                warnings.warn(
                    f"SDE integrator failed: {e}. Using deterministic integration.",
                    UserWarning,
                    stacklevel=2,
                )

        # Use regular integrator with dense output
        integrator = IntegratorFactory.create(
            system=self._continuous_system,
            backend=self._continuous_system._default_backend,
            method=self._method,
            **self._integrator_kwargs,
        )

        result = integrator.integrate(
            x0=x,
            u_func=lambda t, xv: u,
            t_span=(t_start, t_end),
            dense_output=True,
        )

        if "sol" in result and result["sol"] is not None:
            x_end = result["sol"](t_end)
            return x_end.ravel() if x_end.ndim > 1 else x_end

        return result["x"][-1, :] if "x" in result else result["y"][:, -1]

    def simulate(
        self,
        x0: StateVector,
        u_sequence: DiscreteControlInput = None,
        n_steps: int = 100,
        **kwargs,
    ) -> DiscreteSimulationResult:
        return (
            self._simulate_batch(x0, u_sequence, n_steps)
            if self._mode == DiscretizationMode.BATCH_INTERPOLATION
            else self._simulate_step_by_step(x0, u_sequence, n_steps)
        )

    def simulate_stochastic(
        self,
        x0: StateVector,
        u_sequence: DiscreteControlInput = None,
        n_steps: int = 100,
        n_trajectories: int = 100,
        **kwargs,
    ) -> dict:
        """Simulate stochastic system with multiple Monte Carlo trajectories."""
        if not self._is_stochastic:
            raise ValueError(
                "simulate_stochastic() only available for stochastic systems. "
                "Use regular simulate() for deterministic systems.",
            )

        # Validate using method_registry
        if not is_sde_method(self._method):
            raise ValueError(
                f"simulate_stochastic() requires SDE method. "
                f"Current method '{self._method}' is deterministic. "
                f"Use sde_method parameter: DiscretizedSystem(system, sde_method='euler_maruyama')",
            )

        if not self._has_sde_integrator:
            raise ValueError(
                "SDE integrator not available. Install required package:\n"
                "  - NumPy backend: pip install diffeqpy\n"
                "  - PyTorch backend: pip install torchsde\n"
                "  - JAX backend: pip install diffrax",
            )

        # Run multiple trajectories
        all_trajectories = []
        controls_ref = None

        for traj_idx in range(n_trajectories):
            result = self.simulate(x0, u_sequence, n_steps, **kwargs)
            all_trajectories.append(result["states"])

            if controls_ref is None:
                controls_ref = result["controls"]

        # Stack trajectories: (n_trajectories, n_steps+1, nx)
        all_trajectories = np.array(all_trajectories)

        # Compute statistics
        mean_traj = np.mean(all_trajectories, axis=0)
        std_traj = np.std(all_trajectories, axis=0)

        return {
            "states": all_trajectories,
            "controls": controls_ref,
            "mean_trajectory": mean_traj,
            "std_trajectory": std_traj,
            "time_steps": result["time_steps"],
            "dt": self.dt,
            "success": True,
            "n_trajectories": n_trajectories,
            "metadata": {
                "method": self._method,
                "mode": self._mode.value,
                "is_stochastic": True,
                "convergence_type": "strong",
            },
        }

    def _simulate_step_by_step(self, x0, u_sequence, n_steps):
        states = np.zeros((n_steps + 1, self.nx))
        states[0, :] = x0
        controls = []
        u_func = self._prepare_control_sequence(u_sequence, n_steps)

        x = x0
        for k in range(n_steps):
            u = u_func(x, k)
            controls.append(u)
            x = self.step(x, u, k)
            states[k + 1, :] = x

        return {
            "states": states,
            "controls": np.array(controls) if controls and controls[0] is not None else None,
            "time_steps": np.arange(n_steps + 1),
            "dt": self.dt,
            "success": True,
            "metadata": {"method": self._method, "mode": self._mode.value},
        }

    def _simulate_batch(self, x0, u_sequence, n_steps):
        if callable(u_sequence) and len(inspect.signature(u_sequence).parameters) == 2:
            raise ValueError("State-feedback not supported in BATCH_INTERPOLATION mode")

        u_func_discrete = self._prepare_control_sequence(u_sequence, n_steps)
        u_func_continuous = lambda t, x: u_func_discrete(x, min(int(t / self.dt), n_steps - 1))

        result = self._continuous_system.integrate(
            x0=x0,
            u=u_func_continuous,
            t_span=(0.0, n_steps * self.dt),
            method=self._method,
            **self._integrator_kwargs,
        )

        trajectory = result["x"] if "x" in result else result["y"].T
        t_regular = np.arange(0, n_steps + 1) * self.dt
        states_regular = self._interpolate_trajectory(result["t"], trajectory, t_regular)

        return {
            "states": states_regular,
            "controls": None,
            "time_steps": np.arange(n_steps + 1),
            "dt": self.dt,
            "success": result.get("success", True),
            "metadata": {
                "method": self._method,
                "mode": self._mode.value,
                "nfev": result.get("nfev"),
                "adaptive_points": len(result["t"]),
            },
        }

    def _interpolate_trajectory(self, t_adaptive, y_adaptive, t_regular):
        """Interpolate trajectory from adaptive time grid to regular time grid."""
        # Transpose if necessary
        if y_adaptive.shape[0] != len(t_adaptive):
            y_adaptive = y_adaptive.T

        nx = y_adaptive.shape[1]
        y_regular = np.zeros((len(t_regular), nx))

        # Validate time grid compatibility
        FP_TOL = 1e-10
        MISMATCH_TOL = 0.01 * self._dt

        # Check start time
        t_start_diff = t_regular[0] - t_adaptive[0]
        if t_start_diff < -FP_TOL:
            raise ValueError(
                f"Regular time grid starts before adaptive grid: "
                f"t_regular[0]={t_regular[0]:.10f}, t_adaptive[0]={t_adaptive[0]:.10f}. "
                f"Difference: {t_start_diff:.2e}",
            )

        # Check end time
        t_end_diff = t_regular[-1] - t_adaptive[-1]

        if t_end_diff > MISMATCH_TOL:
            raise ValueError(
                f"Adaptive integration appears to have failed or stopped early. "
                f"Requested end time: t={t_regular[-1]:.6f}, "
                f"but integration stopped at t={t_adaptive[-1]:.6f}. "
                f"Shortfall: {t_end_diff:.6f} ({t_end_diff/self._dt:.1f} time steps). "
                f"This may indicate:\n"
                f"  - max_steps limit reached\n"
                f"  - Numerical instability in integration\n"
                f"  - Stiff system requiring different method\n"
                f"Check result['success'] and consider using a more robust integrator.",
            )

        if t_end_diff > FP_TOL:
            import warnings

            warnings.warn(
                f"Time grid mismatch detected. "
                f"Requested t_end={t_regular[-1]:.10f}, "
                f"but adaptive integration ended at t={t_adaptive[-1]:.10f}. "
                f"Difference: {t_end_diff:.2e}. "
                f"Clipping to adaptive range and extrapolating with endpoint values.",
                UserWarning,
                stacklevel=3,
            )

        if t_start_diff > FP_TOL:
            import warnings

            warnings.warn(
                f"Regular grid starts slightly after adaptive grid "
                f"(Δt={t_start_diff:.2e}). Clipping to adaptive range.",
                UserWarning,
                stacklevel=3,
            )

        # Clip to safe range
        t_regular_clipped = np.clip(t_regular, t_adaptive[0], t_adaptive[-1])

        # Choose interpolation kind
        kind = self._interpolation_kind
        if len(t_adaptive) < 4 and kind == "cubic":
            kind = "linear"

        # Perform interpolation
        for i in range(nx):
            interp = interp1d(
                t_adaptive,
                y_adaptive[:, i],
                kind=kind,
                bounds_error=False,
                fill_value=(y_adaptive[0, i], y_adaptive[-1, i]),
                assume_sorted=True,
            )
            y_regular[:, i] = interp(t_regular_clipped)

        return y_regular

    def linearize(
        self,
        x_eq: StateVector,
        u_eq: Optional[ControlVector] = None,
    ) -> DiscreteLinearization:
        lin_result = self._continuous_system.linearize(x_eq, u_eq)
        A, B = lin_result[:2]

        from scipy.linalg import expm

        nx, I = A.shape[0], np.eye(A.shape[0])
        Ad = expm(A * self.dt)

        try:
            if np.linalg.cond(A) > 1e10:
                Bd = self.dt * B
            else:
                Bd = np.linalg.inv(A) @ (Ad - I) @ B
        except np.linalg.LinAlgError:
            Bd = self.dt * B

        return (Ad, Bd)

    def _prepare_control_sequence(self, u_sequence, n_steps):
        if u_sequence is None:
            return lambda x, k: None if self.nu == 0 else np.zeros(self.nu)

        if callable(u_sequence):
            sig = inspect.signature(u_sequence)
            if len(sig.parameters) == 1:
                return lambda x, k: u_sequence(k)
            if len(sig.parameters) == 2:
                names = list(sig.parameters.keys())
                if names[0] in ["x", "state"]:
                    return u_sequence
                if names[0] in ["k", "time"]:
                    return lambda x, k: u_sequence(k, x)
                try:
                    u_sequence(np.zeros(self.nx), 0)
                    return u_sequence
                except:
                    return lambda x, k: u_sequence(k, x)
            else:
                raise TypeError(
                    f"Control function must accept 1 or 2 parameters, got {len(sig.parameters)}",
                )

        if isinstance(u_sequence, np.ndarray):
            if u_sequence.ndim == 1:
                if u_sequence.size != self.nu:
                    raise ValueError(
                        f"Control dimension mismatch: expected {self.nu}, got {u_sequence.size}",
                    )
                return lambda x, k: u_sequence
            if u_sequence.shape[0] == n_steps:
                return lambda x, k: u_sequence[k, :] if k < n_steps else u_sequence[-1, :]
            return lambda x, k: u_sequence[:, k] if k < u_sequence.shape[1] else u_sequence[:, -1]

        if isinstance(u_sequence, (list, tuple)):
            return lambda x, k: np.asarray(u_sequence[k] if k < len(u_sequence) else u_sequence[-1])

        raise TypeError(f"Invalid control type: {type(u_sequence)}")

    def compare_modes(self, x0, u_sequence, n_steps, reference_solution=None):
        if reference_solution is None:
            ref = DiscretizedSystem(
                self._continuous_system,
                dt=self.dt,
                method="LSODA",
                mode=DiscretizationMode.BATCH_INTERPOLATION,
                rtol=1e-12,
                atol=1e-14,
            )
            reference_solution = ref.simulate(x0, u_sequence, n_steps)["states"]

        results, timings, errors = {}, {}, {}

        for name, mode, method in [
            ("fixed_step", DiscretizationMode.FIXED_STEP, "rk4"),
            ("dense_output", DiscretizationMode.DENSE_OUTPUT, "RK45"),
            ("batch", DiscretizationMode.BATCH_INTERPOLATION, "RK45"),
        ]:
            sys = DiscretizedSystem(self._continuous_system, dt=self.dt, method=method, mode=mode)
            start = time.time()
            result = sys.simulate(x0, u_sequence, n_steps)
            timings[name] = time.time() - start
            results[name] = result
            errors[name] = np.sqrt(np.mean((result["states"] - reference_solution) ** 2))

        return {
            "results": results,
            "timings": timings,
            "errors": errors,
            "reference": reference_solution,
            "speedup_batch_vs_fixed": timings["fixed_step"] / timings["batch"],
            "speedup_batch_vs_dense": timings["dense_output"] / timings["batch"],
        }

    def change_method(self, new_method: str, **new_kwargs) -> "DiscretizedSystem":
        """Create new DiscretizedSystem with different method."""
        merged_kwargs = {**self._integrator_kwargs, **new_kwargs}
        return DiscretizedSystem(
            self._continuous_system,
            dt=self.dt,
            method=new_method,
            mode=None,
            interpolation_kind=self._interpolation_kind,
            **merged_kwargs,
        )

    def get_info(self) -> dict:
        """Get comprehensive discretization information and configuration."""
        info = {
            "class": "DiscretizedSystem",
            "mode": self._mode.value,
            "method": self._method,
            "dt": self.dt,
            "is_fixed_step": self._is_fixed_step,
            "interpolation": self._interpolation_kind,
            "supports_step": self._mode != DiscretizationMode.BATCH_INTERPOLATION,
            "supports_closed_loop": self._mode != DiscretizationMode.BATCH_INTERPOLATION,
            "supports_linearization": True,
            "continuous_system_type": type(self._continuous_system).__name__,
            "is_stochastic": self.is_stochastic,
            "is_symbolic": self._is_symbolic,
            "dimensions": {"nx": self.nx, "nu": self.nu, "ny": self.ny},
            "integrator_kwargs": self._integrator_kwargs,
        }

        # Method selection information
        info["method_selection"] = {
            "source": self._method_source,
            "original_method": self._original_method,
            "final_method": self._method,
            "description": self._get_method_selection_description(),
        }

        # Stochastic system information
        if self.is_stochastic:
            stochastic_info = {
                "is_stochastic": True,
                "has_sde_integrator": self._has_sde_integrator,
                "recommended_method": self._detect_sde_method(self._continuous_system),
                "noise_ignored": not is_sde_method(self._method),
            }

            # Get noise structure if available
            try:
                if hasattr(self._continuous_system, "is_additive_noise"):
                    stochastic_info["is_additive_noise"] = (
                        self._continuous_system.is_additive_noise()
                    )
                else:
                    stochastic_info["is_additive_noise"] = None
            except:
                stochastic_info["is_additive_noise"] = None

            try:
                if hasattr(self._continuous_system, "is_diagonal_noise"):
                    stochastic_info["is_diagonal_noise"] = (
                        self._continuous_system.is_diagonal_noise()
                    )
                else:
                    stochastic_info["is_diagonal_noise"] = None
            except:
                stochastic_info["is_diagonal_noise"] = None

            if hasattr(self._continuous_system, "sde_type"):
                stochastic_info["sde_type"] = self._continuous_system.sde_type
            else:
                stochastic_info["sde_type"] = "unknown"

            if hasattr(self._continuous_system, "nw"):
                stochastic_info["noise_dimension"] = self._continuous_system.nw
            else:
                stochastic_info["noise_dimension"] = None

            info["stochastic_info"] = stochastic_info

        # Symbolic system information
        if self._is_symbolic:
            symbolic_info = {
                "is_symbolic": True,
                "has_symbolic_jacobian": self._has_symbolic_jacobian,
                "can_generate_code": True,
            }

            if self._symbolic_state_vars is not None:
                try:
                    symbolic_info["state_vars"] = [str(v) for v in self._symbolic_state_vars]
                except:
                    symbolic_info["state_vars"] = None
            else:
                symbolic_info["state_vars"] = None

            if self._symbolic_control_vars is not None:
                try:
                    symbolic_info["control_vars"] = [str(v) for v in self._symbolic_control_vars]
                except:
                    symbolic_info["control_vars"] = None
            else:
                symbolic_info["control_vars"] = None

            if hasattr(self._continuous_system, "parameters"):
                try:
                    params = self._continuous_system.parameters
                    symbolic_info["parameters"] = {str(k): v for k, v in params.items()}
                except:
                    symbolic_info["parameters"] = None
            else:
                symbolic_info["parameters"] = None

            info["symbolic_info"] = symbolic_info

        # Warnings/Recommendations
        warnings = []

        # Check for stochastic + deterministic method
        if self.is_stochastic and not is_sde_method(self._method):
            if not self._has_sde_integrator:
                warnings.append(
                    "Stochastic system with deterministic integrator - noise is IGNORED. "
                    "Install SDE integration support for proper noise handling.",
                )
            elif self._method_source == "user_specified":
                warnings.append(
                    f"Using deterministic method '{self._method}' on stochastic system - "
                    f"consider using '{self._detect_sde_method(self._continuous_system)}' instead.",
                )

        # Check for BATCH mode with stochastic
        if self._mode == DiscretizationMode.BATCH_INTERPOLATION and self.is_stochastic:
            warnings.append(
                "BATCH_INTERPOLATION mode with stochastic system - "
                "each simulation produces different trajectory.",
            )

        # Check for cubic interpolation with few expected points
        if (
            self._interpolation_kind == "cubic"
            and self._mode == DiscretizationMode.BATCH_INTERPOLATION
        ):
            warnings.append(
                "Using cubic interpolation - automatic fallback to linear if <4 adaptive points.",
            )

        if warnings:
            info["warnings"] = warnings

        return info

    def _get_method_selection_description(self) -> str:
        """Get human-readable description of method selection."""
        source = self._method_source

        if source == "explicit":
            return f"Explicitly set via sde_method='{self._method}'"

        if source == "explicit_unavailable":
            return (
                f"SDE method '{self._original_method}' requested but unavailable, "
                f"using '{self._method}'"
            )

        if source == "deterministic_fallback":
            if self._has_sde_integrator:
                return f"Deterministic fallback (user specified '{self._method}')"
            return "Deterministic fallback (SDE integrator unavailable)"

        if source == "user_specified":
            if self._is_stochastic and not is_sde_method(self._method):
                return f"User-specified method '{self._method}' (auto-detection disabled)"
            return f"User-specified method '{self._method}'"

        if source == "deterministic_system":
            return "Deterministic system"

        return f"Method '{self._method}' (source: {source})"

    def print_info(self):
        """Print formatted discretization information."""
        info = self.get_info()

        print("=" * 70)
        print("DiscretizedSystem")
        print("=" * 70)

        print(f"Continuous System: {info['continuous_system_type']}")
        print(f"Discretization Method: {info['method']}")
        print(f"Mode: {info['mode'].upper()}")
        print(f"Time Step: {info['dt']}s ({1/info['dt']:.1f} Hz)")

        dims = info["dimensions"]
        print(f"Dimensions: nx={dims['nx']}, nu={dims['nu']}, ny={dims['ny']}")

        print(f"Stochastic: {info['is_stochastic']}")
        print(f"Symbolic: {info['is_symbolic']}")

        print(f"Supports step(): {info['supports_step']}")
        print(f"Supports closed-loop: {info['supports_closed_loop']}")

        print("\nMethod Selection:")
        print(f"  Source: {info['method_selection']['description']}")
        if info["method_selection"]["original_method"] != info["method_selection"]["final_method"]:
            print(f"  Original: {info['method_selection']['original_method']}")
            print(f"  Final: {info['method_selection']['final_method']}")

        if "stochastic_info" in info:
            print("\nStochastic System Info:")
            sinfo = info["stochastic_info"]
            print(f"  SDE Type: {sinfo.get('sde_type', 'unknown')}")
            print(f"  Recommended Method: {sinfo['recommended_method']}")
            print(f"  Has SDE Integrator: {sinfo['has_sde_integrator']}")
            print(f"  Noise Ignored: {sinfo['noise_ignored']}")

            if sinfo.get("is_additive_noise") is not None:
                print(f"  Additive Noise: {sinfo['is_additive_noise']}")
            if sinfo.get("is_diagonal_noise") is not None:
                print(f"  Diagonal Noise: {sinfo['is_diagonal_noise']}")
            if sinfo.get("noise_dimension") is not None:
                print(f"  Noise Dimension: {sinfo['noise_dimension']}")

        if "symbolic_info" in info:
            print("\nSymbolic System Info:")
            sinfo = info["symbolic_info"]
            print(f"  Has Symbolic Jacobian: {sinfo['has_symbolic_jacobian']}")
            print(f"  Can Generate Code: {sinfo['can_generate_code']}")

            if sinfo.get("state_vars"):
                print(f"  State Variables: {', '.join(sinfo['state_vars'])}")
            if sinfo.get("control_vars"):
                print(f"  Control Variables: {', '.join(sinfo['control_vars'])}")
            if sinfo.get("parameters"):
                print(f"  Parameters: {len(sinfo['parameters'])} defined")

        if info["integrator_kwargs"]:
            print("\nIntegrator Options:")
            for key, val in info["integrator_kwargs"].items():
                print(f"  {key}: {val}")

        if "warnings" in info:
            print("\n" + "⚠ " * 35)
            print("WARNINGS:")
            for i, warning in enumerate(info["warnings"], 1):
                print(f"{i}. {warning}")
            print("⚠ " * 35)

        print("=" * 70)

    def __repr__(self):
        return (
            f"DiscretizedSystem(dt={self.dt:.4f}, method={self._method}, mode={self._mode.value})"
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def discretize(continuous_system, dt, method="rk4", **kwargs):
    """
    Convenience wrapper for creating a discretized system.

    Creates a DiscretizedSystem with automatic mode selection based on the
    chosen integration method. This is the recommended way to discretize
    systems for step-by-step simulation and control applications.

    Parameters
    ----------
    continuous_system : ContinuousSystemBase
        Continuous system to discretize
    dt : float
        Sampling time step (seconds)
    method : str, default='rk4'
        Integration method. Can use canonical names ('euler_maruyama', 'rk45')
        or backend-specific names. Method name is automatically normalized
        for the system's backend.
    **kwargs
        Additional arguments passed to DiscretizedSystem

    Returns
    -------
    DiscretizedSystem
        Discretized system ready for simulation

    Examples
    --------
    >>> from controldesymulation.systems.examples import Pendulum
    >>> continuous = Pendulum(m=1.0, l=0.5)
    >>> discrete = discretize(continuous, dt=0.01)
    """
    return DiscretizedSystem(continuous_system, dt=dt, method=method, **kwargs)


def discretize_batch(continuous_system, dt, method="LSODA", **kwargs):
    """
    Create discretized system optimized for batch/open-loop simulation.

    Uses BATCH_INTERPOLATION mode for maximum performance on open-loop
    trajectories.

    Parameters
    ----------
    continuous_system : ContinuousSystemBase
        Continuous system to discretize
    dt : float
        Sampling time step (seconds)
    method : str, default='LSODA'
        Adaptive integration method
    **kwargs
        Additional arguments passed to DiscretizedSystem

    Returns
    -------
    DiscretizedSystem
        Discretized system in BATCH_INTERPOLATION mode

    Examples
    --------
    >>> discrete = discretize_batch(continuous, dt=0.01, method='LSODA')
    >>> result = discrete.simulate(x0, None, 1000)
    """
    return DiscretizedSystem(
        continuous_system,
        dt=dt,
        method=method,
        mode=DiscretizationMode.BATCH_INTERPOLATION,
        **kwargs,
    )


def analyze_discretization_error(
    continuous_system,
    x0,
    u_sequence,
    dt_values,
    method="rk4",
    n_steps=100,
    reference_dt=None,
):
    """Analyze discretization error vs time step for convergence study."""
    if reference_dt is None:
        reference_dt = min(dt_values) / 10

    ref = DiscretizedSystem(
        continuous_system,
        dt=reference_dt,
        method="LSODA",
        mode=DiscretizationMode.BATCH_INTERPOLATION,
        rtol=1e-12,
        atol=1e-14,
    )
    n_ref = int(n_steps * max(dt_values) / reference_dt)
    ref_result = ref.simulate(x0, u_sequence, n_ref)
    t_compare = np.arange(n_steps + 1) * max(dt_values)

    errors, timings = [], []

    for dt in dt_values:
        disc = DiscretizedSystem(continuous_system, dt=dt, method=method)
        start = time.time()
        n_curr = int(n_steps * max(dt_values) / dt)
        result = disc.simulate(x0, u_sequence, n_curr)
        timings.append(time.time() - start)

        from scipy.interpolate import interp1d

        t_curr = result["time_steps"] * dt
        t_ref = ref_result["time_steps"] * reference_dt

        t_min = max(t_curr[0], t_ref[0])
        t_max = min(t_curr[-1], t_ref[-1])
        t_compare_valid = t_compare[(t_compare >= t_min) & (t_compare <= t_max)]

        if len(t_compare_valid) == 0:
            error = np.linalg.norm(result["states"][-1, :] - ref_result["states"][-1, :])
            errors.append(error)
            continue

        states_interp = np.zeros((len(t_compare_valid), continuous_system.nx))
        ref_interp = np.zeros((len(t_compare_valid), continuous_system.nx))

        for i in range(continuous_system.nx):
            states_interp[:, i] = interp1d(t_curr, result["states"][:, i], kind="linear")(
                t_compare_valid,
            )
            ref_interp[:, i] = interp1d(t_ref, ref_result["states"][:, i], kind="linear")(
                t_compare_valid,
            )

        errors.append(np.sqrt(np.mean((states_interp - ref_interp) ** 2)))

    log_dt, log_err = np.log(dt_values), np.log(np.array(errors) + 1e-16)
    convergence_rate = np.polyfit(log_dt, log_err, 1)[0]

    return {
        "dt_values": list(dt_values),
        "errors": errors,
        "timings": timings,
        "reference": ref_result,
        "method": method,
        "convergence_rate": convergence_rate,
    }


def recommend_dt(
    continuous_system,
    x0,
    target_error=1e-6,
    method="rk4",
    dt_range=(1e-4, 0.1),
    n_test=10,
):
    """Recommend optimal time step for target discretization accuracy."""
    dt_values = np.logspace(np.log10(dt_range[0]), np.log10(dt_range[1]), n_test)
    analysis = analyze_discretization_error(continuous_system, x0, None, dt_values, method, 100)

    errors = np.array(analysis["errors"])
    valid_mask = errors < target_error

    if not np.any(valid_mask):
        import warnings

        warnings.warn(f"No dt achieves target {target_error:.2e}", UserWarning)
        best_idx = np.argmin(errors)
    else:
        best_idx = np.where(valid_mask)[0][-1]

    return {
        "recommended_dt": float(dt_values[best_idx]),
        "achieved_error": float(errors[best_idx]),
        "timing": analysis["timings"][best_idx],
        "all_results": analysis,
    }


def detect_sde_integrator(continuous_system):
    """Detect best SDE method for stochastic system."""
    if not continuous_system.is_stochastic:
        raise ValueError("System is not stochastic")

    if hasattr(continuous_system, "is_additive_noise") and continuous_system.is_additive_noise():
        return "euler_maruyama"
    if hasattr(continuous_system, "is_diagonal_noise") and continuous_system.is_diagonal_noise():
        return "milstein"
    return "euler_maruyama"


def compute_discretization_quality(discrete_system, x0, u_sequence, n_steps, metrics=None):
    """Compute quality metrics for discretization configuration."""
    if metrics is None:
        metrics = ["timing", "stability"]

    results = {}
    start = time.time()
    sim_result = discrete_system.simulate(x0, u_sequence, n_steps)
    elapsed = time.time() - start

    if "timing" in metrics:
        results["timing"] = {
            "total_time": elapsed,
            "time_per_step": elapsed / n_steps,
            "steps_per_second": n_steps / elapsed,
        }

    if "stability" in metrics:
        norms = np.linalg.norm(sim_result["states"], axis=1)
        results["stability"] = {
            "is_stable": bool(norms[-1] < 100 * norms[0]),
            "final_norm": float(norms[-1]),
            "max_norm": float(np.max(norms)),
        }

    return results


__all__ = [
    "DiscretizationMode",
    "DiscretizedSystem",
    "analyze_discretization_error",
    "compute_discretization_quality",
    "detect_sde_integrator",
    "discretize",
    "discretize_batch",
    "recommend_dt",
]