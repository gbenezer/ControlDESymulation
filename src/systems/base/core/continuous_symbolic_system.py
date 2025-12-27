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
Continuous Symbolic System - Layer 2 Implementation
===================================================

Multi-backend symbolic dynamical system with automatic code generation.

This module provides ContinuousSymbolicSystem, which inherits from
ContinuousSystemBase and adds symbolic mathematics capabilities:

- Symbolic system definition using SymPy
- Automatic code generation for NumPy, PyTorch, and JAX
- Multi-backend execution with zero-copy overhead
- Symbolic differentiation for linearization
- Parameter substitution and equilibrium management

Architecture
-----------
Layer 1: ContinuousSystemBase (abstract interface)
Layer 2: ContinuousSymbolicSystem (this module) ← symbolic implementation
Layer 3: ContinuousStochasticSystem (stochastic extension)

Key Features
-----------
- Write once, run anywhere: NumPy/PyTorch/JAX backends
- Symbolic â†' numerical code generation with caching
- Higher-order system support (1st, 2nd, ... nth order)
- GPU acceleration for PyTorch/JAX
- JIT compilation support
- Verification tools for symbolic vs autodiff

Examples
--------
>>> import sympy as sp
>>> import numpy as np
>>> 
>>> class SimplePendulum(ContinuousSymbolicSystem):
...     def define_system(self, m=1.0, l=1.0, g=9.81):
...         # State variables: theta, theta_dot
...         theta, theta_dot = sp.symbols('theta theta_dot', real=True)
...         # Control: torque
...         u = sp.symbols('u', real=True)
...         # Parameters
...         m_sym, l_sym, g_sym = sp.symbols('m l g', positive=True)
...         
...         # Define state and control
...         self.state_vars = [theta, theta_dot]
...         self.control_vars = [u]
...         
...         # Dynamics: theta_ddot = -g/l*sin(theta) + u/(m*l^2)
...         theta_ddot = -g_sym/l_sym * sp.sin(theta) + u/(m_sym * l_sym**2)
...         self._f_sym = sp.Matrix([theta_ddot])
...         
...         # Parameters
...         self.parameters = {m_sym: m, l_sym: l, g_sym: g}
...         self.order = 2  # Second-order system
... 
>>> # Create system
>>> system = SimplePendulum(m=1.0, l=1.0, g=9.81)
>>> 
>>> # Evaluate dynamics
>>> x = np.array([0.1, 0.0])  # Small angle, zero velocity
>>> u = np.array([0.0])       # No torque
>>> dxdt = system(x, u)       # Returns dx/dt
>>> 
>>> # Linearize at equilibrium
>>> A, B = system.linearize(x_eq=np.zeros(2), u_eq=np.zeros(1))
>>> 
>>> # Integrate ODE
>>> result = system.integrate(
...     x0=np.array([0.5, 0.0]),
...     u=None,
...     t_span=(0.0, 10.0),
...     method='RK45'
... )
>>> 
>>> # Simulate with regular grid
>>> sim_result = system.simulate(
...     x0=np.array([0.5, 0.0]),
...     controller=None,
...     t_span=(0.0, 10.0),
...     dt=0.01
... )
"""

import copy
import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import sympy as sp

# Import base class
from src.systems.base.core.continuous_system_base import ContinuousSystemBase

# Import type definitions
from src.types.core import ArrayLike, ControlVector, StateVector
from src.types.linearization import LinearizationResult
from src.types.trajectories import IntegrationResult, SimulationResult

# Import utility modules (same as SymbolicDynamicalSystem)
from src.systems.base.utils.backend_manager import BackendManager
from src.systems.base.utils.code_generator import CodeGenerator
from src.systems.base.utils.dynamics_evaluator import DynamicsEvaluator
from src.systems.base.utils.equilibrium_handler import EquilibriumHandler
from src.systems.base.utils.linearization_engine import LinearizationEngine
from src.systems.base.utils.observation_engine import ObservationEngine
from src.systems.base.utils.symbolic_validator import SymbolicValidator, ValidationError

if TYPE_CHECKING:
    import jax.numpy as jnp
    import torch


class ContinuousSymbolicSystem(ContinuousSystemBase):
    """
    Continuous-time symbolic dynamical system with multi-backend execution.
    
    Extends ContinuousSystemBase with symbolic mathematics capabilities.
    Systems are defined symbolically using SymPy and automatically compiled
    to efficient numerical implementations for NumPy, PyTorch, and JAX.
    
    The system is defined by:
        dx/dt = f(x, u)  - Dynamics (required)
        y = h(x)         - Output (optional, defaults to identity)
    
    Where:
        x ∈ ℝⁿˣ is the state vector
        u ∈ ℝⁿᵘ is the control input  
        y ∈ ℝⁿʸ is the output vector
    
    Subclasses must implement the `define_system()` method to specify
    symbolic dynamics and parameters.
    
    Attributes
    ----------
    state_vars : List[sp.Symbol]
        State variables as SymPy symbols
    control_vars : List[sp.Symbol]
        Control variables as SymPy symbols
    output_vars : List[sp.Symbol]
        Output variables (optional)
    parameters : Dict[sp.Symbol, float]
        System parameters (Symbol → numeric value)
    _f_sym : sp.Matrix
        Symbolic dynamics expression
    _h_sym : Optional[sp.Matrix]
        Symbolic output expression (None = identity)
    order : int
        System order (1 = first-order, 2 = second-order, etc.)
    backend : BackendManager
        Backend management component
    equilibria : EquilibriumHandler
        Equilibrium point management
    
    Properties
    ----------
    nx : int
        State dimension
    nu : int
        Control dimension
    ny : int
        Output dimension
    is_continuous : bool
        Always True (continuous-time system)
    is_discrete : bool
        Always False (not discrete-time)
    is_stochastic : bool
        False for this class (True for ContinuousStochasticSystem)
    is_time_varying : bool
        Whether dynamics depend explicitly on time
    
    Methods (Abstract Implementation Required)
    -------------------------------------------
    define_system(*args, **kwargs)
        Define symbolic system (must be implemented by subclasses)
    
    Methods (Inherited from ContinuousSystemBase)
    --------------------------------------------
    __call__(x, u, t) -> StateVector
        Evaluate dynamics: dx/dt = f(x, u, t)
    integrate(x0, u, t_span, method, **kwargs) -> IntegrationResult
        Low-level ODE integration with solver diagnostics
    simulate(x0, controller, t_span, dt, **kwargs) -> SimulationResult
        High-level simulation with regular time grid
    linearize(x_eq, u_eq) -> LinearizationResult
        Compute linearization: (A, B) tuple
    
    Additional Methods (Symbolic-Specific)
    -------------------------------------
    forward(x, u, backend) -> ArrayLike
        Evaluate dynamics with explicit backend selection
    linearized_dynamics(x, u, backend) -> Tuple[ArrayLike, ArrayLike]
        Compute numerical Jacobians (A, B)
    linearized_dynamics_symbolic(x_eq, u_eq) -> Tuple[sp.Matrix, sp.Matrix]
        Compute symbolic Jacobians
    set_default_backend(backend, device) -> Self
        Change default backend and device
    compile(backends, verbose, **kwargs) -> Dict[str, float]
        Pre-compile functions for specified backends
    
    Examples
    --------
    See module docstring for complete example.
    
    Notes
    -----
    - State variables must be SymPy Symbol objects
    - Parameters must use Symbol keys: {m: 1.0} not {'m': 1.0}
    - System order must divide state dimension: nx % order == 0
    - For nth-order systems, f(x, u) returns only highest derivative
    - All backends produce numerically equivalent results
    
    See Also
    --------
    ContinuousSystemBase : Abstract base class
    ContinuousStochasticSystem : Stochastic extension (SDE systems)
    BackendManager : Backend detection and conversion
    CodeGenerator : Symbolic to numerical code generation
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize continuous symbolic system.
        
        This constructor follows template method pattern:
        1. Initialize symbolic containers
        2. Create specialized components
        3. Call user-defined define_system()
        4. Validate system definition
        5. Initialize remaining components
        
        Parameters
        ----------
        *args : tuple
            Positional arguments passed to define_system()
        **kwargs : dict
            Keyword arguments passed to define_system()
        
        Raises
        ------
        ValidationError
            If system definition is invalid
        
        Notes
        -----
        Subclasses should NOT override __init__.
        Instead, implement define_system() to specify the system.
        """
        
        # ====================================================================
        # Symbolic Definition Containers
        # ====================================================================
        
        self.state_vars: List[sp.Symbol] = []
        """State variables as SymPy Symbol objects"""
        
        self.control_vars: List[sp.Symbol] = []
        """Control variables as SymPy Symbol objects"""
        
        self.output_vars: List[sp.Symbol] = []
        """Output variables (optional)"""
        
        self.parameters: Dict[sp.Symbol, float] = {}
        """System parameters: Symbol → numeric value"""
        
        self._f_sym: Optional[sp.Matrix] = None
        """Symbolic dynamics: dx/dt = f(x, u)"""
        
        self._h_sym: Optional[sp.Matrix] = None
        """Symbolic output: y = h(x)"""
        
        self.order: int = 1
        """System order: 1 (first-order), 2 (second-order), etc."""
        
        # ====================================================================
        # Component Initialization
        # ====================================================================
        
        self.backend = BackendManager(default_backend="numpy", default_device="cpu")
        """Backend manager for detection, conversion, device placement"""
        
        self._validator: Optional[SymbolicValidator] = None
        """Validator checks system definition correctness"""
        
        self.equilibria = EquilibriumHandler(nx=0, nu=0)
        """Equilibrium handler (dimensions updated after validation)"""
        
        self._code_gen: Optional[CodeGenerator] = None
        """Code generator (initialized after validation)"""
        
        self._initialized: bool = False
        """Initialization and validation status flag"""
        
        # ====================================================================
        # Template Method: Define → Validate → Initialize
        # ====================================================================
        
        # Step 1: Call user-defined system definition
        self.define_system(*args, **kwargs)
        
        # Step 2: Validate system definition
        self._validator = SymbolicValidator(self)
        try:
            validation_result = self._validator.validate(raise_on_error=True)
        except ValidationError as e:
            raise ValidationError(
                f"Validation failed for {self.__class__.__name__}:\n{str(e)}"
            ) from e
        
        # Step 3: Update equilibrium handler dimensions
        self.equilibria.nx = self.nx
        self.equilibria.nu = self.nu
        
        # Step 4: Mark as initialized
        self._initialized = True
        
        # Step 5: Initialize dependent components
        self._code_gen = CodeGenerator(self)
        self._dynamics = DynamicsEvaluator(self, self._code_gen, self.backend)
        self._linearization = LinearizationEngine(self, self._code_gen, self.backend)
        self._observation = ObservationEngine(self, self._code_gen, self.backend)
    
    # ========================================================================
    # Abstract Method from define_system pattern (not from base class)
    # ========================================================================
    
    def define_system(self, *args, **kwargs):
        """
        Define the symbolic system (must be implemented by subclasses).
        
        This method must populate the following attributes:
        
        Required
        --------
        - self.state_vars : List[sp.Symbol]
            State variables (e.g., [x, y, theta])
        - self.control_vars : List[sp.Symbol]
            Control variables (e.g., [u1, u2])
        - self._f_sym : sp.Matrix
            Symbolic dynamics (column vector)
        - self.parameters : Dict[sp.Symbol, float]
            Parameter values with Symbol keys (NOT strings!)
        
        Optional
        --------
        - self._h_sym : sp.Matrix
            Output function (defaults to identity if None)
        - self.output_vars : List[sp.Symbol]
            Output variables (for custom outputs)
        - self.order : int
            System order (defaults to 1 if not set)
        
        Parameters
        ----------
        *args : tuple
            System-specific positional arguments
        **kwargs : dict
            System-specific keyword arguments
        
        Raises
        ------
        NotImplementedError
            If not implemented by subclass
        ValidationError
            If system definition is invalid (caught by __init__)
        
        Examples
        --------
        >>> class LinearSystem(ContinuousSymbolicSystem):
        ...     def define_system(self, a=1.0):
        ...         x = sp.symbols('x', real=True)
        ...         u = sp.symbols('u', real=True)
        ...         a_sym = sp.symbols('a', real=True, positive=True)
        ...         
        ...         self.state_vars = [x]
        ...         self.control_vars = [u]
        ...         self._f_sym = sp.Matrix([-a_sym * x + u])
        ...         self.parameters = {a_sym: a}
        ...         self.order = 1
        
        Notes
        -----
        - Use sp.Symbol for ALL symbolic variables
        - Parameter dict keys must be Symbols, not strings
        - For nth-order systems, f returns ONLY highest derivative
        - System will be validated automatically after this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement define_system() method"
        )
    
    # ========================================================================
    # Properties
    # ========================================================================
    
    @property
    def nx(self) -> int:
        """State dimension."""
        return len(self.state_vars)
    
    @property
    def nu(self) -> int:
        """Control dimension."""
        return len(self.control_vars)
    
    @property
    def ny(self) -> int:
        """Output dimension."""
        if self._h_sym is not None:
            return len(self.output_vars) if self.output_vars else self._h_sym.shape[0]
        return self.nx  # Identity output
    
    @property
    def _default_backend(self) -> str:
        """Default backend name."""
        return self.backend.default_backend
    
    @property
    def _preferred_device(self) -> str:
        """Preferred device."""
        return self.backend.preferred_device
    
    # ========================================================================
    # ContinuousSystemBase Abstract Methods - MUST IMPLEMENT
    # ========================================================================
    
    def __call__(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        t: float = 0.0
    ) -> StateVector:
        """
        Evaluate continuous-time dynamics: dx/dt = f(x, u, t).
        
        This method makes the system object callable and provides the
        primary interface for forward dynamics evaluation.
        
        Parameters
        ----------
        x : StateVector
            Current state vector (nx,) or (nx, n_batch)
        u : Optional[ControlVector]
            Control input vector (nu,) or (nu, n_batch)
            If None, assumes zero control or autonomous dynamics
        t : float
            Current time (default: 0.0)
            Currently ignored (time-invariant systems)
            Future: will be passed to time-varying systems
        
        Returns
        -------
        StateVector
            Time derivative dx/dt with same shape as x
        
        Notes
        -----
        - Backend is auto-detected from input array type
        - For autonomous systems, u can be None
        - Batch evaluation supported: x.shape = (nx, n_batch)
        - Time parameter t currently ignored (may be used in future)
        
        Examples
        --------
        Single evaluation:
        
        >>> x = np.array([1.0, 2.0])
        >>> u = np.array([0.5])
        >>> dxdt = system(x, u)
        
        Autonomous system:
        
        >>> dxdt = system(x)  # u=None
        
        Batch evaluation:
        
        >>> x_batch = np.random.randn(2, 100)
        >>> u_batch = np.random.randn(1, 100)
        >>> dxdt_batch = system(x_batch, u_batch)
        """
        # Delegate to forward() method
        # Note: t parameter currently ignored for time-invariant systems
        return self.forward(x, u, backend=None)
    
    def integrate(
        self,
        x0: StateVector,
        u: Union[ControlVector, Callable[[float], ControlVector], None] = None,
        t_span: Tuple[float, float] = (0.0, 10.0),
        method: str = "RK45",
        **integrator_kwargs
    ) -> IntegrationResult:
        """
        Low-level numerical integration using integrator infrastructure.
        
        Numerically solve the initial value problem:
            dx/dt = f(x, u, t)
            x(t0) = x0
        
        This method uses the IntegratorFactory to create the appropriate
        integrator based on backend and method, then converts the result
        to the TypedDict format expected by ContinuousSystemBase.
        
        Parameters
        ----------
        x0 : StateVector
            Initial state (nx,)
        u : Union[ControlVector, Callable[[float], ControlVector], None]
            Control input:
            - None: Zero control or autonomous
            - Array: Constant control u(t) = u
            - Callable: Time-varying control u(t) = u_func(t)
        t_span : Tuple[float, float]
            Time interval (t_start, t_end)
        method : str
            Integration method (default: 'RK45')
            Scipy: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
            Julia: 'Tsit5', 'Vern9', 'Rodas5', etc.
            PyTorch: 'dopri5', 'dopri8', etc.
            JAX: 'tsit5', 'dopri5', etc.
        **integrator_kwargs
            Additional arguments passed to integrator
            Common: rtol, atol, max_step, t_eval, dense_output
        
        Returns
        -------
        IntegrationResult
            Dictionary (TypedDict) containing:
            - t: Time points (n_points,) - adaptive, chosen by solver
            - y: State trajectory (nx, n_points)
            - success: bool - whether integration succeeded
            - message: str - solver status message
            - nfev: int - number of function evaluations
            - njev: int - number of Jacobian evaluations
            - nlu: int - number of LU decompositions
            - status: int - termination status code
        
        Notes
        -----
        - Uses IntegratorFactory for backend/method selection
        - Supports numpy (scipy, julia), torch (torchdiffeq), jax (diffrax)
        - Backend auto-detected from x0
        - Time points are adaptive, not regular grid
        - For regular grid, use simulate() instead
        
        Examples
        --------
        Basic integration:
        
        >>> x0 = np.array([1.0, 0.0])
        >>> result = system.integrate(x0, u=None, t_span=(0, 10))
        >>> print(f"Success: {result['success']}")
        >>> print(f"Function evals: {result['nfev']}")
        
        Time-varying control:
        
        >>> def u_func(t):
        ...     return np.array([np.sin(t)])
        >>> result = system.integrate(x0, u_func, t_span=(0, 10))
        
        PyTorch with GPU:
        
        >>> x0_torch = torch.tensor([1.0, 0.0], device='cuda')
        >>> result = system.integrate(x0_torch, method='dopri5')
        
        Julia high-order solver:
        
        >>> result = system.integrate(x0, method='Vern9', rtol=1e-10)
        """
        # TODO(TD-001): Remove this adapter when integrators return TypedDict
        # Target: Phase 2.5
        # See: TECHNICAL_DEBT.md#TD-001
        
        # Import integrator infrastructure
        from src.systems.base.numerical_integration.integrator_factory import (
            IntegratorFactory
        )
        
        # Detect backend from x0
        backend = self.backend.detect_backend(x0)
        
        # Create appropriate integrator using factory
        # Factory handles method/backend validation and selection
        integrator = IntegratorFactory.create(
            system=self,
            backend=backend,
            method=method,
            **integrator_kwargs
        )
        
        # Convert control input to callable format
        # Integrators expect: u_func(t, x) -> u
        if u is None:
            # Autonomous or zero control
            u_func = lambda t, x: None
        elif callable(u):
            # Time-varying control function
            # Assume signature is u(t) and wrap to u(t, x)
            # More sophisticated signature detection could be added
            u_func = lambda t, x: u(t)
        else:
            # Constant control
            u_const = u
            u_func = lambda t, x: u_const
        
        # Call integrator's integrate method
        # Returns: integrator_base.IntegrationResult (class instance)
        integrator_result = integrator.integrate(
            x0=x0,
            u_func=u_func,
            t_span=t_span,
            t_eval=integrator_kwargs.get('t_eval', None),
            dense_output=integrator_kwargs.get('dense_output', False)
        )
        
        # TODO(TD-001): This conversion will be unnecessary after refactoring
        # Currently converts: class (n_points, nx) -> TypedDict (nx, n_points)
        
        # Convert class result to TypedDict format
        # CRITICAL: Integrator returns x.shape = (n_points, nx)
        #           TypedDict expects y.shape = (nx, n_points) - TRANSPOSE!
        result: IntegrationResult = {
            "t": integrator_result.t,
            "y": integrator_result.x.T,  # TRANSPOSE: (n_points, nx) -> (nx, n_points)
            "success": integrator_result.success,
            "message": integrator_result.message,
            "nfev": integrator_result.nfev,
            "njev": integrator_result.metadata.get("njev", 0),
            "nlu": integrator_result.metadata.get("nlu", 0),
            "status": 0 if integrator_result.success else -1
        }
        
        return result
    
    def linearize(
        self,
        x_eq: StateVector,
        u_eq: Optional[ControlVector] = None
    ) -> LinearizationResult:
        """
        Compute linearized dynamics around an equilibrium point.
        
        Returns Jacobian matrices:
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
            
            For this class (deterministic), always returns (A, B).
            Stochastic systems override to return (A, B, G).
        
        Notes
        -----
        - Returns TUPLE not dict or dataclass
        - Backend is auto-detected from x_eq
        - For symbolic Jacobians, use linearized_dynamics_symbolic()
        - Equilibrium should satisfy f(x_eq, u_eq) ≈ 0
        
        Examples
        --------
        Linearize at origin:
        
        >>> x_eq = np.zeros(2)
        >>> u_eq = np.zeros(1)
        >>> A, B = system.linearize(x_eq, u_eq)
        >>> print(f"A matrix:\\n{A}")
        >>> print(f"B matrix:\\n{B}")
        
        Check stability:
        
        >>> eigenvalues = np.linalg.eigvals(A)
        >>> is_stable = np.all(np.real(eigenvalues) < 0)
        
        Design LQR controller:
        
        >>> from scipy.linalg import solve_continuous_are
        >>> P = solve_continuous_are(A, B, Q, R)
        >>> K = np.linalg.inv(R) @ B.T @ P
        """
        # Use existing linearization engine
        A, B = self._linearization.compute_dynamics(x_eq, u_eq, backend=None)
        
        # Return as tuple (CRITICAL: must be tuple, not dict)
        return (A, B)
    
    # ========================================================================
    # Additional Methods for Symbolic Systems
    # ========================================================================
    
    def forward(
        self,
        x: ArrayLike,
        u: Optional[ArrayLike] = None,
        backend: Optional[str] = None
    ) -> ArrayLike:
        """
        Evaluate forward dynamics with explicit backend selection.
        
        Computes dx/dt = f(x, u) using specified or auto-detected backend.
        
        Parameters
        ----------
        x : ArrayLike
            State vector (nx,) or (nx, n_batch)
        u : Optional[ArrayLike]
            Control vector (nu,) or (nu, n_batch)
            None for autonomous systems
        backend : Optional[str]
            Backend to use: 'numpy', 'torch', 'jax', or None (auto-detect)
        
        Returns
        -------
        ArrayLike
            State derivative dx/dt in same backend as input
        
        Examples
        --------
        >>> dxdt = system.forward(x, u)  # Auto-detect
        >>> dxdt = system.forward(x, u, backend='torch')  # Force PyTorch
        """
        return self._dynamics.evaluate(x, u, backend)
    
    def linearized_dynamics(
        self,
        x: Union[ArrayLike, str],
        u: Optional[ArrayLike] = None,
        backend: Optional[str] = None
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Compute numerical linearization: A = ∂f/∂x, B = ∂f/∂u.
        
        Evaluates Jacobian matrices at given state and control point.
        
        Parameters
        ----------
        x : Union[ArrayLike, str]
            State to linearize at, OR equilibrium name
        u : Optional[ArrayLike]
            Control to linearize at (ignored if x is equilibrium name)
        backend : Optional[str]
            Backend selection (None = auto-detect)
        
        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            (A, B) where:
            - A: State Jacobian ∂f/∂x, shape (nx, nx)
            - B: Control Jacobian ∂f/∂u, shape (nx, nu)
        
        Examples
        --------
        Direct linearization:
        
        >>> A, B = system.linearized_dynamics(x, u)
        
        Named equilibrium:
        
        >>> A, B = system.linearized_dynamics('upright')
        """
        # If x is string, treat as equilibrium name
        if isinstance(x, str):
            equilibrium_name = x
            backend = backend or self._default_backend
            x, u = self.equilibria.get_both(equilibrium_name, backend)
        
        return self._linearization.compute_dynamics(x, u, backend)
    
    def linearized_dynamics_symbolic(
        self,
        x_eq: Optional[Union[sp.Matrix, str]] = None,
        u_eq: Optional[sp.Matrix] = None
    ) -> Tuple[sp.Matrix, sp.Matrix]:
        """
        Compute symbolic linearization: A = ∂f/∂x, B = ∂f/∂u.
        
        Returns symbolic Jacobian matrices for analytical analysis.
        
        Parameters
        ----------
        x_eq : Optional[Union[sp.Matrix, str]]
            Equilibrium state (symbolic), equilibrium name, or None (origin)
        u_eq : Optional[sp.Matrix]
            Equilibrium control (symbolic) or None (zeros)
        
        Returns
        -------
        Tuple[sp.Matrix, sp.Matrix]
            (A_sym, B_sym) symbolic Jacobian matrices
        
        Examples
        --------
        >>> A_sym, B_sym = system.linearized_dynamics_symbolic()
        >>> print(sp.latex(A_sym))  # LaTeX for paper
        """
        return self._linearization.compute_dynamics_symbolic(x_eq, u_eq)
    
    # ========================================================================
    # Backend Management
    # ========================================================================
    
    def set_default_backend(
        self, backend: str, device: Optional[str] = None
    ) -> "ContinuousSymbolicSystem":
        """
        Set default backend and device for all operations.
        
        Parameters
        ----------
        backend : str
            Backend name: 'numpy', 'torch', or 'jax'
        device : Optional[str]
            Device: 'cpu', 'cuda', 'cuda:0', 'gpu:0', etc.
            None = keep current device
        
        Returns
        -------
        Self
            For method chaining
        
        Examples
        --------
        >>> system.set_default_backend('torch', device='cuda:0')
        >>> system.set_default_backend('jax').compile(verbose=True)
        """
        self.backend.set_default(backend, device)
        return self
    
    def to_device(self, device: str) -> "ContinuousSymbolicSystem":
        """
        Set preferred device for PyTorch/JAX backends.
        
        Parameters
        ----------
        device : str
            Device: 'cpu', 'cuda', 'cuda:0', 'gpu:0', 'tpu:0', etc.
        
        Returns
        -------
        Self
            For method chaining
        
        Examples
        --------
        >>> system.to_device('cuda:0')
        >>> system.to_device('cuda').set_default_backend('torch')
        """
        self.backend.to_device(device)
        if self.backend.default_backend in ["torch", "jax"]:
            self._clear_backend_cache(self.backend.default_backend)
        return self
    
    def _clear_backend_cache(self, backend: str):
        """Clear cached compiled functions for a specific backend."""
        self._code_gen.reset_cache([backend])
    
    @contextmanager
    def use_backend(self, backend: str, device: Optional[str] = None):
        """
        Temporarily switch to a different backend/device.
        
        Parameters
        ----------
        backend : str
            Temporary backend: 'numpy', 'torch', 'jax'
        device : Optional[str]
            Temporary device (None = keep current)
        
        Yields
        ------
        Self
            System with temporary backend configuration
        
        Examples
        --------
        >>> with system.use_backend('torch', device='cuda'):
        ...     dx = system(x, u)  # Uses torch on CUDA
        >>> # Back to original backend
        """
        with self.backend.use_backend(backend, device):
            yield self
    
    def compile(
        self,
        backends: Optional[List[str]] = None,
        verbose: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        """
        Pre-compile dynamics functions for specified backends.
        
        Compilation happens lazily by default. This method allows eager
        compilation to reduce first-call latency.
        
        Parameters
        ----------
        backends : Optional[List[str]]
            Backends to compile: ['numpy', 'torch', 'jax']
            None = compile all available
        verbose : bool
            Print compilation progress and timing
        **kwargs
            Backend-specific compilation options
        
        Returns
        -------
        Dict[str, float]
            Compilation times for each backend
        
        Examples
        --------
        >>> times = system.compile(verbose=True)
        >>> times = system.compile(backends=['torch', 'jax'])
        """
        return self._code_gen.compile_all(backends, verbose, **kwargs)
    
    def get_backend_info(self) -> Dict[str, any]:
        """
        Get comprehensive backend configuration and status information.
        
        Returns
        -------
        dict
            Backend configuration, availability, and compilation status
        
        Examples
        --------
        >>> info = system.get_backend_info()
        >>> print(f"Default: {info['default_backend']}")
        >>> print(f"Compiled: {info['compiled_backends']}")
        """
        info = self.backend.get_info()
        compiled = [
            backend
            for backend in ["numpy", "torch", "jax"]
            if self._code_gen.is_compiled(backend)["f"]
        ]
        info["compiled_backends"] = compiled
        info["initialized"] = self._initialized
        return info
    
    # ========================================================================
    # String Representations
    # ========================================================================
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"nx={self.nx}, nu={self.nu}, ny={self.ny}, order={self.order}, "
            f"backend={self._default_backend}, device={self._preferred_device})"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        equilibria_str = (
            f", {len(self.equilibria.list_names())} equilibria"
            if len(self.equilibria.list_names()) > 1
            else ""
        )
        return (
            f"{self.__class__.__name__}(nx={self.nx}, nu={self.nu}, "
            f"backend={self._default_backend}{equilibria_str})"
        )


# ============================================================================
# Backward Compatibility Alias
# ============================================================================

# IMPORTANT: Maintain backward compatibility
# Old code using SymbolicDynamicalSystem will continue to work
SymbolicDynamicalSystem = ContinuousSymbolicSystem

__all__ = [
    "ContinuousSymbolicSystem",
    "SymbolicDynamicalSystem",  # Backward compatibility
]
