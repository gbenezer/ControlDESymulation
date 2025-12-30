# Ideal Architecture Refactoring Plan
## ControlDESymulation Library

---

## Executive Summary

This document outlines a comprehensive refactoring to achieve clean separation of concerns through strategic multiple inheritance. The goal is to separate:

1. **Time-domain behavior** (continuous vs discrete)
2. **Symbolic machinery** (symbolic expressions, code generation, parameters)
3. **Stochastic extensions** (drift + diffusion)

**Key Insight**: `ContinuousSystemBase` and `DiscreteSystemBase` already exist as abstract interfaces. We add `SymbolicSystemBase` and use multiple inheritance to combine them.

---

## Target Architecture

```
                    ┌─────────────────────┐
                    │ SymbolicSystemBase  │
                    │  (ABC)              │
                    │ - Symbolic vars     │
                    │ - Parameters        │
                    │ - Code generation   │
                    │ - Backend mgmt      │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
    ┌───────────▼────────────┐   ┌───────────▼────────────┐
    │ ContinuousSystemBase   │   │ DiscreteSystemBase     │
    │  (ABC - EXISTING)      │   │  (ABC - EXISTING)      │
    │ - __call__(x,u,t)→dx/dt│   │ - step(x,u,k)→x[k+1]   │
    │ - integrate()          │   │ - simulate()           │
    │ - linearize()→(A,B)    │   │ - linearize()→(Ad,Bd)  │
    └───────────┬────────────┘   └───────────┬────────────┘
                │                            │
                │                            │
    ┌───────────▼────────────┐   ┌───────────▼────────────┐
    │ ContinuousDynamical    │   │ DiscreteDynamical      │
    │ System                 │   │ System                 │
    │ (Concrete)             │   │ (Concrete)             │
    │ Inherits from:         │   │ Inherits from:         │
    │ - SymbolicSystemBase   │   │ - SymbolicSystemBase   │
    │ - ContinuousSystemBase │   │ - DiscreteSystemBase   │
    └───────────┬────────────┘   └───────────┬────────────┘
                │                            │
                │                            │
    ┌───────────▼────────────┐   ┌───────────▼────────────┐
    │ ContinuousStochastic   │   │ DiscreteStochastic     │
    │ System                 │   │ System                 │
    │ (Concrete)             │   │ (Concrete)             │
    │ + Diffusion machinery  │   │ + Diffusion machinery  │
    └────────────────────────┘   └────────────────────────┘
```

### Key Design Principles

1. **Single Responsibility**: Each base class has ONE clear purpose
2. **Interface Segregation**: Abstract bases define contracts, not implementations
3. **Composition Over Inheritance**: Use delegation to specialized handlers
4. **Open/Closed**: Easy to extend (new system types) without modifying base classes
5. **Liskov Substitution**: Any subclass can replace its parent in algorithms

---

## Detailed Class Responsibilities

### 1. SymbolicSystemBase (NEW)

**Purpose**: Provide symbolic machinery for ANY symbolic system (continuous, discrete, hybrid, etc.)

**Location**: `src/systems/base/symbolic_system_base.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
import sympy as sp
from src.systems.base.utils.backend_manager import BackendManager
from src.systems.base.utils.code_generator import CodeGenerator
from src.systems.base.utils.symbolic_validator import SymbolicValidator
from src.systems.base.utils.equilibrium_handler import EquilibriumHandler

class SymbolicSystemBase(ABC):
    """
    Abstract base class for symbolic dynamical systems.
    
    Provides symbolic machinery independent of time domain:
    - Symbolic variable management
    - Parameter handling and substitution
    - Multi-backend code generation (NumPy/PyTorch/JAX)
    - Backend management and device placement
    - Equilibrium point management
    - Configuration persistence
    
    This class does NOT define:
    - Time domain semantics (continuous vs discrete)
    - System interface (__call__, integrate, simulate, etc.)
    - Linearization semantics (continuous A vs discrete Ad)
    
    Subclasses must:
    1. Implement define_system() to populate symbolic expressions
    2. Inherit from appropriate time-domain base (Continuous/DiscreteSystemBase)
    3. Implement time-domain-specific methods
    
    Examples
    --------
    Correct usage (multiple inheritance):
    >>> class MyContinuousSystem(SymbolicSystemBase, ContinuousSystemBase):
    ...     def define_system(self, ...):
    ...         # Define symbolic expressions
    ...         pass
    ...     
    ...     def __call__(self, x, u, t):
    ...         # Implement continuous interface
    ...         return self._dynamics.evaluate(x, u)
    
    Incorrect usage (missing time-domain base):
    >>> class BadSystem(SymbolicSystemBase):  # Missing ContinuousSystemBase!
    ...     pass  # Won't work - no time-domain interface
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize symbolic system using template method pattern.
        
        DO NOT call super().__init__() here - that's handled by concrete classes
        that inherit from both SymbolicSystemBase and TimedomainBase.
        """
        # ====================================================================
        # Symbolic Definition Containers (populated by define_system())
        # ====================================================================
        
        self.state_vars: List[sp.Symbol] = []
        """State variables as SymPy symbols [x1, x2, ...]"""
        
        self.control_vars: List[sp.Symbol] = []
        """Control variables as SymPy symbols [u1, u2, ...]"""
        
        self.output_vars: List[sp.Symbol] = []
        """Output variables (optional, for custom outputs)"""
        
        self.parameters: Dict[sp.Symbol, float] = {}
        """System parameters: {m: 1.0, k: 10.0, ...}"""
        
        self._f_sym: Optional[sp.Matrix] = None
        """
        Symbolic dynamics (interpretation depends on subclass):
        - Continuous: dx/dt = f(x, u)
        - Discrete: x[k+1] = f(x[k], u[k])
        """
        
        self._h_sym: Optional[sp.Matrix] = None
        """Symbolic output: y = h(x). If None, defaults to y = x"""
        
        self.order: int = 1
        """System order (1 = first-order, 2 = second-order, etc.)"""
        
        # ====================================================================
        # Composed Components (delegation pattern)
        # ====================================================================
        
        self.backend = BackendManager(default_backend="numpy", default_device="cpu")
        """Backend manager: NumPy/PyTorch/JAX detection and conversion"""
        
        self._validator: Optional[SymbolicValidator] = None
        """Validator: checks system definition correctness"""
        
        self.equilibria = EquilibriumHandler(nx=0, nu=0)
        """Equilibrium handler: manages multiple equilibrium points"""
        
        self._code_gen: Optional[CodeGenerator] = None
        """Code generator: symbolic → numerical functions"""
        
        self._initialized: bool = False
        """Tracks successful initialization"""
        
        # ====================================================================
        # Template Method Pattern
        # ====================================================================
        
        # Step 1: User defines system
        self.define_system(*args, **kwargs)
        
        # Step 2: Validate symbolic definition
        self._validator = SymbolicValidator(self)
        self._validator.validate(raise_on_error=True)
        
        # Step 3: Update component dimensions
        self.equilibria.nx = self.nx
        self.equilibria.nu = self.nu
        
        # Step 4: Initialize code generation
        self._initialized = True
        self._code_gen = CodeGenerator(self)
        
        # Note: DynamicsEvaluator, LinearizationEngine, etc. are initialized
        # by concrete subclasses after time-domain interface is established
    
    @abstractmethod
    def define_system(self, *args, **kwargs):
        """
        Define symbolic system (implemented by users).
        
        Must populate:
        - self.state_vars: List[sp.Symbol]
        - self.control_vars: List[sp.Symbol]
        - self._f_sym: sp.Matrix (dynamics)
        - self.parameters: Dict[sp.Symbol, float]
        
        Optional:
        - self.output_vars: List[sp.Symbol]
        - self._h_sym: sp.Matrix (output function)
        - self.order: int (default 1)
        """
        pass
    
    # ========================================================================
    # Dimension Properties
    # ========================================================================
    
    @property
    def nx(self) -> int:
        """Number of states."""
        return len(self.state_vars)
    
    @property
    def nu(self) -> int:
        """Number of controls."""
        return len(self.control_vars)
    
    @property
    def ny(self) -> int:
        """Number of outputs."""
        if self.output_vars:
            return len(self.output_vars)
        elif self._h_sym is not None:
            return self._h_sym.shape[0]
        else:
            return self.nx
    
    @property
    def nq(self) -> int:
        """Number of generalized coordinates."""
        return self.nx // self.order if self.order > 1 else self.nx
    
    # ========================================================================
    # Backend Management
    # ========================================================================
    
    @property
    def _default_backend(self) -> str:
        """Get default backend."""
        return self.backend.default_backend
    
    @_default_backend.setter
    def _default_backend(self, value: str):
        """Set default backend."""
        self.backend.set_default(value)
    
    @property
    def _preferred_device(self) -> str:
        """Get preferred device."""
        return self.backend.preferred_device
    
    @_preferred_device.setter
    def _preferred_device(self, value: str):
        """Set preferred device."""
        self.backend.to_device(value)
    
    def set_default_backend(self, backend: str, device: Optional[str] = None):
        """Set default backend and device."""
        self.backend.set_default(backend)
        if device is not None:
            self.backend.to_device(device)
    
    # ========================================================================
    # Symbolic Utilities
    # ========================================================================
    
    def substitute_parameters(self, expr: Union[sp.Expr, sp.Matrix]) -> Union[sp.Expr, sp.Matrix]:
        """Substitute numerical parameter values into symbolic expression."""
        return expr.subs(self.parameters)
    
    @abstractmethod
    def print_equations(self, simplify: bool = True):
        """
        Print symbolic equations (must use appropriate notation).
        
        Subclasses must implement with correct notation:
        - Continuous: dx/dt, t
        - Discrete: x[k+1], k
        """
        pass
    
    # ========================================================================
    # Code Generation
    # ========================================================================
    
    def compile(self, backends: Optional[List[str]] = None, verbose: bool = False, **kwargs) -> Dict[str, float]:
        """Pre-compile dynamics functions for specified backends."""
        all_timings = self._code_gen.compile_all(backends=backends, verbose=verbose, **kwargs)
        return {backend: timings.get("f") for backend, timings in all_timings.items()}
    
    def reset_caches(self, backends: Optional[List[str]] = None):
        """Clear cached compiled functions."""
        self._code_gen.reset_cache(backends)
    
    # ========================================================================
    # Configuration
    # ========================================================================
    
    def get_config_dict(self) -> Dict:
        """Get system configuration as dictionary."""
        return {
            "class_name": self.__class__.__name__,
            "parameters": {str(k): float(v) for k, v in self.parameters.items()},
            "order": self.order,
            "nx": self.nx,
            "nu": self.nu,
            "ny": self.ny,
            "default_backend": self._default_backend,
            "preferred_device": self._preferred_device,
        }
```

**What it provides**: ~1800 lines of symbolic machinery
**What it doesn't provide**: Time-domain semantics, system interface

---

### 2. ContinuousSystemBase (EXISTING - Keep As Is)

**Purpose**: Define the interface that ALL continuous-time systems must implement

**Location**: `src/systems/base/continuous_system_base.py`

**Already defines**:
- `__call__(x, u, t) -> dx/dt`: Evaluate dynamics
- `integrate(x0, u, t_span) -> IntegrationResult`: Numerical integration
- `linearize(x_eq, u_eq) -> (A, B)`: Continuous linearization
- `simulate(x0, controller, t_span) -> SimulationResult`: High-level simulation

**Properties**:
- `is_continuous = True`
- `is_discrete = False`

**Key Point**: This is already perfect! Keep it unchanged.

---

### 3. DiscreteSystemBase (EXISTING - Keep As Is)

**Purpose**: Define the interface that ALL discrete-time systems must implement

**Location**: `src/systems/base/discrete_system_base.py`

**Already defines**:
- `dt` (property): Sampling period
- `step(x, u, k) -> x[k+1]`: Single time step
- `simulate(x0, u_sequence, n_steps) -> DiscreteSimulationResult`: Multi-step simulation
- `linearize(x_eq, u_eq) -> (Ad, Bd)`: Discrete linearization
- `rollout(x0, policy, n_steps)`: Closed-loop simulation

**Properties**:
- `is_continuous = False`
- `is_discrete = True`

**Key Point**: This is already perfect! Keep it unchanged.

---

### 4. ContinuousDynamicalSystem (REPLACES SymbolicDynamicalSystem)

**Purpose**: Concrete symbolic continuous-time dynamical system

**Location**: `src/systems/continuous_dynamical_system.py`

```python
from src.systems.base.symbolic_system_base import SymbolicSystemBase
from src.systems.base.continuous_system_base import ContinuousSystemBase
from src.systems.base.utils.dynamics_evaluator import DynamicsEvaluator
from src.systems.base.utils.linearization_engine import LinearizationEngine
from src.systems.base.utils.observation_engine import ObservationEngine

class ContinuousDynamicalSystem(SymbolicSystemBase, ContinuousSystemBase):
    """
    Concrete symbolic continuous-time dynamical system.
    
    Represents systems of the form:
        dx/dt = f(x, u, t)
        y = h(x)
    
    where:
        x ∈ ℝⁿˣ: State vector
        u ∈ ℝⁿᵘ: Control input
        y ∈ ℝⁿʸ: Output vector
        t ∈ ℝ: Time
    
    Users subclass and implement define_system() to specify symbolic dynamics.
    
    Inherits from:
    - SymbolicSystemBase: Symbolic machinery (variables, parameters, code generation)
    - ContinuousSystemBase: Continuous-time interface (integrate, simulate, __call__)
    
    Examples
    --------
    >>> class Pendulum(ContinuousDynamicalSystem):
    ...     def define_system(self, m=1.0, l=0.5, g=9.81):
    ...         theta, theta_dot = sp.symbols('theta theta_dot', real=True)
    ...         u = sp.symbols('u', real=True)
    ...         
    ...         self.state_vars = [theta, theta_dot]
    ...         self.control_vars = [u]
    ...         self._f_sym = sp.Matrix([
    ...             theta_dot,
    ...             -(g/l)*sp.sin(theta) + u/(m*l**2)
    ...         ])
    ...         self.parameters = {m: m, l: l, g: g}
    ...         self.order = 1
    ...
    >>> system = Pendulum(m=2.0, l=1.0)
    >>> x = np.array([0.1, 0.0])
    >>> u = np.array([0.0])
    >>> dx = system(x, u)  # Returns dx/dt
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize continuous dynamical system.
        
        Order matters for multiple inheritance:
        1. SymbolicSystemBase.__init__() - handles symbolic setup
        2. ContinuousSystemBase.__init__() - sets continuous properties
        """
        # Initialize symbolic machinery (this calls define_system internally)
        SymbolicSystemBase.__init__(self, *args, **kwargs)
        
        # ContinuousSystemBase is abstract - no __init__ to call
        # Just verify we satisfy the interface
        
        # Initialize time-domain-specific evaluators
        self._dynamics = DynamicsEvaluator(self, self._code_gen, self.backend)
        self._linearization = LinearizationEngine(self, self._code_gen, self.backend)
        self._observation = ObservationEngine(self, self._code_gen, self.backend)
    
    # ========================================================================
    # ContinuousSystemBase Interface Implementation
    # ========================================================================
    
    def __call__(self, x, u=None, t: float = 0.0):
        """
        Evaluate continuous-time dynamics: dx/dt = f(x, u, t).
        
        Parameters
        ----------
        x : ArrayLike
            State vector (nx,)
        u : Optional[ArrayLike]
            Control vector (nu,)
        t : float
            Time (currently ignored for time-invariant systems)
        
        Returns
        -------
        ArrayLike
            State derivative dx/dt
        """
        return self._dynamics.evaluate(x, u, backend=None)
    
    def integrate(self, x0, u, t_span=(0.0, 10.0), method="RK45", **kwargs):
        """
        Low-level numerical integration with ODE solver diagnostics.
        
        Returns IntegrationResult with adaptive time points and solver info.
        """
        # Implementation uses scipy.integrate.solve_ivp or similar
        # This is where the actual continuous-time integration happens
        from scipy.integrate import solve_ivp
        
        # Convert control input to function
        if u is None:
            u_func = lambda t: np.zeros(self.nu) if self.nu > 0 else None
        elif callable(u):
            u_func = u
        else:
            u_func = lambda t: u
        
        # Define RHS for ODE solver
        def rhs(t, x):
            u_val = u_func(t)
            return self(x, u_val, t)
        
        # Solve ODE
        result = solve_ivp(
            rhs, t_span, x0, method=method, **kwargs
        )
        
        return {
            "t": result.t,
            "y": result.y,
            "success": result.success,
            "message": result.message,
            "nfev": result.nfev,
            "njev": result.njev if hasattr(result, 'njev') else 0,
            "nlu": result.nlu if hasattr(result, 'nlu') else 0,
            "status": result.status,
        }
    
    def linearize(self, x_eq, u_eq=None):
        """
        Compute continuous-time linearization: A = ∂f/∂x, B = ∂f/∂u.
        
        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            (A, B) where A is continuous-time state matrix
        """
        return self._linearization.compute(x_eq, u_eq, backend=None)
    
    # ========================================================================
    # Convenience Methods
    # ========================================================================
    
    def forward(self, x, u=None, backend=None):
        """
        Alias for __call__ with backend specification.
        
        Returns dx/dt = f(x, u).
        """
        return self._dynamics.evaluate(x, u, backend)
    
    def linearized_dynamics(self, x, u=None, backend=None):
        """
        Alias for linearize with backend specification.
        
        Returns (A, B) continuous-time Jacobians.
        """
        return self._linearization.compute(x, u, backend)
    
    def print_equations(self, simplify: bool = True):
        """Print equations using continuous notation."""
        print("=" * 70)
        print(f"{self.__class__.__name__} (Continuous-Time)")
        print("=" * 70)
        print(f"State Variables: {self.state_vars}")
        print(f"Control Variables: {self.control_vars}")
        print(f"Dimensions: nx={self.nx}, nu={self.nu}, ny={self.ny}")
        
        print("\nDynamics: dx/dt = f(x, u)")
        for var, expr in zip(self.state_vars, self._f_sym):
            expr_sub = self.substitute_parameters(expr)
            if simplify:
                expr_sub = sp.simplify(expr_sub)
            print(f"  d{var}/dt = {expr_sub}")
        
        if self._h_sym is not None:
            print("\nOutput: y = h(x)")
            for i, expr in enumerate(self._h_sym):
                expr_sub = self.substitute_parameters(expr)
                if simplify:
                    expr_sub = sp.simplify(expr_sub)
                print(f"  y[{i}] = {expr_sub}")
        print("=" * 70)
```

**Key Features**:
- Multiple inheritance from both bases
- Implements full continuous interface
- ~300 lines total (vs 2000+ before)
- Clear semantic meaning: dx/dt

---

### 5. DiscreteDynamicalSystem (REPLACES DiscreteSymbolicSystem)

**Purpose**: Concrete symbolic discrete-time dynamical system

**Location**: `src/systems/discrete_dynamical_system.py`

```python
from src.systems.base.symbolic_system_base import SymbolicSystemBase
from src.systems.base.discrete_system_base import DiscreteSystemBase
from src.systems.base.utils.dynamics_evaluator import DynamicsEvaluator
from src.systems.base.utils.linearization_engine import LinearizationEngine

class DiscreteDynamicalSystem(SymbolicSystemBase, DiscreteSystemBase):
    """
    Concrete symbolic discrete-time dynamical system.
    
    Represents difference equations:
        x[k+1] = f(x[k], u[k])
        y[k] = h(x[k])
    
    where:
        x[k] ∈ ℝⁿˣ: State at discrete time k
        u[k] ∈ ℝⁿᵘ: Control input at time k
        y[k] ∈ ℝⁿʸ: Output at time k
        k ∈ ℤ: Discrete time index
    
    Users subclass and implement define_system() to specify symbolic dynamics.
    
    Inherits from:
    - SymbolicSystemBase: Symbolic machinery (variables, parameters, code generation)
    - DiscreteSystemBase: Discrete-time interface (step, simulate, rollout)
    
    Examples
    --------
    >>> class DiscreteLinear(DiscreteDynamicalSystem):
    ...     def define_system(self, a=0.9, b=0.1, dt=0.01):
    ...         x = sp.symbols('x', real=True)
    ...         u = sp.symbols('u', real=True)
    ...         
    ...         self.state_vars = [x]
    ...         self.control_vars = [u]
    ...         self._f_sym = sp.Matrix([a*x + b*u])
    ...         self.parameters = {a: a, b: b}
    ...         self._dt = dt
    ...         self.order = 1
    ...
    >>> system = DiscreteLinear(a=0.95, dt=0.1)
    >>> x_k = np.array([1.0])
    >>> u_k = np.array([0.0])
    >>> x_next = system.step(x_k, u_k)  # Returns x[k+1]
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize discrete dynamical system."""
        # Initialize symbolic machinery
        SymbolicSystemBase.__init__(self, *args, **kwargs)
        
        # DiscreteSystemBase is abstract - no __init__ to call
        
        # Initialize time-domain-specific evaluators
        self._dynamics = DynamicsEvaluator(self, self._code_gen, self.backend)
        self._linearization = LinearizationEngine(self, self._code_gen, self.backend)
        
        # Discrete systems must specify dt (sampling period)
        if not hasattr(self, '_dt'):
            raise ValueError(
                f"{self.__class__.__name__} must define self._dt in define_system(). "
                "Example: self._dt = 0.01"
            )
    
    # ========================================================================
    # DiscreteSystemBase Interface Implementation
    # ========================================================================
    
    @property
    def dt(self) -> float:
        """Sampling period / time step."""
        return self._dt
    
    def step(self, x, u=None, k: int = 0):
        """
        Compute next state: x[k+1] = f(x[k], u[k]).
        
        Parameters
        ----------
        x : ArrayLike
            Current state x[k]
        u : Optional[ArrayLike]
            Control input u[k]
        k : int
            Time step (currently ignored for time-invariant systems)
        
        Returns
        -------
        ArrayLike
            Next state x[k+1]
        """
        return self._dynamics.evaluate(x, u, backend=None)
    
    def simulate(self, x0, u_sequence=None, n_steps=100, **kwargs):
        """
        Simulate system for multiple discrete time steps.
        
        Returns DiscreteSimulationResult with state trajectory.
        """
        import numpy as np
        
        # Initialize trajectory storage
        states = np.zeros((self.nx, n_steps + 1))
        states[:, 0] = x0
        
        # Handle different control input types
        if u_sequence is None:
            u_func = lambda k: np.zeros(self.nu) if self.nu > 0 else None
        elif callable(u_sequence):
            u_func = u_sequence
        elif isinstance(u_sequence, np.ndarray) and u_sequence.ndim == 1:
            # Constant control
            u_func = lambda k: u_sequence
        else:
            # Sequence of controls
            u_func = lambda k: u_sequence[k] if k < len(u_sequence) else u_sequence[-1]
        
        # Simulate forward
        x = x0
        controls = []
        for k in range(n_steps):
            u = u_func(k)
            controls.append(u)
            x = self.step(x, u, k)
            states[:, k + 1] = x
        
        controls_array = np.array(controls).T if controls and controls[0] is not None else None
        
        return {
            "states": states,
            "controls": controls_array,
            "time_steps": np.arange(n_steps + 1),
            "dt": self.dt,
            "metadata": {"method": "discrete_step", "success": True}
        }
    
    def linearize(self, x_eq, u_eq=None):
        """
        Compute discrete-time linearization: Ad = ∂f/∂x, Bd = ∂f/∂u.
        
        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            (Ad, Bd) where Ad is discrete state transition matrix
        """
        return self._linearization.compute(x_eq, u_eq, backend=None)
    
    def print_equations(self, simplify: bool = True):
        """Print equations using discrete notation."""
        print("=" * 70)
        print(f"{self.__class__.__name__} (Discrete-Time, dt={self.dt})")
        print("=" * 70)
        print(f"State Variables: {self.state_vars}")
        print(f"Control Variables: {self.control_vars}")
        print(f"Dimensions: nx={self.nx}, nu={self.nu}, ny={self.ny}")
        
        print("\nDynamics: x[k+1] = f(x[k], u[k])")
        for var, expr in zip(self.state_vars, self._f_sym):
            expr_sub = self.substitute_parameters(expr)
            if simplify:
                expr_sub = sp.simplify(expr_sub)
            print(f"  {var}[k+1] = {expr_sub}")
        
        if self._h_sym is not None:
            print("\nOutput: y[k] = h(x[k])")
            for i, expr in enumerate(self._h_sym):
                expr_sub = self.substitute_parameters(expr)
                if simplify:
                    expr_sub = sp.simplify(expr_sub)
                print(f"  y[{i}][k] = {expr_sub}")
        print("=" * 70)
```

**Key Features**:
- Multiple inheritance from both bases
- Implements full discrete interface
- ~300 lines total (vs 450+ before)
- Clear semantic meaning: x[k+1]

---

### 6. ContinuousStochasticSystem (REPLACES StochasticDynamicalSystem)

**Purpose**: Extend continuous systems with stochastic dynamics

**Location**: `src/systems/continuous_stochastic_system.py`

```python
from src.systems.continuous_dynamical_system import ContinuousDynamicalSystem
from src.systems.base.utils.stochastic.diffusion_handler import DiffusionHandler
from src.systems.base.utils.stochastic.noise_analysis import NoiseCharacterizer

class ContinuousStochasticSystem(ContinuousDynamicalSystem):
    """
    Symbolic continuous-time stochastic dynamical system (SDE).
    
    Represents stochastic differential equations:
        dx = f(x, u, t)dt + g(x, u, t)dW
    
    where:
        f: Drift (deterministic part) - inherited from parent
        g: Diffusion (stochastic part) - added by this class
        dW: Brownian motion increments
    
    Users must define both drift (_f_sym) and diffusion (diffusion_expr)
    in define_system().
    
    Examples
    --------
    >>> class OrnsteinUhlenbeck(ContinuousStochasticSystem):
    ...     def define_system(self, alpha=1.0, sigma=0.5):
    ...         x = sp.symbols('x', real=True)
    ...         u = sp.symbols('u', real=True)
    ...         
    ...         # Drift (deterministic)
    ...         self.state_vars = [x]
    ...         self.control_vars = [u]
    ...         self._f_sym = sp.Matrix([-alpha*x + u])
    ...         
    ...         # Diffusion (stochastic)
    ...         self.diffusion_expr = sp.Matrix([[sigma]])
    ...         self.sde_type = 'ito'
    ...         
    ...         self.parameters = {alpha: alpha, sigma: sigma}
    ...         self.order = 1
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize stochastic system."""
        # Initialize deterministic base
        super().__init__(*args, **kwargs)
        
        # Initialize stochastic-specific components
        self._diffusion = DiffusionHandler(self, self._code_gen)
        self._noise_analyzer = NoiseCharacterizer(self)
        
        # Analyze noise structure
        self.noise_type = self._noise_analyzer.classify_noise()
    
    @property
    def is_stochastic(self) -> bool:
        """Return True (this is a stochastic system)."""
        return True
    
    def linearize(self, x_eq, u_eq=None):
        """
        Compute linearization including diffusion.
        
        Returns
        -------
        Tuple[ArrayLike, ArrayLike, ArrayLike]
            (A, B, G) where:
            - A: State Jacobian ∂f/∂x
            - B: Control Jacobian ∂f/∂u
            - G: Diffusion matrix g(x_eq, u_eq)
        """
        A, B = super().linearize(x_eq, u_eq)
        G = self._diffusion.evaluate(x_eq, u_eq)
        return (A, B, G)
```

---

### 7. DiscreteStochasticSystem (MOSTLY UNCHANGED)

**Purpose**: Extend discrete systems with stochastic dynamics

**Location**: `src/systems/discrete_stochastic_system.py`

```python
from src.systems.discrete_dynamical_system import DiscreteDynamicalSystem
from src.systems.base.utils.stochastic.diffusion_handler import DiffusionHandler

class DiscreteStochasticSystem(DiscreteDynamicalSystem):
    """
    Symbolic discrete-time stochastic dynamical system.
    
    Represents stochastic difference equations:
        x[k+1] = f(x[k], u[k]) + g(x[k], u[k]) * w[k]
    
    where:
        f: Deterministic next state
        g: Noise gain matrix
        w[k]: IID standard normal noise
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize discrete stochastic system."""
        super().__init__(*args, **kwargs)
        self._diffusion = DiffusionHandler(self, self._code_gen)
    
    @property
    def is_stochastic(self) -> bool:
        """Return True (this is a stochastic system)."""
        return True
    
    def linearize(self, x_eq, u_eq=None):
        """Return (Ad, Bd, Gd) with diffusion."""
        Ad, Bd = super().linearize(x_eq, u_eq)
        Gd = self._diffusion.evaluate(x_eq, u_eq)
        return (Ad, Bd, Gd)
```

---

### 8. Discretization Strategy

**Purpose**: Convert continuous systems to discrete approximations

**Option A: Wrapper Class (Recommended)**

```python
# Location: src/systems/discretization/discretized_system.py

class DiscretizedSystem(DiscreteSystemBase):
    """
    Discrete approximation of a continuous system.
    
    Wraps a ContinuousSystemBase and provides discrete interface
    through numerical discretization (Euler, RK4, ZOH, etc.).
    
    Examples
    --------
    >>> continuous_sys = Pendulum(m=1.0, l=0.5)
    >>> discrete_sys = DiscretizedSystem(
    ...     continuous_sys,
    ...     dt=0.01,
    ...     method='rk4'
    ... )
    >>> x_next = discrete_sys.step(x, u)
    """
    
    def __init__(
        self,
        continuous_system: ContinuousSystemBase,
        dt: float = 0.01,
        method: str = 'euler'
    ):
        """
        Discretize continuous system.
        
        Parameters
        ----------
        continuous_system : ContinuousSystemBase
            Continuous system to discretize
        dt : float
            Sampling period
        method : str
            Discretization method:
            - 'euler': Forward Euler (1st order)
            - 'rk4': Runge-Kutta 4 (4th order)
            - 'zoh': Zero-order hold (exact for linear systems)
            - 'bilinear': Tustin/bilinear transform
        """
        self._continuous_system = continuous_system
        self._dt = dt
        self._method = method
        
        # Initialize discretization method
        if method == 'euler':
            self._discretize = self._euler_step
        elif method == 'rk4':
            self._discretize = self._rk4_step
        elif method == 'zoh':
            self._discretize = self._zoh_step
        else:
            raise ValueError(f"Unknown discretization method: {method}")
    
    @property
    def dt(self) -> float:
        """Sampling period."""
        return self._dt
    
    def step(self, x, u=None, k: int = 0):
        """Compute x[k+1] using discretized continuous dynamics."""
        return self._discretize(x, u)
    
    def _euler_step(self, x, u):
        """Forward Euler: x[k+1] = x[k] + dt * f(x[k], u[k])."""
        dx = self._continuous_system(x, u, t=0.0)
        return x + self._dt * dx
    
    def _rk4_step(self, x, u):
        """Runge-Kutta 4th order."""
        dt = self._dt
        k1 = self._continuous_system(x, u, t=0.0)
        k2 = self._continuous_system(x + 0.5*dt*k1, u, t=0.0)
        k3 = self._continuous_system(x + 0.5*dt*k2, u, t=0.0)
        k4 = self._continuous_system(x + dt*k3, u, t=0.0)
        return x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def _zoh_step(self, x, u):
        """
        Zero-order hold discretization.
        
        For linear systems: Ad = exp(A*dt), Bd = A^-1(Ad - I)B
        For nonlinear: falls back to RK4
        """
        # Check if system is linear (has linearize method)
        if hasattr(self._continuous_system, 'linearize'):
            try:
                A, B = self._continuous_system.linearize(x, u)
                import scipy.linalg
                Ad = scipy.linalg.expm(A * self._dt)
                return Ad @ x + self._compute_Bd(A, Ad, B) @ u if u is not None else Ad @ x
            except:
                # Fall back to RK4 for nonlinear
                return self._rk4_step(x, u)
        else:
            return self._rk4_step(x, u)
    
    def simulate(self, x0, u_sequence=None, n_steps=100, **kwargs):
        """Simulate discretized system."""
        # Delegate to parent's default implementation
        return super().simulate(x0, u_sequence, n_steps, **kwargs)
    
    def linearize(self, x_eq, u_eq=None):
        """
        Linearize discretized system.
        
        Returns discrete-time (Ad, Bd) matrices.
        """
        if hasattr(self._continuous_system, 'linearize'):
            A, B = self._continuous_system.linearize(x_eq, u_eq)
            
            # Convert to discrete using ZOH
            import scipy.linalg
            Ad = scipy.linalg.expm(A * self._dt)
            
            # Bd = A^-1 (Ad - I) B
            nx = A.shape[0]
            I = np.eye(nx)
            A_inv = scipy.linalg.inv(A) if np.linalg.det(A) != 0 else np.zeros_like(A)
            Bd = A_inv @ (Ad - I) @ B if B is not None else np.zeros((nx, 0))
            
            return (Ad, Bd)
        else:
            raise NotImplementedError("Continuous system does not support linearization")
```

**Option B: Mixin Class**

```python
class DiscretizationMixin:
    """Mixin to add discretization to continuous systems."""
    
    def discretize(self, dt: float, method: str = 'euler'):
        """Return discrete approximation of this system."""
        return DiscretizedSystem(self, dt=dt, method=method)

# Then ContinuousSystemBase could inherit from this mixin
```

---

## Migration Path

### Phase 1: Create SymbolicSystemBase (Week 1)
1. Create `src/systems/base/symbolic_system_base.py`
2. Extract ~1800 lines from SymbolicDynamicalSystem
3. Make forward(), linearized_dynamics(), print_equations() abstract
4. Add comprehensive documentation

### Phase 2: Create ContinuousDynamicalSystem (Week 1)
1. Create `src/systems/continuous_dynamical_system.py`
2. Inherit from (SymbolicSystemBase, ContinuousSystemBase)
3. Implement continuous interface (~300 lines)
4. Add migration alias: `SymbolicDynamicalSystem = ContinuousDynamicalSystem`

### Phase 3: Create DiscreteDynamicalSystem (Week 2)
1. Create `src/systems/discrete_dynamical_system.py`
2. Inherit from (SymbolicSystemBase, DiscreteSystemBase)
3. Implement discrete interface (~300 lines)
4. Add migration alias: `DiscreteSymbolicSystem = DiscreteDynamicalSystem`

### Phase 4: Refactor Stochastic Systems (Week 2)
1. Rename `StochasticDynamicalSystem` → `ContinuousStochasticSystem`
2. Change parent from `SymbolicDynamicalSystem` → `ContinuousDynamicalSystem`
3. Rename `DiscreteStochasticSystem` and update parent
4. Verify diffusion machinery works unchanged

### Phase 5: Add Discretization (Week 3)
1. Create `src/systems/discretization/discretized_system.py`
2. Implement Euler, RK4, ZOH methods
3. Add convenience method to ContinuousSystemBase
4. Test against hand-coded discrete systems

### Phase 6: Deprecation and Cleanup (Week 3)
1. Add deprecation warnings to old names
2. Update all examples and tests
3. Update documentation
4. Remove old files (after 1-2 releases)

---

## Benefits Summary

### Code Quality
- **-1800 lines duplication**: SymbolicSystemBase extracts common code
- **+Clean separation**: Time domain vs symbolic machinery
- **+Maintainability**: Changes propagate correctly
- **+Testability**: Each layer can be tested independently

### Architecture
- **+SOLID principles**: Single responsibility, interface segregation
- **+Composition**: Delegation to specialized handlers
- **+Open/Closed**: Easy to add new system types
- **+Liskov substitution**: Proper subtyping

### User Experience
- **+Clearer names**: ContinuousDynamicalSystem vs SymbolicDynamicalSystem
- **+Better docs**: Each class documents its domain
- **+Type hints**: Clear contracts via abstract methods
- **+Flexibility**: Mix symbolic with data-driven systems

### Future Extensions
- **+Data-driven systems**: Inherit only from Continuous/DiscreteSystemBase
- **+Hybrid systems**: Combine discrete and continuous
- **+Neural systems**: Replace symbolic with neural networks
- **+Multi-rate**: Different sampling rates
- **+Switched systems**: Mode-dependent dynamics

---

## File Structure (Final)

```
src/systems/
├── base/
│   ├── symbolic_system_base.py           # NEW - Abstract symbolic machinery
│   ├── continuous_system_base.py         # EXISTING - Keep unchanged
│   ├── discrete_system_base.py           # EXISTING - Keep unchanged
│   └── utils/                            # EXISTING - Keep unchanged
│       ├── backend_manager.py
│       ├── code_generator.py
│       ├── dynamics_evaluator.py
│       ├── linearization_engine.py
│       └── ...
├── continuous_dynamical_system.py        # NEW - Replaces symbolic_dynamical_system.py
├── discrete_dynamical_system.py          # NEW - Replaces discrete_symbolic_system.py
├── continuous_stochastic_system.py       # RENAMED from stochastic_dynamical_system.py
├── discrete_stochastic_system.py         # UPDATED to use new parent
└── discretization/
    ├── __init__.py
    └── discretized_system.py             # NEW - Continuous → Discrete wrapper
```

---

## Success Criteria

1. ✅ All tests pass unchanged
2. ✅ Zero code duplication between time domains
3. ✅ Clear separation: time domain vs symbolic
4. ✅ No performance regression
5. ✅ Stochastic systems work unchanged
6. ✅ Documentation is clearer
7. ✅ Easy to add data-driven systems
8. ✅ Discretization works for any continuous system

---

## Conclusion

This architecture achieves:
- **Perfect separation of concerns**
- **Maximal code reuse**
- **Clear semantic domains**
- **Easy extensibility**
- **Type safety through abstract interfaces**
- **Zero breaking changes** (via aliases during migration)

This is the **ideal architecture** for your library. The multiple inheritance is justified and clean - each base class has a single, clear responsibility.

Ready to implement? I can help create each component.
