# Integration Framework Architecture
## How Numerical Integrators Fit Into the Refactored System Hierarchy

---

## Executive Summary

The numerical integration framework (IntegratorBase, IntegratorFactory, etc.) integrates cleanly into the new architecture through the **Factory Pattern**. Each continuous system class uses the appropriate factory to create integrators on-demand during `integrate()` calls.

**Key Insight**: Integrators are **tools** that systems use, not components they own. Systems create integrators as needed via factories, allowing flexible method selection per integration call.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   System Hierarchy                          │
│                                                             │
│  ContinuousSystemBase (ABC)                                │
│  ├─ integrate(x0, u, t_span, method='RK45', ...)  [abstract]│
│  │                                                          │
│  ContinuousDynamicalSystem                                 │
│  ├─ integrate() → uses IntegratorFactory                   │
│  │                                                          │
│  ContinuousStochasticSystem                                │
│  └─ integrate() → uses SDEIntegratorFactory                │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ creates via factory
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Integration Framework                          │
│                                                             │
│  IntegratorFactory.create(system, backend, method, ...)    │
│  ├─ Returns: IntegratorBase instance                       │
│  │   - ScipyIntegrator (numpy)                            │
│  │   - DiffraxIntegrator (jax)                            │
│  │   - TorchDiffEqIntegrator (torch)                      │
│  │   - DiffEqPyIntegrator (julia)                         │
│  │   - Manual integrators (RK4, Euler, etc.)              │
│  │                                                         │
│  SDEIntegratorFactory.create(system, backend, method, ...) │
│  └─ Returns: SDEIntegratorBase instance                    │
│      - TorchSDEIntegrator (torch)                          │
│      - DiffEqPySDEIntegrator (julia)                       │
│      - Manual SDE integrators (Euler-Maruyama, etc.)       │
└─────────────────────────────────────────────────────────────┘
```

---

## Design Principles

### 1. Factory Pattern (Not Composition)

**❌ BAD: Store integrator as attribute**
```python
class ContinuousDynamicalSystem:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._integrator = None  # ❌ Don't do this
    
    def integrate(self, x0, u, t_span):
        if self._integrator is None:
            self._integrator = IntegratorFactory.create(self, ...)
        return self._integrator.integrate(x0, u, t_span)
```

**✅ GOOD: Create integrator on-demand via factory**
```python
class ContinuousDynamicalSystem:
    def integrate(self, x0, u, t_span, method='RK45', **integrator_kwargs):
        # Create integrator for this specific call
        integrator = IntegratorFactory.create(
            self, 
            backend=self._default_backend,
            method=method,
            **integrator_kwargs
        )
        return integrator.integrate(x0, u, t_span)
```

**Why?**
- ✅ Different methods per call: `system.integrate(x0, u, t_span, method='RK45')` vs `method='LSODA'`
- ✅ No state management: Each call is independent
- ✅ Thread-safe: No shared mutable state
- ✅ Memory efficient: No permanent integrator storage
- ✅ Flexible: Easy to override factory in subclasses

### 2. Separation of Concerns

**Systems** (ContinuousDynamicalSystem):
- Define symbolic dynamics: `_f_sym`
- Provide evaluation interface: `__call__(x, u, t)`
- Coordinate integration: `integrate()` method
- Don't know integration details

**Integrators** (IntegratorBase subclasses):
- Implement numerical methods: RK45, LSODA, etc.
- Handle time stepping: fixed vs adaptive
- Manage backends: NumPy, PyTorch, JAX
- Don't know symbolic machinery

**Factories** (IntegratorFactory, SDEIntegratorFactory):
- Select appropriate integrator for system + backend + method
- Handle backend availability checking
- Provide intelligent defaults
- Encapsulate integrator creation logic

---

## Implementation Details

### 1. ContinuousSystemBase (Abstract Interface)

**Location**: `src/systems/base/continuous_system_base.py`

**Status**: KEEP EXACTLY AS IS - Already defines perfect interface

```python
class ContinuousSystemBase(ABC):
    """Abstract base for continuous-time systems."""
    
    @abstractmethod
    def __call__(self, x, u=None, t=0.0):
        """Evaluate dx/dt = f(x, u, t)."""
        pass
    
    @abstractmethod
    def integrate(
        self,
        x0: StateVector,
        u: ControlInput = None,
        t_span: tuple[float, float] = (0.0, 10.0),
        method: IntegrationMethod = "RK45",
        **integrator_kwargs
    ) -> IntegrationResult:
        """
        Low-level numerical integration.
        
        Parameters
        ----------
        x0 : StateVector
            Initial state
        u : ControlInput
            Control input (constant, callable, or None)
        t_span : tuple[float, float]
            Time interval
        method : str
            Integration method ('RK45', 'LSODA', 'dopri5', etc.)
        **integrator_kwargs
            Additional integrator options (rtol, atol, etc.)
        
        Returns
        -------
        IntegrationResult
            TypedDict with t, y, success, nfev, etc.
        """
        pass
```

**Key Points**:
- Already has `method` parameter for flexibility
- Returns TypedDict (IntegrationResult)
- Accepts **integrator_kwargs for rtol, atol, etc.
- Perfect interface - no changes needed!

---

### 2. ContinuousDynamicalSystem (Concrete Implementation)

**Location**: `src/systems/continuous_dynamical_system.py`

```python
from src.systems.base.symbolic_system_base import SymbolicSystemBase
from src.systems.base.continuous_system_base import ContinuousSystemBase
from src.systems.base.numerical_integration.integrator_factory import IntegratorFactory

class ContinuousDynamicalSystem(SymbolicSystemBase, ContinuousSystemBase):
    """
    Concrete symbolic continuous-time system.
    
    Implements integration via IntegratorFactory for flexible solver selection.
    """
    
    # ... (initialization code from previous design) ...
    
    def integrate(
        self,
        x0,
        u=None,
        t_span=(0.0, 10.0),
        method="RK45",
        t_eval=None,
        dense_output=False,
        **integrator_kwargs
    ):
        """
        Integrate continuous system using numerical ODE solver.
        
        This method creates an appropriate integrator via IntegratorFactory
        and delegates the actual integration. Different methods can be used
        for different calls.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,)
        u : Union[ArrayLike, Callable, None]
            Control input:
            - None: Zero control (autonomous)
            - Array: Constant control u(t) = u_const
            - Callable: Time-varying u(t) or state-feedback u(t, x)
        t_span : tuple[float, float]
            Integration interval (t_start, t_end)
        method : str
            Integration method. Options:
            
            **NumPy backend**:
            - 'RK45': Explicit Runge-Kutta 4-5 (default, adaptive)
            - 'RK23': Explicit Runge-Kutta 2-3 (adaptive)
            - 'DOP853': High-order Dormand-Prince (adaptive)
            - 'LSODA': Automatic stiff/non-stiff switching (adaptive)
            - 'Radau': Implicit Runge-Kutta (stiff, adaptive)
            - 'BDF': Backward Differentiation Formula (stiff, adaptive)
            - 'Tsit5': Julia Tsitouras 5/4 (requires diffeqpy)
            - 'Vern9': Julia Verner 9/8 (high accuracy, requires diffeqpy)
            - 'rk4': Manual Runge-Kutta 4 (fixed-step, dt required)
            - 'euler': Forward Euler (fixed-step, dt required)
            
            **PyTorch backend**:
            - 'dopri5': Dormand-Prince 5 (adaptive, torchdiffeq)
            - 'dopri8': Dormand-Prince 8 (high accuracy, torchdiffeq)
            - 'rk4': Manual RK4 (fixed-step, dt required)
            
            **JAX backend**:
            - 'tsit5': Tsitouras 5/4 (adaptive, diffrax)
            - 'dopri5': Dormand-Prince 5 (adaptive, diffrax)
            - 'dopri8': Dormand-Prince 8 (high accuracy, diffrax)
            - 'rk4': Manual RK4 (fixed-step, dt required)
        
        t_eval : Optional[ArrayLike]
            Specific times at which to return solution.
            If None:
            - Adaptive methods: Uses solver's internal time points
            - Fixed-step methods: Uses regular grid with dt
        
        dense_output : bool
            If True, return dense interpolated solution (adaptive methods only)
        
        **integrator_kwargs
            Additional integrator options:
            - dt : float
                Time step (required for fixed-step methods)
            - rtol : float
                Relative tolerance (adaptive, default: 1e-6)
            - atol : float
                Absolute tolerance (adaptive, default: 1e-8)
            - max_steps : int
                Maximum number of steps (adaptive, default: 10000)
            - first_step : float
                Initial step size guess (adaptive)
            - max_step : float
                Maximum step size (adaptive)
        
        Returns
        -------
        IntegrationResult
            TypedDict containing:
            - t: Time points (T,)
            - y: State trajectory (nx, T)
            - success: Whether integration succeeded
            - message: Status message
            - nfev: Number of function evaluations
            - njev: Number of Jacobian evaluations (if applicable)
            - nsteps: Number of steps taken
            - integration_time: Computation time (seconds)
            - solver: Integrator name
            - sol: Dense output object (if dense_output=True)
        
        Examples
        --------
        Basic integration with default solver:
        
        >>> result = system.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     t_span=(0.0, 10.0)
        ... )
        >>> plt.plot(result['t'], result['y'][0, :])
        
        Stiff system with tight tolerances:
        
        >>> result = system.integrate(
        ...     x0=x0,
        ...     t_span=(0, 100),
        ...     method='Radau',
        ...     rtol=1e-9,
        ...     atol=1e-11
        ... )
        
        State feedback control:
        
        >>> def controller(t, x):
        ...     K = np.array([[-1.0, -2.0]])
        ...     return -K @ x
        >>> result = system.integrate(x0, u=controller, t_span=(0, 5))
        
        Time-varying control:
        
        >>> result = system.integrate(
        ...     x0,
        ...     u=lambda t, x: np.array([np.sin(t)]),
        ...     t_span=(0, 10)
        ... )
        
        Fixed-step RK4:
        
        >>> result = system.integrate(
        ...     x0, t_span=(0, 10),
        ...     method='rk4',
        ...     dt=0.01
        ... )
        
        High-accuracy Julia solver:
        
        >>> result = system.integrate(
        ...     x0, t_span=(0, 10),
        ...     method='Vern9',
        ...     rtol=1e-12
        ... )
        
        Notes
        -----
        The integrator is created fresh for each call, allowing different
        methods to be used for different integration tasks. The factory
        automatically selects the appropriate integrator class based on
        the backend and method.
        
        For autonomous systems (nu=0), u can be None or omitted entirely.
        
        For systems requiring gradients (PyTorch/JAX), the integration
        result can be backpropagated through if the integrator supports it.
        """
        # Convert control input to standard function form
        u_func = self._prepare_control_input(u)
        
        # Create integrator via factory
        integrator = IntegratorFactory.create(
            system=self,
            backend=self._default_backend,
            method=method,
            **integrator_kwargs
        )
        
        # Delegate to integrator
        return integrator.integrate(
            x0=x0,
            u_func=u_func,
            t_span=t_span,
            t_eval=t_eval,
            dense_output=dense_output
        )
    
    def _prepare_control_input(self, u):
        """
        Convert various control input formats to standard function.
        
        Parameters
        ----------
        u : Union[ArrayLike, Callable, None]
            Control input in various formats
        
        Returns
        -------
        Callable[[float, ArrayLike], Optional[ArrayLike]]
            Standard control function: (t, x) -> u
        """
        if u is None:
            # Autonomous system or zero control
            if self.nu == 0:
                return lambda t, x: None
            else:
                # Zero control for non-autonomous system
                return lambda t, x: self.backend.zeros(self.nu, backend=self._default_backend)
        
        elif callable(u):
            # Already a function - check signature
            import inspect
            sig = inspect.signature(u)
            n_params = len(sig.parameters)
            
            if n_params == 1:
                # u(t) - time-varying only
                return lambda t, x: u(t)
            elif n_params == 2:
                # u(t, x) - already correct form
                return u
            else:
                raise ValueError(
                    f"Control function must have signature u(t) or u(t, x), "
                    f"got {n_params} parameters"
                )
        
        else:
            # Constant control - convert array to function
            return lambda t, x: u
    
    def simulate(
        self,
        x0,
        controller=None,
        t_span=(0.0, 10.0),
        dt=0.01,
        method="RK45",
        **kwargs
    ):
        """
        High-level simulation with regular time grid.
        
        This is a convenience wrapper around integrate() that:
        - Returns results on a regular time grid (not adaptive)
        - Supports state-feedback controllers with cleaner interface
        - Hides solver diagnostics for simpler output
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state
        controller : Optional[Callable[[ArrayLike, float], ArrayLike]]
            State-feedback controller: (x, t) -> u
            If None, uses zero control
        t_span : tuple[float, float]
            Simulation interval
        dt : float
            Output time step for regular grid
        method : str
            Integration method (passed to integrate())
        **kwargs
            Additional arguments for integrate()
        
        Returns
        -------
        SimulationResult
            TypedDict containing:
            - time: Regular time points (n_steps,)
            - states: State trajectory (nx, n_steps)
            - controls: Control trajectory (nu, n_steps) if controller provided
            - metadata: Additional info (method, success, etc.)
        
        Examples
        --------
        >>> result = system.simulate(x0, t_span=(0, 5), dt=0.01)
        >>> plt.plot(result['time'], result['states'][0, :])
        
        >>> def controller(x, t):
        ...     return -K @ x
        >>> result = system.simulate(x0, controller, t_span=(0, 5))
        """
        # Create time grid
        import numpy as np
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
        
        # Convert controller to standard form
        if controller is not None:
            u_func = lambda t, x: controller(x, t)  # Swap argument order
        else:
            u_func = None
        
        # Call integrate with regular time grid
        result = self.integrate(
            x0=x0,
            u=u_func,
            t_span=t_span,
            t_eval=t_eval,
            method=method,
            **kwargs
        )
        
        # Reconstruct control trajectory if controller provided
        controls = None
        if controller is not None:
            controls = np.array([controller(result['y'][:, i], result['t'][i]) 
                                for i in range(len(result['t']))]).T
        
        return {
            "time": result['t'],
            "states": result['y'],
            "controls": controls,
            "metadata": {
                "method": method,
                "dt": dt,
                "success": result['success'],
                "nfev": result.get('nfev'),
            }
        }
```

**Key Features**:
- Creates integrator on-demand via factory
- Flexible method selection per call
- Comprehensive docstring with all options
- Helper methods for control input conversion
- Both low-level (`integrate`) and high-level (`simulate`) interfaces

---

### 3. ContinuousStochasticSystem (SDE Integration)

**Location**: `src/systems/continuous_stochastic_system.py`

```python
from src.systems.continuous_dynamical_system import ContinuousDynamicalSystem
from src.systems.base.numerical_integration.sde_integrator_factory import SDEIntegratorFactory

class ContinuousStochasticSystem(ContinuousDynamicalSystem):
    """
    Symbolic continuous-time stochastic system (SDE).
    
    Overrides integration to use SDE integrators for proper noise handling.
    """
    
    def integrate(
        self,
        x0,
        u=None,
        t_span=(0.0, 10.0),
        method="euler_maruyama",
        t_eval=None,
        n_paths=1,
        seed=None,
        **integrator_kwargs
    ):
        """
        Integrate stochastic system using SDE solver.
        
        This method overrides the parent's integrate() to use SDEIntegratorFactory
        instead of IntegratorFactory. This ensures proper handling of Brownian
        motion and noise structure.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,)
        u : Union[ArrayLike, Callable, None]
            Control input
        t_span : tuple[float, float]
            Integration interval
        method : str
            SDE integration method:
            
            **PyTorch backend (torchsde)**:
            - 'euler_maruyama': Euler-Maruyama (order 0.5)
            - 'milstein': Milstein (order 1.0, diagonal noise only)
            - 'sde_solve': Adaptive SDE solver
            
            **NumPy backend (diffeqpy)**:
            - 'EM': Euler-Maruyama
            - 'SRIW1': Stochastic Runge-Kutta (order 1.5)
            - 'RI5': Rössler (order 2.0)
            
            **JAX backend (diffrax)**:
            - 'euler_maruyama': Euler-Maruyama
            - 'heun': Heun (order 1.0)
        
        t_eval : Optional[ArrayLike]
            Specific times to return solution
        
        n_paths : int
            Number of Monte Carlo paths to simulate (default: 1)
            For n_paths > 1, performs Monte Carlo simulation
        
        seed : Optional[int]
            Random seed for reproducibility
        
        **integrator_kwargs
            Additional options:
            - dt : float (required for fixed-step methods)
            - rtol, atol : float (adaptive methods)
        
        Returns
        -------
        SDEIntegrationResult
            TypedDict containing:
            - t: Time points (T,)
            - x: State trajectories
              - Single path: (T, nx)
              - Multiple paths: (n_paths, T, nx)
            - success: Integration success
            - n_paths: Number of paths
            - noise_type: Detected noise type (additive/multiplicative)
            - sde_type: Itô or Stratonovich
            - nfev: Function evaluations
            - integration_time: Computation time
        
        Examples
        --------
        Single trajectory:
        
        >>> result = sde_system.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     t_span=(0.0, 10.0),
        ...     method='euler_maruyama',
        ...     dt=0.01
        ... )
        >>> plt.plot(result['t'], result['x'][:, 0])
        
        Monte Carlo simulation:
        
        >>> result = sde_system.integrate(
        ...     x0=x0,
        ...     t_span=(0, 10),
        ...     n_paths=1000,
        ...     seed=42
        ... )
        >>> # result['x'] has shape (1000, n_steps, nx)
        >>> mean_traj = result['x'].mean(axis=0)
        >>> std_traj = result['x'].std(axis=0)
        
        High-order method:
        
        >>> result = sde_system.integrate(
        ...     x0=x0,
        ...     t_span=(0, 10),
        ...     method='SRIW1',  # Julia, order 1.5
        ...     dt=0.01
        ... )
        """
        # Convert control input
        u_func = self._prepare_control_input(u)
        
        # Create SDE integrator via factory
        integrator = SDEIntegratorFactory.create(
            system=self,
            backend=self._default_backend,
            method=method,
            **integrator_kwargs
        )
        
        # Single path vs Monte Carlo
        if n_paths == 1:
            # Single trajectory
            return integrator.integrate(
                x0=x0,
                u_func=u_func,
                t_span=t_span,
                t_eval=t_eval,
                seed=seed
            )
        else:
            # Monte Carlo simulation
            return integrator.integrate_monte_carlo(
                x0=x0,
                u_func=u_func,
                t_span=t_span,
                n_paths=n_paths,
                t_eval=t_eval,
                seed=seed
            )
    
    @property
    def is_stochastic(self) -> bool:
        """Return True (this is a stochastic system)."""
        return True
```

**Key Features**:
- Uses SDEIntegratorFactory instead of IntegratorFactory
- Supports Monte Carlo simulation (n_paths > 1)
- Returns SDEIntegrationResult with path information
- Handles noise structure automatically

---

### 4. DiscreteSystemBase (No Integration Needed!)

**Location**: `src/systems/base/discrete_system_base.py`

**Status**: UNCHANGED - Discrete systems don't need numerical integrators!

```python
class DiscreteSystemBase(ABC):
    """
    Abstract base for discrete-time systems.
    
    No integration needed - discrete systems compute x[k+1] directly.
    """
    
    @abstractmethod
    def step(self, x, u=None, k=0):
        """Compute x[k+1] = f(x[k], u[k]) directly."""
        pass
    
    def simulate(self, x0, u_sequence=None, n_steps=100):
        """
        Multi-step simulation by repeated stepping.
        
        No integrator needed - just call step() repeatedly.
        """
        states = []
        x = x0
        for k in range(n_steps):
            states.append(x)
            u = self._get_control(u_sequence, k)
            x = self.step(x, u, k)
        return {"states": np.array(states), ...}
```

**Key Point**: Discrete systems don't use the integration framework at all! They compute next states directly.

---

### 5. DiscretizedSystem (Wrapper Uses Integrators)

**Location**: `src/systems/discretization/discretized_system.py`

```python
from src.systems.base.discrete_system_base import DiscreteSystemBase
from src.systems.base.numerical_integration.integrator_factory import IntegratorFactory

class DiscretizedSystem(DiscreteSystemBase):
    """
    Discrete approximation of continuous system via numerical integration.
    
    Uses integrators internally to discretize continuous dynamics.
    """
    
    def __init__(
        self,
        continuous_system,
        dt=0.01,
        method='rk4'
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
            - 'euler': Forward Euler
            - 'rk4': Runge-Kutta 4
            - 'zoh': Zero-order hold (exact for linear)
        """
        self._continuous_system = continuous_system
        self._dt = dt
        self._method = method
    
    @property
    def dt(self) -> float:
        return self._dt
    
    def step(self, x, u=None, k=0):
        """
        Compute x[k+1] by integrating one time step.
        
        Uses integrator internally to perform single step:
        x[k+1] = x[k] + ∫[k*dt, (k+1)*dt] f(x, u) dt
        """
        # Create integrator for single step
        integrator = IntegratorFactory.create(
            system=self._continuous_system,
            backend=self._continuous_system._default_backend,
            method=self._method,
            dt=self._dt  # Fixed step size
        )
        
        # Single step integration
        return integrator.step(x, u, dt=self._dt)
```

**Key Feature**: Even discretized systems use the integration framework internally!

---

## Factory Method Selection Logic

### IntegratorFactory Decision Tree

```
IntegratorFactory.create(system, backend, method, **kwargs)
│
├─ method specified?
│  ├─ Yes: Create that specific integrator
│  │      - 'RK45' → ScipyIntegrator(method='RK45')
│  │      - 'Tsit5' → DiffEqPyIntegrator(algorithm='Tsit5')
│  │      - 'dopri5' → TorchDiffEqIntegrator(method='dopri5')
│  │      - 'tsit5' → DiffraxIntegrator(method='tsit5')
│  │      - 'rk4' → RK4Integrator(dt=dt)
│  │
│  └─ No: Use backend default
│         - numpy → 'LSODA'
│         - torch → 'dopri5'
│         - jax → 'tsit5'
│
└─ Check availability
   ├─ Backend installed? → Create integrator
   └─ Backend missing? → Raise RuntimeError with install instructions
```

### SDEIntegratorFactory Decision Tree

```
SDEIntegratorFactory.create(system, backend, method, **kwargs)
│
├─ Analyze noise structure
│  ├─ noise_type = system.noise_type (additive/multiplicative/diagonal)
│  ├─ sde_type = system.sde_type (ito/stratonovich)
│  └─ nw = system.nw (noise dimension)
│
├─ method specified?
│  ├─ Yes: Create that SDE integrator
│  │      - 'euler_maruyama' → EulerMaruyamaIntegrator()
│  │      - 'milstein' → MilsteinIntegrator() [diagonal only]
│  │      - 'SRIW1' → DiffEqPySDEIntegrator(algorithm='SRIW1')
│  │
│  └─ No: Select based on noise structure
│         - Additive → 'euler_maruyama' (most efficient)
│         - Diagonal → 'milstein' if available
│         - General → Backend default
│
└─ Check compatibility
   ├─ Method supports noise type? → Create
   └─ Incompatible? → Raise ValueError with suggestion
```

---

## Integration Result Types (TypedDict)

### IntegrationResult (ODE)

```python
from typing_extensions import TypedDict

class IntegrationResult(TypedDict, total=False):
    """Result from ODE integration."""
    
    # Required fields
    t: ArrayLike              # Time points (T,)
    y: ArrayLike              # State trajectory (nx, T)
    success: bool             # Integration succeeded
    message: str              # Status message
    
    # Optional solver diagnostics
    nfev: int                 # Function evaluations
    njev: int                 # Jacobian evaluations
    nlu: int                  # LU decompositions
    nsteps: int               # Integration steps
    integration_time: float   # Computation time (seconds)
    solver: str               # Integrator name
    
    # Optional extras
    sol: Any                  # Dense output object
    status: int               # Termination status code
```

### SDEIntegrationResult (SDE)

```python
class SDEIntegrationResult(TypedDict, total=False):
    """Result from SDE integration."""
    
    # Required fields
    t: ArrayLike              # Time points (T,)
    x: ArrayLike              # State trajectories:
                              # - Single: (T, nx)
                              # - Multiple: (n_paths, T, nx)
    success: bool
    message: str
    
    # SDE-specific
    n_paths: int              # Number of Monte Carlo paths
    noise_type: NoiseType     # additive/multiplicative/diagonal
    sde_type: SDEType         # ito/stratonovich
    
    # Optional
    nfev: int
    integration_time: float
    solver: str
    seed: Optional[int]       # Random seed used
```

---

## Usage Examples

### Example 1: Basic ODE Integration

```python
# Define system
class Pendulum(ContinuousDynamicalSystem):
    def define_system(self, m=1.0, l=0.5, g=9.81):
        theta, theta_dot = sp.symbols('theta theta_dot')
        u = sp.symbols('u')
        
        self.state_vars = [theta, theta_dot]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([
            theta_dot,
            -(g/l)*sp.sin(theta) + u/(m*l**2)
        ])
        self.parameters = {m: m, l: l, g: g}
        self.order = 1

# Create system
system = Pendulum(m=2.0, l=1.0)

# Integrate with default solver (LSODA)
result = system.integrate(
    x0=np.array([0.1, 0.0]),
    u=None,  # Autonomous
    t_span=(0.0, 10.0)
)

print(f"Integration succeeded: {result['success']}")
print(f"Function evaluations: {result['nfev']}")
plt.plot(result['t'], result['y'][0, :])
```

### Example 2: Method Comparison

```python
# Try different methods
methods = ['RK45', 'LSODA', 'Radau', 'rk4']
results = {}

for method in methods:
    result = system.integrate(
        x0=x0,
        t_span=(0, 10),
        method=method,
        dt=0.01 if method == 'rk4' else None
    )
    results[method] = result
    print(f"{method}: {result['nfev']} function evals")

# Compare accuracy
for name, res in results.items():
    plt.plot(res['t'], res['y'][0, :], label=name)
plt.legend()
```

### Example 3: Stochastic System

```python
# Define SDE
class OrnsteinUhlenbeck(ContinuousStochasticSystem):
    def define_system(self, alpha=1.0, sigma=0.5):
        x = sp.symbols('x')
        self.state_vars = [x]
        self.control_vars = []
        self._f_sym = sp.Matrix([-alpha*x])
        self.diffusion_expr = sp.Matrix([[sigma]])
        self.sde_type = 'ito'
        self.parameters = {alpha: alpha, sigma: sigma}
        self.order = 1

# Create system
sde = OrnsteinUhlenbeck(alpha=2.0, sigma=0.3)

# Monte Carlo simulation
result = sde.integrate(
    x0=np.array([1.0]),
    t_span=(0, 10),
    method='euler_maruyama',
    dt=0.01,
    n_paths=1000,
    seed=42
)

# Analyze statistics
mean_traj = result['x'].mean(axis=0)  # (T, nx)
std_traj = result['x'].std(axis=0)
plt.plot(result['t'], mean_traj[:, 0], label='Mean')
plt.fill_between(
    result['t'],
    mean_traj[:, 0] - std_traj[:, 0],
    mean_traj[:, 0] + std_traj[:, 0],
    alpha=0.3
)
```

### Example 4: Backend Comparison

```python
# Same system, different backends
backends = ['numpy', 'torch', 'jax']

for backend in backends:
    system.set_default_backend(backend)
    
    result = system.integrate(
        x0=x0,
        t_span=(0, 10),
        method='default'  # Uses backend default
    )
    
    print(f"{backend}: {result['solver']} took {result['integration_time']:.3f}s")
```

### Example 5: Discretization via Integration

```python
# Create continuous system
continuous = Pendulum(m=1.0, l=0.5)

# Discretize using RK4
discrete = DiscretizedSystem(
    continuous,
    dt=0.01,
    method='rk4'
)

# Now it's a discrete system
x = x0
for k in range(100):
    x = discrete.step(x, u=None, k=k)
    # Internally uses integrator to take dt step
```

---

## Benefits of This Architecture

### 1. Separation of Concerns
- **Systems**: Define dynamics symbolically
- **Integrators**: Implement numerical methods
- **Factories**: Handle integrator selection and creation
- Clean interfaces, no mixing of responsibilities

### 2. Flexibility
- Different methods per call: `integrate(..., method='RK45')` vs `method='LSODA'`
- Easy to add new integrators without changing systems
- User can override factory in subclasses if needed

### 3. Maintainability
- Integrator bugs don't affect system definitions
- Can test integrators independently
- Clear dependency graph

### 4. Extensibility
- Add new backends: Just extend IntegratorBase
- Add new methods: Register in factory
- Add new integrator types: Create new factory

### 5. User Experience
- Simple default: `system.integrate(x0, t_span=(0, 10))`
- Power user: Full control over method and options
- Consistent API across all system types

### 6. Type Safety
- TypedDict results provide IDE autocomplete
- Clear contracts via abstract methods
- Type hints throughout

---

## Migration Checklist

### Phase 1: No Changes Needed!
- ✅ IntegratorBase already complete
- ✅ IntegratorFactory already complete
- ✅ SDEIntegratorBase already complete
- ✅ SDEIntegratorFactory already complete
- ✅ ContinuousSystemBase already has correct interface

### Phase 2: Implement ContinuousDynamicalSystem
1. Add `integrate()` method using IntegratorFactory
2. Add `_prepare_control_input()` helper
3. Add `simulate()` convenience method
4. Test with existing integrators

### Phase 3: Implement ContinuousStochasticSystem
1. Override `integrate()` to use SDEIntegratorFactory
2. Add Monte Carlo support (n_paths parameter)
3. Test with SDE integrators

### Phase 4: Update DiscretizedSystem
1. Use IntegratorFactory in `step()` method
2. Support multiple discretization methods
3. Test accuracy

### Phase 5: Documentation
1. Update docstrings with all available methods
2. Add examples for each backend
3. Document method selection guidelines

### Phase 6: Testing
1. Test all method/backend combinations
2. Verify TypedDict results work
3. Performance benchmarks
4. Integration tests with real systems

---

## Conclusion

The numerical integration framework integrates **perfectly** into the refactored architecture:

✅ **Clean separation**: Systems define dynamics, integrators implement methods
✅ **Factory pattern**: Flexible integrator creation on-demand
✅ **No state management**: Each integrate() call is independent
✅ **Extensible**: Easy to add new integrators and methods
✅ **Type safe**: TypedDict results throughout
✅ **User friendly**: Simple defaults, full control when needed

The key insight: **Integrators are tools that systems use, not components they own.**

No architectural changes needed - the existing integration framework is already perfectly designed for this!
