# ControlDESymulation Refactoring Plan
## Comprehensive Architecture Redesign

**Version**: 1.0  
**Date**: December 2025  
**Status**: Pre-Implementation Planning  
**Estimated Duration**: 4-6 weeks  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Architecture Problems](#current-architecture-problems)
3. [Target Architecture](#target-architecture)
4. [Detailed Component Design](#detailed-component-design)
5. [Implementation Phases](#implementation-phases)
6. [Migration Strategy](#migration-strategy)
7. [Testing Strategy](#testing-strategy)
8. [Risk Assessment](#risk-assessment)
9. [Success Criteria](#success-criteria)
10. [Timeline and Resources](#timeline-and-resources)

---

## Executive Summary

### Problem Statement

The current architecture has **~1,800 lines of duplicated code** between `SymbolicDynamicalSystem` (continuous) and `DiscreteSymbolicSystem` (discrete), creating maintenance burden and semantic confusion. The symbolic machinery is entangled with time-domain behavior, making it difficult to extend.

### Proposed Solution

Refactor to **three-layer architecture** using strategic multiple inheritance:

1. **SymbolicSystemBase**: Common symbolic machinery (NEW)
2. **ContinuousSystemBase / DiscreteSystemBase**: Time-domain interfaces (EXISTING - unchanged)
3. **Concrete classes**: Combine layers via multiple inheritance

### Key Benefits

- ✅ **Eliminate 1,800 lines of duplication**
- ✅ **Clear separation of concerns** (symbolic vs time-domain vs stochastic)
- ✅ **Zero breaking changes** (migration via aliases)
- ✅ **Easy extensibility** (data-driven systems, hybrid systems, etc.)
- ✅ **Better type safety** (abstract interfaces enforce contracts)
- ✅ **Integration framework** already compatible (no changes needed)

### Timeline

- **Week 1-2**: Create SymbolicSystemBase, ContinuousDynamicalSystem
- **Week 3**: Create DiscreteDynamicalSystem
- **Week 4**: Refactor stochastic systems
- **Week 5**: Implement DiscretizedSystem with multi-mode support
- **Week 6**: Testing, documentation, migration support

---

## Current Architecture Problems

### Problem 1: Massive Code Duplication

**Scale**: ~1,800 lines duplicated between two classes

```
SymbolicDynamicalSystem (continuous)     DiscreteSymbolicSystem (discrete)
├─ Symbolic variables                    ├─ Symbolic variables          [DUPLICATE]
├─ Parameters management                 ├─ Parameters management       [DUPLICATE]
├─ Code generation                       ├─ Code generation             [DUPLICATE]
├─ Backend management                    ├─ Backend management          [DUPLICATE]
├─ Equilibrium handling                  ├─ Equilibrium handling        [DUPLICATE]
├─ Configuration persistence             ├─ Configuration persistence   [DUPLICATE]
└─ ~2000 lines total                     └─ ~450 lines total
```

**Impact**:
- Bug fixes must be applied twice
- Features added once miss the other
- Documentation diverges
- Maintenance burden increases over time

### Problem 2: Semantic Confusion

**Current inheritance**:
```python
class DiscreteSymbolicSystem(SymbolicDynamicalSystem):  # ❌ Problematic
    """Inherits from continuous system!"""
```

**Issues**:
- Discrete system inherits continuous-time semantics
- `_f_sym` means different things (dx/dt vs x[k+1])
- Linearization returns different matrix types (A vs Ad)
- Documentation must carefully explain overriding

### Problem 3: Difficult Extension

Want to add new system types:
- ✅ Data-driven systems (neural ODEs, learned dynamics)
- ✅ Hybrid systems (switched dynamics, event-driven)
- ✅ Multi-rate systems (different sampling rates)

**Current architecture makes this hard**:
- Must choose: inherit from continuous or discrete?
- Can't easily mix symbolic with data-driven
- Symbolic machinery is coupled to time domain

### Problem 4: Discretization Complexity

Converting continuous → discrete requires:
- Fixed-step integration (simple but lower accuracy)
- Adaptive integration (accurate but irregular time grid)
- Need interpolation for adaptive methods

**Currently**: No systematic solution

---

## Target Architecture

### Three-Layer Hierarchy

```
                    ┌─────────────────────────┐
                    │   SymbolicSystemBase    │
                    │   (NEW - Abstract)      │
                    │                         │
                    │ - Symbolic variables    │
                    │ - Parameters            │
                    │ - Code generation       │
                    │ - Backend management    │
                    │ - Equilibria            │
                    │ - Configuration         │
                    │                         │
                    │ ~1800 lines extracted   │
                    └────────────┬────────────┘
                                 │
                  ┌──────────────┴───────────────┐
                  │                              │
      ┌───────────▼──────────────┐   ┌──────────▼───────────────┐
      │ ContinuousSystemBase     │   │ DiscreteSystemBase       │
      │ (EXISTING - unchanged)   │   │ (EXISTING - unchanged)   │
      │                          │   │                          │
      │ - __call__(x,u,t)→dx/dt  │   │ - step(x,u,k)→x[k+1]     │
      │ - integrate()            │   │ - simulate()             │
      │ - linearize()→(A,B)      │   │ - linearize()→(Ad,Bd)    │
      │ - simulate()             │   │ - rollout()              │
      │                          │   │                          │
      │ Interface only           │   │ Interface only           │
      └───────────┬──────────────┘   └──────────┬───────────────┘
                  │                              │
                  │   Multiple Inheritance       │
                  │                              │
      ┌───────────▼──────────────┐   ┌──────────▼───────────────┐
      │ ContinuousDynamical      │   │ DiscreteDynamical        │
      │ System                   │   │ System                   │
      │ (NEW - replaces          │   │ (NEW - replaces          │
      │  SymbolicDynamicalSystem)│   │  DiscreteSymbolicSystem) │
      │                          │   │                          │
      │ Inherits:                │   │ Inherits:                │
      │ - SymbolicSystemBase     │   │ - SymbolicSystemBase     │
      │ - ContinuousSystemBase   │   │ - DiscreteSystemBase     │
      │                          │   │                          │
      │ ~300 lines               │   │ ~300 lines               │
      └───────────┬──────────────┘   └──────────┬───────────────┘
                  │                              │
                  │                              │
      ┌───────────▼──────────────┐   ┌──────────▼───────────────┐
      │ ContinuousStochastic     │   │ DiscreteStochastic       │
      │ System                   │   │ System                   │
      │ (Renamed from            │   │ (Updated parent)         │
      │  StochasticDynamical     │   │                          │
      │  System)                 │   │                          │
      │                          │   │                          │
      │ Adds: Diffusion          │   │ Adds: Diffusion          │
      │ Uses: SDEIntegrator      │   │ Uses: SDEIntegrator      │
      └──────────────────────────┘   └──────────────────────────┘
```

### Additional Components

```
┌──────────────────────────────────────────────────────────┐
│ DiscretizedSystem                                        │
│ (NEW - Wrapper for continuous → discrete)                │
│                                                          │
│ Wraps: ContinuousSystemBase (any implementation)         │
│ Modes:                                                   │
│ - FIXED_STEP: RK4, Euler (natural for step())           │
│ - DENSE_OUTPUT: Adaptive + dense interpolation          │
│ - BATCH_INTERPOLATION: Adaptive + grid interpolation    │
│                                                          │
│ ~400 lines                                               │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ Integration Framework                                    │
│ (EXISTING - No changes needed!)                          │
│                                                          │
│ - IntegratorBase                                         │
│ - IntegratorFactory                                      │
│ - SDEIntegratorBase                                      │
│ - SDEIntegratorFactory                                   │
│                                                          │
│ Used by: ContinuousDynamicalSystem.integrate()          │
│ Used by: ContinuousStochasticSystem.integrate()         │
│ Used by: DiscretizedSystem (internally)                 │
└──────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Single Responsibility**: Each class has one clear purpose
2. **Interface Segregation**: Abstract bases define contracts only
3. **Composition**: Delegate to specialized handlers (BackendManager, CodeGenerator, etc.)
4. **Factory Pattern**: Integrators created on-demand, not stored
5. **Multiple Inheritance**: Justified - combines orthogonal concerns

---

## Detailed Component Design

### Component 1: SymbolicSystemBase (NEW)

**Purpose**: Provide symbolic machinery for ANY symbolic system

**Location**: `src/systems/base/symbolic_system_base.py`

**Responsibilities**:
- Symbolic variable management (state_vars, control_vars, output_vars)
- Parameter handling and substitution
- Code generation via CodeGenerator
- Backend management (NumPy/PyTorch/JAX)
- Equilibrium point management
- Configuration persistence (save/load)
- Performance tracking

**What it provides**: ~1800 lines of time-domain-agnostic code

**What it does NOT provide**:
- Time-domain semantics (continuous vs discrete)
- System interface (__call__, integrate, step, etc.)
- Linearization interpretation

**Key Methods**:
```python
class SymbolicSystemBase(ABC):
    # Template method pattern
    @abstractmethod
    def define_system(self, *args, **kwargs):
        """User implements to define symbolic system"""
    
    # Symbolic utilities
    def substitute_parameters(self, expr) -> sp.Expr:
        """Substitute numerical parameter values"""
    
    @abstractmethod
    def print_equations(self, simplify=True):
        """Print equations (notation depends on subclass)"""
    
    # Code generation
    def compile(self, backends=None, verbose=False):
        """Pre-compile dynamics functions"""
    
    def reset_caches(self, backends=None):
        """Clear cached compiled functions"""
    
    # Configuration
    def get_config_dict(self) -> Dict:
        """Get system configuration"""
    
    def save_config(self, filename):
        """Save configuration to file"""
    
    # Properties
    @property
    def nx(self) -> int: ...
    @property
    def nu(self) -> int: ...
    @property
    def ny(self) -> int: ...
    @property
    def nq(self) -> int: ...
```

**Composed Components** (delegation pattern):
```python
self.backend = BackendManager()
self._validator = SymbolicValidator()
self.equilibria = EquilibriumHandler()
self._code_gen = CodeGenerator()
```

**Status**: To be created (extract from SymbolicDynamicalSystem)

---

### Component 2: ContinuousSystemBase (EXISTING)

**Purpose**: Define interface for continuous-time systems

**Location**: `src/systems/base/continuous_system_base.py`

**Status**: ✅ KEEP UNCHANGED - Already perfect!

**Interface**:
```python
class ContinuousSystemBase(ABC):
    @abstractmethod
    def __call__(self, x, u=None, t=0.0) -> StateVector:
        """Evaluate dx/dt = f(x, u, t)"""
    
    @abstractmethod
    def integrate(
        self, x0, u, t_span, method='RK45', **kwargs
    ) -> IntegrationResult:
        """Numerical integration"""
    
    @abstractmethod
    def linearize(self, x_eq, u_eq) -> LinearizationResult:
        """Continuous linearization: (A, B)"""
    
    def simulate(self, x0, controller, t_span, dt) -> SimulationResult:
        """High-level simulation with regular grid"""
    
    @property
    def is_continuous(self) -> bool:
        return True
    
    @property
    def is_discrete(self) -> bool:
        return False
```

**Changes**: None needed

---

### Component 3: DiscreteSystemBase (EXISTING)

**Purpose**: Define interface for discrete-time systems

**Location**: `src/systems/base/discrete_system_base.py`

**Status**: ✅ KEEP UNCHANGED - Already perfect!

**Interface**:
```python
class DiscreteSystemBase(ABC):
    @property
    @abstractmethod
    def dt(self) -> float:
        """Sampling period"""
    
    @abstractmethod
    def step(self, x, u=None, k=0) -> StateVector:
        """Compute x[k+1] = f(x[k], u[k])"""
    
    @abstractmethod
    def simulate(
        self, x0, u_sequence, n_steps
    ) -> DiscreteSimulationResult:
        """Multi-step simulation"""
    
    @abstractmethod
    def linearize(self, x_eq, u_eq) -> DiscreteLinearization:
        """Discrete linearization: (Ad, Bd)"""
    
    def rollout(self, x0, policy, n_steps) -> DiscreteSimulationResult:
        """Closed-loop simulation"""
    
    @property
    def is_continuous(self) -> bool:
        return False
    
    @property
    def is_discrete(self) -> bool:
        return True
```

**Changes**: None needed

---

### Component 4: ContinuousDynamicalSystem (NEW)

**Purpose**: Concrete symbolic continuous-time system

**Location**: `src/systems/continuous_dynamical_system.py`

**Status**: To be created (replaces SymbolicDynamicalSystem)

**Inheritance**: `(SymbolicSystemBase, ContinuousSystemBase)`

**Size**: ~300 lines (vs 2000+ currently)

**Key Implementation**:
```python
class ContinuousDynamicalSystem(SymbolicSystemBase, ContinuousSystemBase):
    """
    Symbolic continuous-time system: dx/dt = f(x, u)
    
    Combines symbolic machinery with continuous interface.
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize symbolic machinery first
        SymbolicSystemBase.__init__(self, *args, **kwargs)
        
        # Initialize continuous-specific components
        self._dynamics = DynamicsEvaluator(self, self._code_gen, self.backend)
        self._linearization = LinearizationEngine(self, self._code_gen, self.backend)
        self._observation = ObservationEngine(self, self._code_gen, self.backend)
    
    # Implement ContinuousSystemBase interface
    def __call__(self, x, u=None, t=0.0):
        """Evaluate dx/dt = f(x, u, t)"""
        return self._dynamics.evaluate(x, u, backend=None)
    
    def integrate(self, x0, u, t_span, method='RK45', **kwargs):
        """Integration via IntegratorFactory"""
        u_func = self._prepare_control_input(u)
        integrator = IntegratorFactory.create(
            self, backend=self._default_backend,
            method=method, **kwargs
        )
        return integrator.integrate(x0, u_func, t_span)
    
    def linearize(self, x_eq, u_eq=None):
        """Continuous linearization: (A, B)"""
        return self._linearization.compute(x_eq, u_eq, backend=None)
    
    def simulate(self, x0, controller, t_span, dt, method='RK45', **kwargs):
        """High-level simulation with regular grid"""
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
        u_func = lambda t, x: controller(x, t) if controller else None
        result = self.integrate(x0, u_func, t_span, t_eval=t_eval, method=method)
        return self._format_simulation_result(result, controller)
    
    def print_equations(self, simplify=True):
        """Print using continuous notation (dx/dt)"""
        print("Dynamics: dx/dt = f(x, u)")
        for var, expr in zip(self.state_vars, self._f_sym):
            expr_sub = self.substitute_parameters(expr)
            if simplify:
                expr_sub = sp.simplify(expr_sub)
            print(f"  d{var}/dt = {expr_sub}")
    
    def forward(self, x, u=None, backend=None):
        """Alias for __call__ with backend specification"""
        return self._dynamics.evaluate(x, u, backend)
    
    def _prepare_control_input(self, u):
        """Convert various control formats to standard function"""
        # ... implementation ...
```

---

### Component 5: DiscreteDynamicalSystem (NEW)

**Purpose**: Concrete symbolic discrete-time system

**Location**: `src/systems/discrete_dynamical_system.py`

**Status**: To be created (replaces DiscreteSymbolicSystem)

**Inheritance**: `(SymbolicSystemBase, DiscreteSystemBase)`

**Size**: ~300 lines (vs 450+ currently)

**Key Implementation**:
```python
class DiscreteDynamicalSystem(SymbolicSystemBase, DiscreteSystemBase):
    """
    Symbolic discrete-time system: x[k+1] = f(x[k], u[k])
    
    Combines symbolic machinery with discrete interface.
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize symbolic machinery
        SymbolicSystemBase.__init__(self, *args, **kwargs)
        
        # Initialize discrete-specific components
        self._dynamics = DynamicsEvaluator(self, self._code_gen, self.backend)
        self._linearization = LinearizationEngine(self, self._code_gen, self.backend)
        
        # Discrete systems must define dt
        if not hasattr(self, '_dt'):
            raise ValueError("Must define self._dt in define_system()")
    
    # Implement DiscreteSystemBase interface
    @property
    def dt(self) -> float:
        return self._dt
    
    def step(self, x, u=None, k=0):
        """Compute x[k+1] = f(x[k], u[k])"""
        return self._dynamics.evaluate(x, u, backend=None)
    
    def simulate(self, x0, u_sequence, n_steps, **kwargs):
        """Multi-step simulation"""
        states = np.zeros((self.nx, n_steps + 1))
        states[:, 0] = x0
        
        u_func = self._prepare_control_sequence(u_sequence, n_steps)
        
        x = x0
        controls = []
        for k in range(n_steps):
            u = u_func(k)
            controls.append(u)
            x = self.step(x, u, k)
            states[:, k + 1] = x
        
        return {
            "states": states,
            "controls": np.array(controls).T if controls else None,
            "time_steps": np.arange(n_steps + 1),
            "dt": self.dt,
            "metadata": {"method": "discrete_step", "success": True}
        }
    
    def linearize(self, x_eq, u_eq=None):
        """Discrete linearization: (Ad, Bd)"""
        return self._linearization.compute(x_eq, u_eq, backend=None)
    
    def print_equations(self, simplify=True):
        """Print using discrete notation (x[k+1])"""
        print(f"Dynamics: x[k+1] = f(x[k], u[k]) (dt={self.dt})")
        for var, expr in zip(self.state_vars, self._f_sym):
            expr_sub = self.substitute_parameters(expr)
            if simplify:
                expr_sub = sp.simplify(expr_sub)
            print(f"  {var}[k+1] = {expr_sub}")
```

---

### Component 6: ContinuousStochasticSystem (REFACTORED)

**Purpose**: Extend continuous systems with stochastic dynamics

**Location**: `src/systems/continuous_stochastic_system.py`

**Status**: Rename and update parent

**Changes**:
```python
# OLD:
class StochasticDynamicalSystem(SymbolicDynamicalSystem):
    pass

# NEW:
class ContinuousStochasticSystem(ContinuousDynamicalSystem):
    """SDE: dx = f(x,u)dt + g(x,u)dW"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add stochastic-specific components
        self._diffusion = DiffusionHandler(self, self._code_gen)
        self._noise_analyzer = NoiseCharacterizer(self)
        self.noise_type = self._noise_analyzer.classify_noise()
    
    def integrate(self, x0, u, t_span, method='euler_maruyama', 
                  n_paths=1, seed=None, **kwargs):
        """SDE integration using SDEIntegratorFactory"""
        u_func = self._prepare_control_input(u)
        
        integrator = SDEIntegratorFactory.create(
            self, backend=self._default_backend,
            method=method, **kwargs
        )
        
        if n_paths == 1:
            return integrator.integrate(x0, u_func, t_span, seed=seed)
        else:
            return integrator.integrate_monte_carlo(
                x0, u_func, t_span, n_paths, seed=seed
            )
    
    def linearize(self, x_eq, u_eq=None):
        """Returns (A, B, G) with diffusion matrix"""
        A, B = super().linearize(x_eq, u_eq)
        G = self._diffusion.evaluate(x_eq, u_eq)
        return (A, B, G)
    
    @property
    def is_stochastic(self) -> bool:
        return True
```

**Migration Alias**:
```python
# For backward compatibility
StochasticDynamicalSystem = ContinuousStochasticSystem
```

---

### Component 7: DiscreteStochasticSystem (UPDATED)

**Purpose**: Extend discrete systems with stochastic dynamics

**Location**: `src/systems/discrete_stochastic_system.py`

**Status**: Update parent class

**Changes**:
```python
# OLD:
class DiscreteStochasticSystem(StochasticDynamicalSystem):
    pass

# NEW:
class DiscreteStochasticSystem(DiscreteDynamicalSystem):
    """Stochastic difference: x[k+1] = f(x,u) + g(x,u)w[k]"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._diffusion = DiffusionHandler(self, self._code_gen)
    
    def linearize(self, x_eq, u_eq=None):
        """Returns (Ad, Bd, Gd) with diffusion matrix"""
        Ad, Bd = super().linearize(x_eq, u_eq)
        Gd = self._diffusion.evaluate(x_eq, u_eq)
        return (Ad, Bd, Gd)
    
    @property
    def is_stochastic(self) -> bool:
        return True
```

---

### Component 8: DiscretizedSystem (NEW)

**Purpose**: Discrete approximation of continuous systems

**Location**: `src/systems/discretization/discretized_system.py`

**Status**: To be created

**Size**: ~400 lines

**Three Modes**:

1. **FIXED_STEP** (default): Uses fixed-step integrators
   - Natural for `step()`
   - Most efficient
   - Lower accuracy

2. **DENSE_OUTPUT**: Uses adaptive integrators with dense interpolation
   - Supports `step()` via dense output
   - High accuracy
   - Slightly slower

3. **BATCH_INTERPOLATION**: Integrates full trajectory then interpolates
   - Cannot use `step()` (only `simulate()`)
   - Most efficient for full trajectories
   - Highest accuracy

**Key Implementation**:
```python
class DiscretizationMode(Enum):
    FIXED_STEP = "fixed_step"
    DENSE_OUTPUT = "dense_output"
    BATCH_INTERPOLATION = "batch_interpolation"

class DiscretizedSystem(DiscreteSystemBase):
    def __init__(
        self,
        continuous_system,
        dt=0.01,
        method='rk4',
        mode=None,  # Auto-selected if None
        interpolation_kind='cubic',
        **integrator_kwargs
    ):
        self._continuous_system = continuous_system
        self._dt = dt
        self._method = method
        self._is_fixed_step = method.lower() in ['euler', 'rk4', 'midpoint']
        
        # Auto-select mode
        if mode is None:
            self._mode = (DiscretizationMode.FIXED_STEP if self._is_fixed_step 
                         else DiscretizationMode.DENSE_OUTPUT)
        else:
            self._mode = mode
    
    def step(self, x, u, k):
        """Single step (mode-dependent)"""
        if self._mode == DiscretizationMode.BATCH_INTERPOLATION:
            raise NotImplementedError("Use simulate() for BATCH mode")
        
        t_start = k * self.dt
        t_end = (k + 1) * self.dt
        
        if self._mode == DiscretizationMode.FIXED_STEP:
            return self._step_fixed(x, u, t_start, t_end)
        else:
            return self._step_dense(x, u, t_start, t_end)
    
    def simulate(self, x0, u_sequence, n_steps):
        """Multi-step simulation (mode-dependent)"""
        if self._mode == DiscretizationMode.BATCH_INTERPOLATION:
            return self._simulate_batch(x0, u_sequence, n_steps)
        else:
            return self._simulate_step_by_step(x0, u_sequence, n_steps)
```

---

### Component 9: Integration Framework (UNCHANGED)

**Status**: ✅ Already perfect - no changes needed!

**Components**:
- `IntegratorBase` - Abstract ODE integrator interface
- `IntegratorFactory` - Creates appropriate integrators
- `SDEIntegratorBase` - Abstract SDE integrator interface
- `SDEIntegratorFactory` - Creates SDE integrators

**Usage**:
```python
# In ContinuousDynamicalSystem.integrate():
integrator = IntegratorFactory.create(
    system=self,
    backend=self._default_backend,
    method=method,
    **kwargs
)
return integrator.integrate(x0, u_func, t_span)

# In ContinuousStochasticSystem.integrate():
integrator = SDEIntegratorFactory.create(
    system=self,
    backend=self._default_backend,
    method=method,
    **kwargs
)
return integrator.integrate(x0, u_func, t_span)
```

**Why no changes needed**: Already uses factory pattern, already compatible with system interface.

---

## Implementation Phases

### Phase 1: Foundation (Week 1)

**Goal**: Create SymbolicSystemBase and establish pattern

**Tasks**:
1. Create `src/systems/base/symbolic_system_base.py`
   - Extract ~1800 lines from SymbolicDynamicalSystem
   - Keep all symbolic machinery
   - Make forward(), linearize(), print_equations() abstract
   - Add comprehensive docstrings

2. Add type hints throughout
   - Import from src.types.core
   - Use TypedDict for results
   - Add proper return type annotations

3. Create test file `tests/test_symbolic_system_base.py`
   - Cannot test directly (abstract)
   - Create mock concrete subclass for testing
   - Test all non-abstract methods

**Deliverables**:
- ✅ symbolic_system_base.py (~1800 lines)
- ✅ Unit tests (mock implementation)
- ✅ Documentation

**Success Criteria**:
- All extracted methods work identically
- Mock subclass passes all tests
- No regressions in existing tests

---

### Phase 2: Continuous Systems (Week 1-2)

**Goal**: Create ContinuousDynamicalSystem with integration support

**Tasks**:
1. Create `src/systems/continuous_dynamical_system.py`
   - Inherit from (SymbolicSystemBase, ContinuousSystemBase)
   - Implement all abstract methods
   - Add integrate() using IntegratorFactory
   - Add simulate() convenience method
   - Size target: ~300 lines

2. Add migration alias
   ```python
   # In src/systems/__init__.py
   from src.systems.continuous_dynamical_system import ContinuousDynamicalSystem
   SymbolicDynamicalSystem = ContinuousDynamicalSystem  # Alias
   ```

3. Test thoroughly
   - All existing SymbolicDynamicalSystem tests must pass
   - Test all integration methods (RK45, LSODA, etc.)
   - Test all backends (numpy, torch, jax)
   - Performance benchmarks

**Deliverables**:
- ✅ continuous_dynamical_system.py (~300 lines)
- ✅ Migration alias
- ✅ All existing tests pass
- ✅ New integration tests

**Success Criteria**:
- Zero breaking changes
- All test suites pass
- Performance within 5% of original

---

### Phase 3: Discrete Systems (Week 3)

**Goal**: Create DiscreteDynamicalSystem

**Tasks**:
1. Create `src/systems/discrete_dynamical_system.py`
   - Inherit from (SymbolicSystemBase, DiscreteSystemBase)
   - Implement all abstract methods
   - Add simulate() with proper control handling
   - Size target: ~300 lines

2. Add migration alias
   ```python
   DiscreteSymbolicSystem = DiscreteDynamicalSystem  # Alias
   ```

3. Test thoroughly
   - All existing DiscreteSymbolicSystem tests must pass
   - Test step() method extensively
   - Test simulate() with various control inputs
   - Test linearization (discrete matrices)

**Deliverables**:
- ✅ discrete_dynamical_system.py (~300 lines)
- ✅ Migration alias
- ✅ All existing tests pass

**Success Criteria**:
- Zero breaking changes
- All discrete system tests pass
- Correct dt handling

---

### Phase 4: Stochastic Systems (Week 4)

**Goal**: Refactor stochastic systems with new parents

**Tasks**:
1. Rename StochasticDynamicalSystem → ContinuousStochasticSystem
   - Change parent to ContinuousDynamicalSystem
   - Verify diffusion machinery works
   - Update integrate() to use SDEIntegratorFactory
   - Add Monte Carlo support (n_paths parameter)

2. Update DiscreteStochasticSystem
   - Change parent to DiscreteDynamicalSystem
   - Verify discrete stochastic works
   - Test noise handling

3. Add migration aliases
   ```python
   StochasticDynamicalSystem = ContinuousStochasticSystem
   ```

4. Test extensively
   - All SDE integration methods
   - Monte Carlo simulations
   - Noise structure detection
   - Both Ito and Stratonovich

**Deliverables**:
- ✅ continuous_stochastic_system.py (renamed)
- ✅ discrete_stochastic_system.py (updated)
- ✅ Migration aliases
- ✅ All stochastic tests pass

**Success Criteria**:
- All SDE tests pass
- Monte Carlo works correctly
- Noise analysis unchanged

---

### Phase 5: Discretization (Week 5)

**Goal**: Create multi-mode DiscretizedSystem

**Tasks**:
1. Create `src/systems/discretization/discretized_system.py`
   - Implement three modes (FIXED_STEP, DENSE_OUTPUT, BATCH_INTERPOLATION)
   - Add automatic mode selection
   - Implement interpolation for BATCH mode
   - Add compare_modes() utility
   - Size target: ~400 lines

2. Add comprehensive testing
   - Test each mode independently
   - Compare accuracy vs reference solution
   - Performance benchmarks
   - Edge cases (stiff systems, long trajectories)

3. Add examples and documentation
   - When to use each mode
   - Performance guidelines
   - Accuracy comparison

**Deliverables**:
- ✅ discretized_system.py (~400 lines)
- ✅ Comprehensive tests
- ✅ Usage examples
- ✅ Performance analysis

**Success Criteria**:
- All three modes work correctly
- BATCH mode is 5-10x faster for open-loop
- Interpolation accuracy is within tolerances
- Clear documentation

---

### Phase 6: Testing and Documentation (Week 6)

**Goal**: Comprehensive testing, documentation, migration guide

**Tasks**:

1. **Integration Testing**
   - End-to-end system creation
   - All method/backend combinations
   - Stochastic + deterministic
   - Continuous → discrete discretization
   - Performance benchmarks

2. **Documentation Updates**
   - Update architecture docs
   - Update API reference
   - Create migration guide
   - Update all examples
   - Add tutorials for new features

3. **Migration Support**
   - Create deprecation warnings (optional)
   - Add migration script (rename imports)
   - FAQ for common issues
   - Version compatibility matrix

4. **Performance Validation**
   - Benchmark against old implementation
   - Profile memory usage
   - Identify any regressions
   - Optimize bottlenecks

**Deliverables**:
- ✅ Full integration test suite
- ✅ Updated documentation
- ✅ Migration guide
- ✅ Performance report

**Success Criteria**:
- 100% test coverage on new code
- All examples updated and working
- Performance within 5% of original
- Clear migration path

---

## Migration Strategy

### For Library Maintainers (You)

#### Step 1: Backward Compatibility (Immediate)

Add aliases in `src/systems/__init__.py`:
```python
# New names (preferred)
from src.systems.continuous_dynamical_system import ContinuousDynamicalSystem
from src.systems.discrete_dynamical_system import DiscreteDynamicalSystem
from src.systems.continuous_stochastic_system import ContinuousStochasticSystem

# Old names (deprecated but still work)
SymbolicDynamicalSystem = ContinuousDynamicalSystem
DiscreteSymbolicSystem = DiscreteDynamicalSystem
StochasticDynamicalSystem = ContinuousStochasticSystem
```

Result: **Zero breaking changes** - all old code works!

#### Step 2: Internal Updates (Gradual)

Update internal code to use new names:
```python
# OLD:
from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem
class MySystem(SymbolicDynamicalSystem):
    pass

# NEW:
from src.systems import ContinuousDynamicalSystem
class MySystem(ContinuousDynamicalSystem):
    pass
```

#### Step 3: Documentation Updates

- Mark old names as deprecated in docstrings
- Update all examples to use new names
- Add migration guide to docs

#### Step 4: Deprecation Warnings (Optional, after 2-3 releases)

```python
import warnings

def _deprecated_alias(old_name, new_name, new_class):
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{old_name} is deprecated, use {new_name} instead",
            DeprecationWarning,
            stacklevel=2
        )
        return new_class(*args, **kwargs)
    return wrapper

SymbolicDynamicalSystem = _deprecated_alias(
    "SymbolicDynamicalSystem",
    "ContinuousDynamicalSystem",
    ContinuousDynamicalSystem
)
```

#### Step 5: Full Removal (Optional, after 6+ months)

Remove old files and aliases (only if you want clean break).

### For Future Users (Zero Changes Needed)

If users have no existing code:
```python
# They use new names from day 1
from src.systems import ContinuousDynamicalSystem

class Pendulum(ContinuousDynamicalSystem):
    def define_system(self, m=1.0, l=0.5):
        # ... same as before
```

### For Existing Users (Minimal Changes)

#### Option A: No Changes (Recommended)

Old code continues to work via aliases:
```python
# Existing code - no changes needed!
from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem

class MySystem(SymbolicDynamicalSystem):  # Still works!
    pass
```

#### Option B: Update Names (Preferred)

Simple find-and-replace:
```bash
# In all Python files:
SymbolicDynamicalSystem → ContinuousDynamicalSystem
DiscreteSymbolicSystem → DiscreteDynamicalSystem
StochasticDynamicalSystem → ContinuousStochasticSystem
```

Update imports:
```python
# OLD:
from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem

# NEW:
from src.systems import ContinuousDynamicalSystem
```

That's it! Everything else stays the same.

---

## Testing Strategy

### Test Categories

#### 1. Unit Tests (Per Component)

**SymbolicSystemBase**:
- Symbolic variable management
- Parameter substitution
- Code generation
- Backend management
- Configuration persistence
- Mock subclass for testing abstract class

**ContinuousDynamicalSystem**:
- All integration methods
- All backends (numpy, torch, jax)
- Control input conversion
- Linearization
- Simulation vs integrate consistency

**DiscreteDynamicalSystem**:
- Step method
- Simulate with various control inputs
- Linearization (discrete matrices)
- Time index handling

**ContinuousStochasticSystem**:
- SDE integration methods
- Monte Carlo simulations
- Noise structure detection
- Diffusion evaluation

**DiscreteStochasticSystem**:
- Discrete stochastic stepping
- Noise handling

**DiscretizedSystem**:
- Each mode independently
- Mode switching
- Interpolation accuracy
- Performance vs accuracy tradeoffs

#### 2. Integration Tests

**End-to-End Workflows**:
```python
def test_pendulum_workflow():
    # Define system
    system = Pendulum(m=1.0, l=0.5)
    
    # Integrate
    result = system.integrate(x0, None, (0, 10))
    assert result['success']
    
    # Linearize
    A, B = system.linearize(np.zeros(2), np.zeros(1))
    
    # Simulate with controller
    result = system.simulate(x0, controller, (0, 10), dt=0.01)
    
    # Discretize
    discrete = DiscretizedSystem(system, dt=0.01)
    d_result = discrete.simulate(x0, None, 1000)
    
    # Verify consistency
    np.testing.assert_allclose(
        result['states'][:, ::100],
        d_result['states'],
        rtol=1e-3
    )
```

**Cross-Backend Consistency**:
```python
def test_backend_consistency():
    """Verify all backends give same results"""
    system = MySystem()
    x0 = np.array([1.0, 0.0])
    
    results = {}
    for backend in ['numpy', 'torch', 'jax']:
        system.set_default_backend(backend)
        result = system.integrate(x0, None, (0, 10))
        results[backend] = result['y'][:, -1]
    
    # All backends agree within tolerance
    for b1, b2 in itertools.combinations(results.keys(), 2):
        np.testing.assert_allclose(
            results[b1], results[b2], rtol=1e-10
        )
```

#### 3. Regression Tests

**Ensure No Breaking Changes**:
- All existing test files must pass without modification
- Load old pickled systems (if any)
- Verify numerical results unchanged
- Performance within 5% threshold

#### 4. Performance Tests

**Benchmarks**:
```python
def benchmark_integration():
    system = BenchmarkSystem()
    x0 = np.random.randn(10)
    
    times = []
    for method in ['RK45', 'LSODA', 'rk4']:
        start = time.time()
        for _ in range(100):
            system.integrate(x0, None, (0, 10), method=method)
        elapsed = time.time() - start
        times.append((method, elapsed))
    
    return times
```

**Memory Profiling**:
```python
@memory_profiler.profile
def test_memory_usage():
    system = LargeSystem(nx=1000)
    result = system.integrate(x0, None, (0, 100))
    # Track memory growth
```

#### 5. Property-Based Tests (Hypothesis)

**Invariants**:
```python
from hypothesis import given, strategies as st

@given(
    x=st.lists(st.floats(min_value=-10, max_value=10), min_size=2, max_size=2),
    u=st.lists(st.floats(min_value=-5, max_value=5), min_size=1, max_size=1)
)
def test_linearization_matches_jacobian(x, u):
    """Linearization should match numerical Jacobian"""
    x, u = np.array(x), np.array(u)
    
    A, B = system.linearize(x, u)
    
    # Numerical Jacobian via finite differences
    A_num = numerical_jacobian(lambda x_: system(x_, u), x)
    B_num = numerical_jacobian(lambda u_: system(x, u_), u)
    
    np.testing.assert_allclose(A, A_num, rtol=1e-5)
    np.testing.assert_allclose(B, B_num, rtol=1e-5)
```

### Test Coverage Goals

- **Overall**: 95%+ coverage
- **New code**: 100% coverage
- **Critical paths**: 100% coverage (integration, linearization, code generation)
- **Edge cases**: Comprehensive (empty control, zero dimensions, etc.)

### Continuous Integration

**GitHub Actions Workflow**:
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## Risk Assessment

### Risk 1: Breaking Existing Code

**Probability**: Low  
**Impact**: High  
**Mitigation**:
- ✅ Use migration aliases (SymbolicDynamicalSystem = ContinuousDynamicalSystem)
- ✅ Run ALL existing tests
- ✅ Provide clear migration guide
- ✅ Gradual rollout with deprecation warnings

**Contingency**: Keep old files as fallback for 1-2 releases

### Risk 2: Performance Regression

**Probability**: Low  
**Impact**: Medium  
**Mitigation**:
- ✅ Benchmark before/after
- ✅ Profile critical paths
- ✅ Abstract methods have zero overhead (inlined)
- ✅ Same code generation, just reorganized

**Target**: < 5% performance difference

### Risk 3: Stochastic Systems Break

**Probability**: Medium  
**Impact**: High  
**Mitigation**:
- ✅ Extensive SDE testing
- ✅ Verify diffusion machinery unchanged
- ✅ Test Monte Carlo simulations
- ✅ Check both Ito and Stratonovich

**Contingency**: Stochastic systems can keep old inheritance temporarily

### Risk 4: Integration Framework Incompatibility

**Probability**: Very Low  
**Impact**: High  
**Mitigation**:
- ✅ Integration framework already uses factory pattern
- ✅ Integrators already take system as parameter
- ✅ No changes needed to integration framework
- ✅ Test all integrator/system combinations

**Assessment**: Integration framework is already perfectly designed for this

### Risk 5: User Confusion

**Probability**: Medium  
**Impact**: Low  
**Mitigation**:
- ✅ Clear documentation
- ✅ Migration guide with examples
- ✅ Deprecation warnings with suggestions
- ✅ FAQ section

**Contingency**: Support period with both naming schemes

### Risk 6: Multiple Inheritance Issues

**Probability**: Low  
**Impact**: Medium  
**Mitigation**:
- ✅ Careful MRO (Method Resolution Order) design
- ✅ No diamond problem (bases are independent ABCs)
- ✅ Explicit __init__ ordering
- ✅ Test inheritance chain

**Validation**: Python's MRO handles this cleanly when designed properly

### Risk 7: Discretization Mode Confusion

**Probability**: Medium  
**Impact**: Low  
**Mitigation**:
- ✅ Automatic mode selection based on method
- ✅ Clear error messages for incompatible combinations
- ✅ Comprehensive documentation
- ✅ Compare_modes() utility for users

**Contingency**: Default to simplest mode (FIXED_STEP)

---

## Success Criteria

### Must Have (Hard Requirements)

1. ✅ **Zero Breaking Changes**
   - All existing tests pass without modification
   - Old class names work via aliases
   - API unchanged

2. ✅ **Code Duplication Eliminated**
   - < 100 lines of duplication between continuous/discrete
   - SymbolicSystemBase contains all common code

3. ✅ **Performance Maintained**
   - Integration performance within 5% of original
   - Code generation unchanged
   - Memory usage similar

4. ✅ **Full Test Coverage**
   - 95%+ overall coverage
   - 100% coverage on new code
   - All edge cases tested

5. ✅ **Stochastic Systems Work**
   - All SDE tests pass
   - Monte Carlo simulations work
   - Diffusion handling unchanged

### Should Have (Important but Not Critical)

6. ✅ **Documentation Complete**
   - Architecture docs updated
   - API reference complete
   - Migration guide available
   - Examples updated

7. ✅ **DiscretizedSystem Multi-Mode**
   - All three modes implemented
   - BATCH mode is 5-10x faster
   - Automatic mode selection works

8. ✅ **Type Safety**
   - Full type hints
   - TypedDict for results
   - MyPy passes with no errors

### Nice to Have (Enhancements)

9. ✅ **Performance Improvements**
   - Faster than original (due to cleaner code)
   - Better caching strategies

10. ✅ **Extended Functionality**
    - Easier to add new system types
    - Better error messages
    - More comprehensive utilities

---

## Timeline and Resources

### Week-by-Week Breakdown

**Week 1: Foundation**
- Mon-Tue: Create SymbolicSystemBase (~12 hours)
- Wed-Thu: Create ContinuousDynamicalSystem (~12 hours)
- Fri: Testing and documentation (~6 hours)
- **Total**: 30 hours

**Week 2: Continuous Systems Complete**
- Mon-Tue: Integration with IntegratorFactory (~10 hours)
- Wed: Migration aliases and compatibility (~6 hours)
- Thu-Fri: Comprehensive testing (~10 hours)
- **Total**: 26 hours

**Week 3: Discrete Systems**
- Mon-Tue: Create DiscreteDynamicalSystem (~10 hours)
- Wed: Testing (~6 hours)
- Thu: Migration and compatibility (~5 hours)
- Fri: Documentation (~5 hours)
- **Total**: 26 hours

**Week 4: Stochastic Systems**
- Mon-Tue: Rename and refactor ContinuousStochasticSystem (~10 hours)
- Wed: Update DiscreteStochasticSystem (~6 hours)
- Thu-Fri: Testing and verification (~10 hours)
- **Total**: 26 hours

**Week 5: Discretization**
- Mon-Tue: Create DiscretizedSystem with three modes (~12 hours)
- Wed: Interpolation and mode switching (~6 hours)
- Thu: Testing all modes (~6 hours)
- Fri: Performance analysis and documentation (~6 hours)
- **Total**: 30 hours

**Week 6: Finalization**
- Mon-Tue: Integration testing (~10 hours)
- Wed: Documentation updates (~8 hours)
- Thu: Migration guide and examples (~6 hours)
- Fri: Final review and cleanup (~6 hours)
- **Total**: 30 hours

### Resource Requirements

**Developer Time**: ~168 hours (1 full-time developer for 6 weeks)

**Tools Needed**:
- Python 3.9+
- pytest, pytest-cov
- mypy for type checking
- black, isort for formatting
- Git for version control

**Optional**:
- hypothesis for property-based testing
- memory_profiler for performance analysis

---

## Appendix A: File Structure (Before/After)

### Before Refactoring

```
src/systems/
├── base/
│   ├── symbolic_dynamical_system.py       # 2035 lines (BLOATED)
│   ├── discrete_symbolic_system.py        # 450 lines (DUPLICATION)
│   ├── stochastic_dynamical_system.py     # 1391 lines
│   ├── discrete_stochastic_system.py      # 624 lines
│   ├── continuous_system_base.py          # 471 lines (interface)
│   └── discrete_system_base.py            # 496 lines (interface)
└── ...
```

### After Refactoring

```
src/systems/
├── base/
│   ├── symbolic_system_base.py            # 1800 lines [NEW]
│   ├── continuous_system_base.py          # 471 lines [UNCHANGED]
│   ├── discrete_system_base.py            # 496 lines [UNCHANGED]
│   └── utils/  # Unchanged
│       ├── backend_manager.py
│       ├── code_generator.py
│       └── ...
├── continuous_dynamical_system.py         # 300 lines [NEW - replaces symbolic_dynamical_system.py]
├── discrete_dynamical_system.py           # 300 lines [NEW - replaces discrete_symbolic_system.py]
├── continuous_stochastic_system.py        # 400 lines [RENAMED from stochastic_dynamical_system.py]
├── discrete_stochastic_system.py          # 200 lines [UPDATED]
└── discretization/
    ├── __init__.py
    └── discretized_system.py              # 400 lines [NEW]
```

**Net Result**:
- Before: ~5,500 lines
- After: ~4,400 lines + better organization
- Savings: ~1,100 lines (20% reduction)
- Duplication: Eliminated

---

## Appendix B: Migration Checklist

### For Each Phase

- [ ] Create new file(s)
- [ ] Implement all required methods
- [ ] Add comprehensive docstrings
- [ ] Add type hints
- [ ] Create unit tests
- [ ] Run existing tests (ensure no breaks)
- [ ] Add integration tests
- [ ] Performance benchmark
- [ ] Update documentation
- [ ] Add examples
- [ ] Code review
- [ ] Merge to main

### Final Checklist

- [ ] All phases complete
- [ ] All tests passing
- [ ] Performance validated
- [ ] Documentation complete
- [ ] Migration guide written
- [ ] Examples updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Release notes written

---

## Appendix C: Quick Reference

### Old → New Class Names

| Old Name | New Name | Status |
|----------|----------|--------|
| `SymbolicDynamicalSystem` | `ContinuousDynamicalSystem` | Alias remains |
| `DiscreteSymbolicSystem` | `DiscreteDynamicalSystem` | Alias remains |
| `StochasticDynamicalSystem` | `ContinuousStochasticSystem` | Alias remains |
| `DiscreteStochasticSystem` | `DiscreteStochasticSystem` | Name unchanged |

### Import Changes (Optional)

```python
# OLD:
from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem

# NEW (preferred):
from src.systems import ContinuousDynamicalSystem

# NEW (alternative):
from src.systems.continuous_dynamical_system import ContinuousDynamicalSystem
```

### Usage Examples (Unchanged)

```python
# System definition - IDENTICAL to before
class Pendulum(ContinuousDynamicalSystem):  # Just class name changed
    def define_system(self, m=1.0, l=0.5, g=9.81):
        # ... exactly the same as before
        pass

# Usage - IDENTICAL
system = Pendulum(m=2.0)
result = system.integrate(x0, None, (0, 10))
A, B = system.linearize(x_eq, u_eq)
```

---

## Conclusion

This refactoring achieves:

✅ **Massive code reduction** (~1,800 lines of duplication eliminated)  
✅ **Clear architecture** (three-layer separation of concerns)  
✅ **Zero breaking changes** (via migration aliases)  
✅ **Better extensibility** (easy to add new system types)  
✅ **Type safety** (abstract interfaces + TypedDict)  
✅ **Integration framework** fits perfectly (no changes needed)  
✅ **Discretization** solved elegantly (multi-mode approach)  

**Timeline**: 6 weeks for one developer  
**Risk**: Low (comprehensive mitigation strategies)  
**Benefit**: Sustainable, maintainable, extensible architecture  

**Recommendation**: Proceed with refactoring. The benefits far outweigh the costs, and the migration path is straightforward.

---

**Document Version**: 1.0  
**Last Updated**: December 2025  
**Status**: Ready for Implementation
