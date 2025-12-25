# Phase 3: Type System Integration and Adoption
## Refactoring Existing Code to Use `src/types`

**This is Phase 3 in the renumbered checklist (formerly Phase 2.5)**

---

## Overview

This phase ensures that all existing code properly uses the type definitions, structured result types, and utility types from `src/types/`. This is critical for:

1. **Type Safety**: Enable static type checking with mypy/pyright
2. **API Consistency**: Standardize return types across the codebase
3. **Documentation**: Self-documenting code through type hints
4. **IDE Support**: Better autocomplete and inline documentation

---

## Phase 3.1: Core Types Adoption (`src/types/core.py`)

### Files That Should Use Core Types

**Target Files** (approx. 50+ files):
- All files in `src/systems/base/core/`
- All files in `src/systems/base/numerical_integration/`
- All files in `src/systems/base/discretization/`
- All files in `src/systems/base/utils/`
- All files in `src/systems/builtin/`

### Core Types to Adopt

```python
# From src/types/core.py
ArrayLike          # NumPy, PyTorch, JAX arrays
StateVector        # System state x
ControlVector      # Control input u
OutputVector       # System output y
TimeArray          # Time points for simulation
ScalarLike         # Scalar values (float, int, array scalar)
```

### Implementation Checklist

#### 2.5.1.1 Update Base System Classes
**Files**: 
- `src/systems/base/core/continuous_symbolic_system.py`
- `src/systems/base/core/continuous_stochastic_system.py`
- `src/systems/base/core/discrete_symbolic_system.py`
- `src/systems/base/core/discrete_stochastic_system.py`

**Tasks**:
- [ ] Replace `np.ndarray` with `ArrayLike` in type hints
- [ ] Use `StateVector` for state variables
- [ ] Use `ControlVector` for control inputs
- [ ] Use `OutputVector` for outputs
- [ ] Update `__call__` signature: `def __call__(self, x: StateVector, u: ControlVector) -> StateVector`
- [ ] Update `linearize` return types to use structured types
- [ ] Run mypy on updated files

**Example Before**:
```python
def __call__(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
    """Evaluate dynamics."""
    pass
```

**Example After**:
```python
from src.types.core import StateVector, ControlVector

def __call__(self, x: StateVector, u: Optional[ControlVector] = None) -> StateVector:
    """Evaluate dynamics."""
    pass
```

**Dependencies**: Phase 2.1, 2.2  
**Estimated Time**: 3-4 hours  
**Risk**: Low

#### 2.5.1.2 Update Integrators
**Files**: All files in `src/systems/base/numerical_integration/`

**Tasks**:
- [ ] Use `StateVector` for state variables
- [ ] Use `TimeArray` for time arrays
- [ ] Use `ArrayLike` for generic arrays
- [ ] Update function signatures
- [ ] Run mypy validation

**Dependencies**: 2.5.1.1  
**Estimated Time**: 2-3 hours  
**Risk**: Low

#### 2.5.1.3 Update Discretizers
**Files**: All files in `src/systems/base/discretization/`

**Tasks**:
- [ ] Use `StateVector`, `ControlVector` in signatures
- [ ] Use `ScalarLike` for dt parameter
- [ ] Update type hints throughout
- [ ] Run mypy validation

**Dependencies**: 2.5.1.1  
**Estimated Time**: 2-3 hours  
**Risk**: Low

---

## Phase 3.2: Backend Types Adoption (`src/types/backends.py`)

### Files That Should Use Backend Types

**Target Files**:
- `src/systems/base/utils/backend_manager.py`
- `src/systems/base/utils/code_generator.py`
- All integrator files
- All utility files that handle multi-backend arrays

### Backend Types to Adopt

```python
# From src/types/backends.py
BackendType        # Literal["numpy", "torch", "jax"]
NumpyArray         # numpy.ndarray
TorchTensor        # torch.Tensor
JaxArray           # jax.Array
BackendArray       # Union[NumpyArray, TorchTensor, JaxArray]
```

### Implementation Checklist

#### 2.5.2.1 Update Backend Manager
**File**: `src/systems/base/utils/backend_manager.py`

**Tasks**:
- [ ] Use `BackendType` for backend specifications
- [ ] Use `BackendArray` for return types
- [ ] Add type guards for backend detection
- [ ] Update all method signatures
- [ ] Run mypy validation

**Dependencies**: Phase 7.1  
**Estimated Time**: 2 hours  
**Risk**: Low

#### 2.5.2.2 Update Code Generator
**File**: `src/systems/base/utils/code_generator.py`

**Tasks**:
- [ ] Use `BackendType` for target backend
- [ ] Update generated code type hints
- [ ] Run mypy validation

**Dependencies**: 2.5.2.1  
**Estimated Time**: 1-2 hours  
**Risk**: Low

---

## Phase 3.3: Symbolic Types Adoption (`src/types/symbolic.py`)

### Files That Should Use Symbolic Types

**Target Files**:
- All files in `src/systems/base/core/` that use SymPy
- `src/systems/base/utils/symbolic_validator.py`
- `src/systems/base/utils/linearization_engine.py`

### Symbolic Types to Adopt

```python
# From src/types/symbolic.py
SymbolicExpression    # Union[sp.Expr, sp.Symbol, sp.Matrix]
SymbolicMatrix        # sp.Matrix or sp.MatrixSymbol
SymbolicVector        # sp.Matrix (column vector)
SymbolicScalar        # sp.Expr or sp.Symbol
StateSymbols          # List of state variable symbols
```

### Implementation Checklist

#### 2.5.3.1 Update Core System Classes
**Files**: All symbolic system files

**Tasks**:
- [ ] Use `SymbolicMatrix` for F (drift) matrix
- [ ] Use `SymbolicMatrix` for G (diffusion) matrix
- [ ] Use `StateSymbols` for state variable lists
- [ ] Update property return types
- [ ] Run mypy validation

**Dependencies**: 2.5.1.1  
**Estimated Time**: 2-3 hours  
**Risk**: Low

#### 2.5.3.2 Update Symbolic Utilities
**Files**:
- `src/systems/base/utils/symbolic_validator.py`
- `src/systems/base/utils/linearization_engine.py`

**Tasks**:
- [ ] Use symbolic types in all signatures
- [ ] Update validation functions
- [ ] Run mypy validation

**Dependencies**: 2.5.3.1  
**Estimated Time**: 1-2 hours  
**Risk**: Low

---

## Phase 3.4: Linearization Types Adoption (`src/types/linearization.py`)

### Files That Should Use Linearization Types

**Target Files**:
- All system classes (for `linearize` method)
- `src/systems/base/utils/linearization_engine.py`
- `src/systems/base/discretization/discrete_linearization.py`
- All control modules
- All observer modules

### Linearization Types to Adopt

```python
# From src/types/linearization.py
LinearizationResult         # Base result for linearization
ContinuousLinearization     # (A, B) matrices
DiscreteLinearization       # (Ad, Bd) matrices
StochasticLinearization     # (A, B, G) matrices for SDEs
LinearizedStateSpace        # Complete linearized system
```

### Implementation Checklist

#### 2.5.4.1 Update System Linearize Methods
**Files**: All system classes

**Tasks**:
- [ ] `ContinuousSymbolicSystem.linearize()` returns `ContinuousLinearization`
- [ ] `ContinuousStochasticSystem.linearize()` returns `StochasticLinearization`
- [ ] `DiscreteSymbolicSystem.linearize()` returns `DiscreteLinearization`
- [ ] `DiscreteStochasticSystem.linearize()` returns stochastic discrete result
- [ ] Update all docstrings
- [ ] Run mypy validation

**Example Before**:
```python
def linearize(self, x_eq, u_eq):
    """Returns tuple (A, B)"""
    return A, B
```

**Example After**:
```python
from src.types.linearization import ContinuousLinearization

def linearize(self, x_eq, u_eq) -> ContinuousLinearization:
    """Returns structured linearization result."""
    A, B = ...  # compute
    return ContinuousLinearization(A=A, B=B, x_eq=x_eq, u_eq=u_eq)
```

**Dependencies**: 2.5.1.1  
**Estimated Time**: 4-5 hours  
**Risk**: Medium (breaking change for consumers)

#### 2.5.4.2 Update Linearization Engine
**File**: `src/systems/base/utils/linearization_engine.py`

**Tasks**:
- [ ] Return structured linearization types
- [ ] Update all helper functions
- [ ] Run mypy validation

**Dependencies**: 2.5.4.1  
**Estimated Time**: 2-3 hours  
**Risk**: Low

#### 2.5.4.3 Update Control Modules
**Files**: All files in `src/control/`

**Tasks**:
- [ ] Accept `LinearizationResult` types
- [ ] Extract A, B matrices from structured types
- [ ] Update type hints
- [ ] Run control tests

**Dependencies**: 2.5.4.1  
**Estimated Time**: 2-3 hours  
**Risk**: Low

---

## Phase 3.5: Trajectory Types Adoption (`src/types/trajectories.py`)

### Files That Should Use Trajectory Types

**Target Files**:
- All integrator files
- All simulator files
- `src/visualization/trajectory_plotter.py`
- All control modules that simulate systems

### Trajectory Types to Adopt

```python
# From src/types/trajectories.py
Trajectory              # Base trajectory container
StateTrajectory         # State evolution over time
ControlTrajectory       # Control sequence over time
SimulationResult        # Complete simulation (states, controls, outputs, time)
MonteCarloResult       # Ensemble of stochastic simulations
```

### Implementation Checklist

#### 2.5.5.1 Update Integrators
**Files**: All integrator files

**Tasks**:
- [ ] `integrate()` method returns `SimulationResult`
- [ ] Include time, states, controls, outputs in result
- [ ] Update docstrings
- [ ] Run integrator tests

**Example Before**:
```python
def integrate(self, x0, u, t_span):
    """Returns tuple (t, x)"""
    return t, x
```

**Example After**:
```python
from src.types.trajectories import SimulationResult

def integrate(self, x0, u, t_span) -> SimulationResult:
    """Returns structured simulation result."""
    t, x = ...  # compute
    return SimulationResult(
        time=t,
        states=x,
        controls=u_eval,
        outputs=y if has_output else None
    )
```

**Dependencies**: Phase 6.1, 6.2  
**Estimated Time**: 4-5 hours  
**Risk**: Medium (breaking change)

#### 2.5.5.2 Update Simulators
**Files**: 
- `src/systems/base/discretization/discrete_simulator.py`
- `src/systems/base/discretization/stochastic/stochastic_discrete_simulator.py`

**Tasks**:
- [ ] `simulate()` method returns `SimulationResult`
- [ ] Stochastic simulator returns `MonteCarloResult`
- [ ] Update docstrings
- [ ] Run simulator tests

**Dependencies**: 2.5.5.1, Phase 5  
**Estimated Time**: 3-4 hours  
**Risk**: Medium

#### 2.5.5.3 Update Trajectory Plotter
**File**: `src/visualization/trajectory_plotter.py`

**Tasks**:
- [ ] Accept `SimulationResult` or `MonteCarloResult`
- [ ] Extract data from structured types
- [ ] Update plotting functions
- [ ] Run visualization tests

**Dependencies**: 2.5.5.1, 2.5.5.2  
**Estimated Time**: 2-3 hours  
**Risk**: Low

---

## Phase 3.6: Estimation Types Adoption (`src/types/estimation.py`)

### Files That Should Use Estimation Types

**Target Files**:
- All files in `src/observers/`
- State estimation utilities

### Estimation Types to Adopt

```python
# From src/types/estimation.py
EstimationResult        # Base estimation result
KalmanFilterResult      # Kalman filter output
ObserverGains           # Observer gain matrices
EstimationCovariance    # Covariance matrices
```

### Implementation Checklist

#### 2.5.6.1 Update Observer Modules
**Files**:
- `src/observers/kalman_filter.py`
- `src/observers/extended_kalman_filter.py`
- `src/observers/luenberger_observer.py`

**Tasks**:
- [ ] Observer methods return `EstimationResult`
- [ ] Kalman filter returns `KalmanFilterResult`
- [ ] Include covariances in results
- [ ] Update type hints
- [ ] Run observer tests

**Dependencies**: Phase 10.2  
**Estimated Time**: 3-4 hours  
**Risk**: Low-Medium

---

## Phase 3.7: Control Types Adoption (`src/types/control_classical.py` & `control_advanced.py`)

### Files That Should Use Control Types

**Target Files**:
- All files in `src/control/`

### Control Types to Adopt

```python
# From src/types/control_classical.py
LQRResult               # LQR controller result
LQGResult               # LQG controller result
ControllerGains         # Feedback gain matrices

# From src/types/control_advanced.py
MPCResult               # MPC solution
MPCProblem              # MPC problem specification
RobustControlResult     # Robust control design
```

### Implementation Checklist

#### 2.5.7.1 Update Classical Control
**Files**:
- `src/control/lqr.py`
- `src/control/pid.py`

**Tasks**:
- [ ] LQR design returns `LQRResult`
- [ ] Include cost matrices, gains, closed-loop poles
- [ ] Update type hints
- [ ] Run control tests

**Dependencies**: Phase 10.1  
**Estimated Time**: 2-3 hours  
**Risk**: Low

#### 2.5.7.2 Update Advanced Control
**Files**:
- `src/control/mpc.py`
- `src/control/feedback_linearization.py`

**Tasks**:
- [ ] MPC solve returns `MPCResult`
- [ ] MPC setup takes `MPCProblem`
- [ ] Update type hints
- [ ] Run control tests

**Dependencies**: Phase 10.1  
**Estimated Time**: 2-3 hours  
**Risk**: Low

---

## Phase 3.8: Optimization Types Adoption (`src/types/optimization.py`)

### Files That Should Use Optimization Types

**Target Files**:
- MPC solver internals
- Trajectory optimization modules
- Parameter identification modules

### Optimization Types to Adopt

```python
# From src/types/optimization.py
OptimizationProblem     # Generic optimization problem
OptimizationResult      # Optimization solution
Constraints             # Constraint specifications
ObjectiveFunction       # Objective function definition
```

### Implementation Checklist

#### 2.5.8.1 Update MPC Internals
**File**: `src/control/mpc.py`

**Tasks**:
- [ ] Use `OptimizationProblem` for MPC QP
- [ ] Return `OptimizationResult` from solver
- [ ] Update type hints
- [ ] Run tests

**Dependencies**: 2.5.7.2  
**Estimated Time**: 2-3 hours  
**Risk**: Low

---

## Phase 3.9: Learning Types Adoption (`src/types/learning.py`)

### Files That Should Use Learning Types

**Target Files**:
- Learning-based control modules (if present)
- System identification with ML
- Data-driven dynamics

### Learning Types to Adopt

```python
# From src/types/learning.py
RLPolicy                # Reinforcement learning policy
TrainingResult          # Training metrics
Dataset                 # Training/validation data
```

### Implementation Checklist

#### 2.5.9.1 Update Learning Modules (if applicable)
**Files**: TBD based on actual learning modules

**Tasks**:
- [ ] Identify learning modules
- [ ] Apply appropriate types
- [ ] Update type hints
- [ ] Run tests

**Dependencies**: Phase varies  
**Estimated Time**: 2-4 hours  
**Risk**: Low

---

## Phase 3.10: Identification Types Adoption (`src/types/identification.py`)

### Files That Should Use Identification Types

**Target Files**:
- System identification modules
- Parameter estimation modules

### Identification Types to Adopt

```python
# From src/types/identification.py
IdentificationResult    # System ID result
ParameterEstimate       # Estimated parameters
ValidationMetrics       # Model validation metrics
```

### Implementation Checklist

#### 2.5.10.1 Update Identification Modules (if applicable)
**Files**: TBD based on actual identification modules

**Tasks**:
- [ ] Identify system ID modules
- [ ] Apply appropriate types
- [ ] Update type hints
- [ ] Run tests

**Dependencies**: Phase varies  
**Estimated Time**: 2-4 hours  
**Risk**: Low

---

## Phase 3.11: Comprehensive Type Checking

### 2.5.11.1 Set Up Type Checking Infrastructure
**Files**: Project root

**Tasks**:
- [ ] Create `mypy.ini` or `pyproject.toml` with mypy config
- [ ] Set strict mode: `strict = true`
- [ ] Add type checking to CI/CD pipeline
- [ ] Document type checking requirements

**Example `mypy.ini`**:
```ini
[mypy]
python_version = 3.10
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[mypy-numpy.*]
ignore_missing_imports = true

[mypy-scipy.*]
ignore_missing_imports = true
```

**Dependencies**: All previous 2.5.x phases  
**Estimated Time**: 1-2 hours  
**Risk**: Low

### 2.5.11.2 Run Full Type Check
**Tasks**:
- [ ] Run `mypy src/` on entire codebase
- [ ] Fix all type errors
- [ ] Document any `type: ignore` comments
- [ ] Aim for 100% type coverage

**Dependencies**: 2.5.11.1  
**Estimated Time**: 4-8 hours (depends on errors found)  
**Risk**: Medium

### 2.5.11.3 Add Runtime Type Validation (Optional)
**Tasks**:
- [ ] Consider using `typeguard` or `beartype` for runtime checking
- [ ] Add runtime validation in critical paths
- [ ] Configure for debug/development mode only

**Dependencies**: 2.5.11.2  
**Estimated Time**: 2-3 hours  
**Risk**: Low

---

## Summary: Files That MUST Use Types from `src/types`

### Critical (High Priority)

| Module | Files | Types Used |
|--------|-------|-----------|
| **Core Systems** | All in `src/systems/base/core/` | `StateVector`, `ControlVector`, `OutputVector`, `ContinuousLinearization`, `DiscreteLinearization` |
| **Integrators** | All in `src/systems/base/numerical_integration/` | `StateVector`, `TimeArray`, `SimulationResult` |
| **Discretizers** | All in `src/systems/base/discretization/` | `StateVector`, `ControlVector`, `DiscreteLinearization` |
| **Linearization** | `linearization_engine.py` | `LinearizationResult`, `ContinuousLinearization`, `DiscreteLinearization` |
| **Control** | All in `src/control/` | `LQRResult`, `MPCResult`, `ControllerGains` |
| **Observers** | All in `src/observers/` | `EstimationResult`, `KalmanFilterResult` |

### Important (Medium Priority)

| Module | Files | Types Used |
|--------|-------|-----------|
| **Backend Manager** | `backend_manager.py` | `BackendType`, `BackendArray` |
| **Code Generator** | `code_generator.py` | `BackendType`, `SymbolicExpression` |
| **Symbolic Utils** | `symbolic_validator.py` | `SymbolicMatrix`, `SymbolicExpression` |
| **Visualization** | `trajectory_plotter.py` | `SimulationResult`, `MonteCarloResult` |

### Optional (Low Priority)

| Module | Files | Types Used |
|--------|-------|-----------|
| **Built-in Systems** | All in `src/systems/builtin/` | Core types in method signatures |
| **Utilities** | Misc utils | Appropriate types as needed |

---

## Integration with Main Checklist

**INSERT PHASE 2.5 AFTER PHASE 2 (Rename Continuous Systems)**

### Updated Dependencies

- **Phase 3** (Discrete Systems) now depends on: Phase 3.1, 2.5.3, 2.5.4
- **Phase 4** (Wrapper) now depends on: Phase 3.1, 2.5.5
- **Phase 5** (Discretization) now depends on: Phase 3.4, 2.5.5
- **Phase 6** (Integration) now depends on: Phase 3.5
- **Phase 7** (Utils) now depends on: Phase 3.2, 2.5.3
- **Phase 10** (Control/Observers) now depends on: Phase 3.6, 2.5.7

---

## Updated Risk Matrix

| Sub-Phase | Risk Level | Mitigation Strategy |
|-----------|-----------|---------------------|
| 2.5.1 (Core Types) | Low | Incremental adoption, maintain backward compatibility |
| 2.5.2 (Backend Types) | Low | Limited scope, well-defined types |
| 2.5.3 (Symbolic Types) | Low | SymPy integration already exists |
| 2.5.4 (Linearization) | Medium | Breaking change, provide migration guide |
| 2.5.5 (Trajectories) | Medium | Breaking change, provide migration guide |
| 2.5.6 (Estimation) | Low | Limited files affected |
| 2.5.7 (Control) | Low | Limited files affected |
| 2.5.8-10 | Low | Optional modules, limited impact |
| 2.5.11 (Type Check) | Medium | May reveal hidden bugs, time to fix |

---

## Updated Timeline

| Sub-Phase | Estimated Time | Cumulative (within Phase 2.5) |
|-----------|---------------|-------------------------------|
| 2.5.1 | 9 hours | 9 hours |
| 2.5.2 | 4 hours | 13 hours |
| 2.5.3 | 5 hours | 18 hours |
| 2.5.4 | 11 hours | 29 hours |
| 2.5.5 | 12 hours | 41 hours |
| 2.5.6 | 4 hours | 45 hours |
| 2.5.7 | 6 hours | 51 hours |
| 2.5.8 | 3 hours | 54 hours |
| 2.5.9-10 | 6 hours | 60 hours |
| 2.5.11 | 8 hours | 68 hours |

**Phase 2.5 Total**: ~68 hours (~1.5-2 weeks)

---

## Breaking Changes Introduced

### Method Return Types Changed

1. **`linearize()` methods**: Now return structured types instead of tuples
   - Migration: Unpack from result objects instead of tuples
   
2. **`integrate()` methods**: Now return `SimulationResult` instead of `(t, x)` tuple
   - Migration: Access via `result.time`, `result.states`
   
3. **`simulate()` methods**: Now return `SimulationResult` or `MonteCarloResult`
   - Migration: Access via structured attributes

### Backward Compatibility Strategy

**Option 1: Property-based unpacking** (RECOMMENDED)
```python
class SimulationResult:
    def __iter__(self):
        """Allow tuple unpacking for backward compatibility."""
        yield self.time
        yield self.states
    
# Works with old code:
t, x = system.integrate(...)  # Still works!
# And new code:
result = system.integrate(...)
t = result.time
```

**Option 2: Deprecation period**
- Keep old method names (e.g., `linearize_tuple()`)
- New methods return structured types
- Deprecate old methods after 2 versions

---

## Success Criteria for Phase 2.5

✅ All core system methods use appropriate types from `src/types`  
✅ `mypy --strict` passes with 0 errors  
✅ All return types are structured (no naked tuples)  
✅ All function signatures have complete type hints  
✅ Backward compatibility maintained (or migration guide provided)  
✅ All tests pass with new type signatures  
✅ Documentation updated with type information  
✅ IDE autocomplete works for all structured types  

---

## References

- `src/types/core.py` - Core array and vector types
- `src/types/backends.py` - Multi-backend array types
- `src/types/symbolic.py` - SymPy expression types
- `src/types/linearization.py` - Linearization result types
- `src/types/trajectories.py` - Simulation and trajectory types
- `src/types/estimation.py` - State estimation types
- `src/types/control_classical.py` - Classical control types
- `src/types/control_advanced.py` - Advanced control types
- `src/types/optimization.py` - Optimization problem types
- `src/types/learning.py` - Learning and RL types
- `src/types/identification.py` - System identification types

---

**END OF PHASE 2.5**
