# GitHub Issue Template for TD-001

---

## Title
[Technical Debt] Refactor Integrators to Use TypedDict Instead of Class

---

## Labels
- `technical-debt`
- `refactoring`
- `high-priority`
- `phase-2.5`

---

## Description

### Summary

The integrator infrastructure currently returns `IntegrationResult` as a **class instance**, while the project design philosophy states "Result types are TypedDict". This creates inconsistency and requires adapter conversion at system boundaries.

**Tracking**: TD-001 in `TECHNICAL_DEBT.md`

### Problem

There are **two types** with the same name:

1. **`integrator_base.IntegrationResult`** (Class)
   - Returns: `result.x` with shape `(n_points, nx)`
   - Access: Attribute-based
   
2. **`src.types.trajectories.IntegrationResult`** (TypedDict)
   - Returns: `result["y"]` with shape `(nx, n_points)`
   - Access: Dict-based

**Current workaround**: Adapter in `ContinuousSymbolicSystem.integrate()` converts class → TypedDict.

### Why This Matters

- ❌ Violates design principle: "Result types are TypedDict"
- ❌ Inconsistent with all other results (LQR, MPC, etc.)
- ❌ Requires conversion overhead at boundaries
- ❌ Different shape conventions cause confusion
- ❌ Limits static type checking benefits

### Scope

**Files to update** (~20-30 files):
- `src/systems/base/numerical_integration/integrator_base.py`
- `src/systems/base/numerical_integration/scipy_integrator.py`
- `src/systems/base/numerical_integration/diffeqpy_integrator.py`
- `src/systems/base/numerical_integration/diffrax_integrator.py`
- `src/systems/base/numerical_integration/torchdiffeq_integrator.py`
- `src/systems/base/numerical_integration/fixed_step_integrators.py`
- `src/systems/base/core/continuous_symbolic_system.py`
- All integrator tests (~1000 lines)

**Estimated effort**: 2-3 days

### Proposed Solution

1. **Remove class from `integrator_base.py`**
   ```python
   # DELETE class definition
   # Import TypedDict from types module instead
   from src.types.trajectories import IntegrationResult
   ```

2. **Update all integrators to return TypedDict**
   ```python
   # Example: scipy_integrator.py
   def integrate(...) -> IntegrationResult:
       result: IntegrationResult = {
           "t": sol.t,
           "y": sol.y,  # scipy convention (nx, n_points)
           "success": sol.success,
           "message": sol.message,
           "nfev": sol.nfev,
           "njev": getattr(sol, 'njev', 0),
           "nlu": getattr(sol, 'nlu', 0),
           "status": sol.status
       }
       return result
   ```

3. **Standardize shape convention**
   - Use scipy convention: `(nx, n_points)` - dimension first
   - Update all integrators to use this consistently
   - Update tests for new shape

4. **Remove adapter from system classes**
   ```python
   # ContinuousSymbolicSystem.integrate()
   def integrate(...) -> IntegrationResult:
       # No conversion needed!
       return integrator.integrate(...)
   ```

5. **Update all tests**
   - Change `result.x` → `result["y"]`
   - Update shape assertions
   - Verify all backends work

### Benefits

- ✅ Consistent with project design philosophy
- ✅ Uniform dict access pattern across all results
- ✅ Better static type checking with mypy
- ✅ No conversion overhead
- ✅ Aligns with scipy/MATLAB conventions
- ✅ Eliminates technical debt

### Implementation Checklist

#### Phase 1: Update Base (1-2 hours)
- [ ] Remove `IntegrationResult` class from `integrator_base.py`
- [ ] Import TypedDict from `src.types.trajectories`
- [ ] Update docstrings and type hints

#### Phase 2: Update Integrators (4-6 hours)
- [ ] Update `scipy_integrator.py`
- [ ] Update `diffeqpy_integrator.py`
- [ ] Update `diffrax_integrator.py`
- [ ] Update `torchdiffeq_integrator.py`
- [ ] Update `fixed_step_integrators.py`
- [ ] Verify all return TypedDict with correct shape

#### Phase 3: Update System Classes (1 hour)
- [ ] Remove adapter from `ContinuousSymbolicSystem.integrate()`
- [ ] Update docstrings
- [ ] Verify return types

#### Phase 4: Update Tests (4-6 hours)
- [ ] Update all integrator unit tests
- [ ] Update system integration tests
- [ ] Add type checking tests
- [ ] Verify all backends

#### Phase 5: Documentation (1-2 hours)
- [ ] Update integrator documentation
- [ ] Update shape convention docs
- [ ] Update migration guide if needed
- [ ] Mark TD-001 as COMPLETE

### Testing Strategy

```bash
# 1. Run all integrator tests
pytest tests/unit/numerical_integration/ -v

# 2. Run system tests
pytest tests/unit/core_class_unit_tests/ -v

# 3. Static type checking
mypy src/systems/base/

# 4. Full test suite
pytest tests/ -v

# 5. Verify examples still work
python examples/pendulum_simulation.py
python examples/lqr_design.py
```

### Success Criteria

- [ ] All integrators return TypedDict
- [ ] No class-based `IntegrationResult` exists
- [ ] All results use `(nx, n_points)` shape
- [ ] No adapter in `ContinuousSymbolicSystem`
- [ ] All tests pass
- [ ] Static type checking passes
- [ ] Documentation updated
- [ ] Zero regressions in examples

### References

- **Design document**: `docs/INTEGRATOR_TYPEDDICT_REFACTORING.md`
- **Decision rationale**: `docs/FINAL_RECOMMENDATION.md`
- **Technical debt registry**: `TECHNICAL_DEBT.md#TD-001`
- **Roadmap**: `ROADMAP.md` - Phase 2.5

### Target

- **Phase**: 2.5
- **Priority**: High (technical debt)
- **Assignee**: TBD
- **Due date**: Before Phase 3 starts

---

## Comments

<!-- Discussion about approach, challenges, progress updates -->

---

## Related Issues

- Phase 2.1 completion (#XXX)
- Type system refactoring (#XXX)

---

**Note**: This is HIGH priority technical debt that should be addressed before moving to Phase 3 (discrete systems) to maintain architectural integrity.
