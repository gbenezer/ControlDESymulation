# Technical Debt Registry

## Overview

This document tracks known technical debt, planned refactorings, and architectural improvements for the ControlDESymulation project.

**Purpose**: Ensure technical debt is visible, tracked, and eventually addressed.

**Update**: Add new items as they're created, mark as COMPLETE when resolved.

---

## Active Technical Debt

### TD-001: IntegrationResult Type Mismatch 🔴 HIGH PRIORITY

**Status**: ACTIVE  
**Created**: Phase 2.1 (December 2025)  
**Target**: Phase 2.5  
**Effort**: 2-3 days  
**Owner**: TBD

#### Problem

There are **two different types** with the same name:

1. **`integrator_base.IntegrationResult`** (Class)
   - Location: `src/systems/base/numerical_integration/integrator_base.py`
   - Returns: Class instance with `.x` attribute
   - Shape: `(n_points, nx)` - time first

2. **`src.types.trajectories.IntegrationResult`** (TypedDict)
   - Location: `src/types/trajectories.py`
   - Returns: Dict with `["y"]` key
   - Shape: `(nx, n_points)` - dimension first (scipy convention)

**Current workaround**: Adapter in `ContinuousSymbolicSystem.integrate()` converts class → TypedDict.

#### Why This Is Debt

- ❌ Violates design principle: "Result types are TypedDict"
- ❌ Inconsistent with all other results (LQR, MPC, etc.)
- ❌ Requires conversion at every boundary
- ❌ Different shape conventions cause confusion
- ❌ Limits static type checking

#### Refactoring Plan

**Goal**: Make all integrators return `IntegrationResult` TypedDict directly.

**Scope**:
- 5 integrator implementations (~2000 lines)
- 1 factory class (~400 lines)
- All integrator tests (~1000 lines)
- Remove adapter from `ContinuousSymbolicSystem`

**Steps**:
1. Update `integrator_base.py` - remove class, import TypedDict
2. Update `scipy_integrator.py` - return TypedDict
3. Update `diffeqpy_integrator.py` - return TypedDict
4. Update `diffrax_integrator.py` - return TypedDict
5. Update `torchdiffeq_integrator.py` - return TypedDict
6. Update `fixed_step_integrators.py` - return TypedDict
7. Standardize on scipy shape convention: `(nx, n_points)`
8. Update all tests
9. Remove adapter from `ContinuousSymbolicSystem.integrate()`
10. Update documentation

**Estimated effort**: 2-3 days

**Benefits**:
- ✅ Consistent with design philosophy
- ✅ Uniform dict access everywhere
- ✅ Better static type checking
- ✅ No conversion overhead
- ✅ Scipy convention alignment

#### Related Files

**Files with technical debt**:
- `src/systems/base/numerical_integration/integrator_base.py` (class definition)
- `src/systems/base/core/continuous_symbolic_system.py` (adapter)
- All 5 integrator implementations

**Files to update**:
- All integrator implementations
- All integrator tests
- `ContinuousSymbolicSystem.integrate()`
- Documentation

#### References

- Design document: `INTEGRATOR_TYPEDDICT_REFACTORING.md`
- Decision rationale: `FINAL_RECOMMENDATION.md`
- Original discussion: Phase 2.1 review

#### Success Criteria

- [ ] All integrators return TypedDict
- [ ] No class-based IntegrationResult exists
- [ ] All results use `(nx, n_points)` shape
- [ ] No adapter needed in `ContinuousSymbolicSystem`
- [ ] All tests pass
- [ ] Static type checking passes
- [ ] Documentation updated

---

## Completed Technical Debt

### TD-000: Example Completed Item ✅

**Status**: COMPLETE  
**Completed**: Phase X  
**Effort**: X hours

Brief description of what was done...

---

## Future Considerations

### FC-001: Discrete System Integration

**Priority**: MEDIUM  
**Phase**: 3+

Need to establish similar integration patterns for discrete-time systems. Should follow same TypedDict conventions.

### FC-002: SDE Integrator Unification

**Priority**: LOW  
**Phase**: TBD

SDE integrators may need similar refactoring to match continuous integrators.

---

## How to Use This Document

### Adding New Technical Debt

```markdown
### TD-XXX: Short Description 🔴/🟡/🟢

**Status**: ACTIVE  
**Created**: Phase X.X  
**Target**: Phase Y.Y  
**Effort**: X hours/days  
**Owner**: Name

#### Problem
What is wrong and why it's debt.

#### Refactoring Plan
How to fix it.

#### Success Criteria
- [ ] Criterion 1
- [ ] Criterion 2
```

### Priority Levels

- 🔴 **HIGH**: Blocks other work or violates core principles
- 🟡 **MEDIUM**: Should be fixed but not urgent
- 🟢 **LOW**: Nice to have, low impact

### Status Values

- **ACTIVE**: Known debt, not yet started
- **IN_PROGRESS**: Currently being addressed
- **BLOCKED**: Waiting on dependency
- **COMPLETE**: Resolved and verified

---

## Review Schedule

- **Weekly**: Check ACTIVE items, update priorities
- **Phase end**: Review all items, plan next phase work
- **Quarterly**: Archive completed items, add new items

Last reviewed: December 2025
