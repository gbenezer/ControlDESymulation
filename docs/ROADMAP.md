# ControlDESymulation Refactoring Roadmap

## Project Vision

Transform ControlDESymulation into a clean, type-driven, 4-layer architecture with comprehensive symbolic and numerical capabilities.

---

## Phase Timeline

### ✅ Phase 1: Foundation (COMPLETE)
**Duration**: Completed  
**Deliverables**:
- `ContinuousSystemBase` abstract interface
- `DiscreteSystemBase` abstract interface
- Comprehensive type system (170+ types)
- 130+ passing tests

### 🚀 Phase 2: Symbolic Systems (IN PROGRESS)

#### ✅ Phase 2.1: ContinuousSymbolicSystem (COMPLETE)
**Status**: Complete  
**Deliverables**:
- `ContinuousSymbolicSystem` inherits from `ContinuousSystemBase`
- All abstract methods implemented
- Proper TypedDict return types
- Backward compatibility via `SymbolicDynamicalSystem` alias
- Multi-backend integration via `IntegratorFactory`

**Known Debt**: TD-001 (IntegrationResult type mismatch) - deferred to Phase 2.5

#### 🔄 Phase 2.2: ContinuousStochasticSystem (NEXT)
**Status**: Ready to start  
**Duration**: 3-4 hours  
**Deliverables**:
- `ContinuousStochasticSystem` inherits from `ContinuousSymbolicSystem`
- Overrides `linearize()` to return `(A, B, G)` tuple
- SDE integration support
- `is_stochastic` property returns `True`
- Backward compatibility via `StochasticDynamicalSystem` alias

**Dependencies**: Phase 2.1 complete ✅

#### 📋 Phase 2.5: Integrator TypedDict Refactoring (PLANNED)
**Status**: Documented, not started  
**Priority**: High (technical debt)  
**Duration**: 2-3 days  
**Effort**: 20-30 files, ~3000 lines

**Goal**: Unify integration result types - make integrators return TypedDict directly.

**Scope**:
1. Remove `IntegrationResult` class from `integrator_base.py`
2. Update all 5 integrator implementations to return TypedDict
3. Standardize on scipy shape convention: `(nx, n_points)`
4. Remove adapter from `ContinuousSymbolicSystem.integrate()`
5. Update all integrator tests
6. Update documentation

**Benefits**:
- ✅ Eliminates technical debt TD-001
- ✅ Design consistency ("Result types are TypedDict")
- ✅ Better type checking
- ✅ No conversion overhead
- ✅ Scipy convention alignment

**Success Criteria**:
- [ ] All integrators return TypedDict
- [ ] No adapter needed in system classes
- [ ] All tests pass
- [ ] Static type checking passes
- [ ] Documentation updated

**Tracking**: See `TECHNICAL_DEBT.md#TD-001`

**Recommendation**: Complete after Phase 2.2, before Phase 3.

---

### 📅 Phase 3: Discrete Systems (FUTURE)

#### Phase 3.1: DiscreteSymbolicSystem
**Status**: Planned  
**Dependencies**: Phase 2 complete

Create `DiscreteSymbolicSystem(DiscreteSystemBase)`:
- Symbolic difference equations
- Multi-backend support
- Discrete linearization

#### Phase 3.2: DiscreteStochasticSystem
**Status**: Planned  
**Dependencies**: Phase 3.1 complete

Create `DiscreteStochasticSystem(DiscreteSymbolicSystem)`:
- Stochastic difference equations
- Discrete-time noise

---

### 📅 Phase 4: Bridges and Converters (FUTURE)

#### Phase 4.1: Discretization
**Status**: Planned  
**Dependencies**: Phase 2 and 3 complete

Create discretization bridges:
- `DiscreteTimeWrapper` - wraps continuous as discrete
- `Discretizer` - converts continuous → discrete
- Multiple discretization methods (euler, tustin, zoh)

#### Phase 4.2: Interpolation
**Status**: Planned

Create continuous-time interpolators:
- `ContinuousTimeInterpolator` - interpolates discrete → continuous
- Spline-based interpolation

---

## Technical Debt Tracking

### Active Debt

| ID | Description | Priority | Target Phase | Effort |
|----|-------------|----------|--------------|--------|
| TD-001 | IntegrationResult type mismatch | 🔴 HIGH | 2.5 | 2-3 days |

See `TECHNICAL_DEBT.md` for full details.

---

## Priority Queue

### Immediate (Next 2 Weeks)
1. ✅ Phase 2.1: ContinuousSymbolicSystem (DONE)
2. 🔄 Phase 2.2: ContinuousStochasticSystem (IN PROGRESS)

### Near-term (Next Month)
3. Phase 2.5: Integrator refactoring (HIGH PRIORITY DEBT)
4. Phase 3.1: DiscreteSymbolicSystem

### Medium-term (Next Quarter)
5. Phase 3.2: DiscreteStochasticSystem
6. Phase 4.1: Discretization bridges

### Long-term (6+ Months)
7. Phase 4.2: Interpolation
8. Additional features TBD

---

## Decision Log

### Decision 2025-12-26: Defer Integrator Refactoring

**Context**: During Phase 2.1, discovered type mismatch between integrator infrastructure and base class interface.

**Options**:
1. Fix now (2-3 days, delays Phase 2.1)
2. Use adapter temporarily (quick, technical debt)
3. Change TypedDict to match integrators (violates design)

**Decision**: Option 2 - Use adapter, fix in Phase 2.5

**Rationale**:
- Maintains design integrity
- Ships Phase 2.1 immediately
- Clear path forward
- Debt is documented and tracked

**Tracking**: TD-001 in `TECHNICAL_DEBT.md`

---

## Success Metrics

### Phase 2 Complete When:
- [x] Phase 2.1 complete
- [ ] Phase 2.2 complete
- [ ] Phase 2.5 complete (debt resolved)
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Zero active HIGH priority debt

### Project Complete When:
- [ ] All 4 layers implemented
- [ ] Zero technical debt
- [ ] Comprehensive test coverage
- [ ] Full documentation
- [ ] Example gallery

---

## Review Schedule

- **Weekly**: Check progress, update priorities
- **Phase completion**: Review debt, plan next phase
- **Monthly**: Update roadmap, adjust timeline

**Last updated**: December 26, 2025
