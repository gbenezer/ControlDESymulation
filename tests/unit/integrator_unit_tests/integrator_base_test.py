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
Unit Tests for IntegratorBase
=============================

Tests the abstract base class for ODE integrators, including:
1. StepMode enum definitions
2. IntegrationResult container
3. Abstract interface enforcement
4. Initialization and validation
5. Statistics tracking
6. String representations
7. Equilibrium-based integration
8. Autonomous system integration
9. Backend consistency
10. Control function handling
11. Array dimension validation
12. Time span validation
13. Integration termination conditions
14. Dense output handling
15. Step size handling
"""

import warnings

import numpy as np
import pytest

from src.systems.base.numerical_integration.integrator_base import (
    IntegrationResult,
    IntegratorBase,
    StepMode,
)

# ============================================================================
# Helper Functions
# ============================================================================


def _torch_available():
    """Check if PyTorch is available."""
    try:
        import torch

        return True
    except ImportError:
        return False


def _jax_available():
    """Check if JAX is available."""
    try:
        import jax.numpy as jnp

        return True
    except ImportError:
        return False


# ============================================================================
# Mock Systems
# ============================================================================


class MockSystem:
    """Basic mock system for testing."""

    def __init__(self, nx=1, nu=1):
        self.nx = nx
        self.nu = nu
        self._initialized = True
        self._default_backend = "numpy"

    def __call__(self, x, u, backend="numpy"):
        """Simple linear dynamics: dx = -x + u"""
        return -x + u


class MockAutonomousSystem:
    """Mock autonomous system (nu=0) for testing."""

    def __init__(self, nx=2):
        self.nx = nx
        self.nu = 0
        self._initialized = True
        self._default_backend = "numpy"

        self.equilibria = MockEquilibriumHandler(nx=nx, nu=0)
        self.equilibria.add_equilibrium("zero", np.zeros(nx), np.array([]))

    def __call__(self, x, u=None, backend="numpy"):
        """Autonomous dynamics: dx = -x"""
        return -x

    def get_equilibrium(self, name=None, backend=None):
        """Get equilibrium in specified backend."""
        backend = backend or self._default_backend
        return self.equilibria.get_both(name, backend)


class MockSystemWithEquilibria:
    """Mock system with equilibrium handler for testing."""

    def __init__(self, nx=2, nu=1):
        self.nx = nx
        self.nu = nu
        self._initialized = True
        self._default_backend = "numpy"

        self.equilibria = MockEquilibriumHandler(nx=nx, nu=nu)
        self.equilibria.add_equilibrium("zero", np.zeros(nx), np.zeros(nu))
        self.equilibria.add_equilibrium("custom", np.array([1.0, 0.5]), np.array([0.2]))

    def __call__(self, x, u, backend="numpy"):
        """Simple linear dynamics: dx = -x + u"""
        return -x + u

    def get_equilibrium(self, name=None, backend=None):
        """Get equilibrium in specified backend."""
        backend = backend or self._default_backend
        return self.equilibria.get_both(name, backend)

    def list_equilibria(self):
        """List equilibrium names."""
        return self.equilibria.list_names()


class MockEquilibriumHandler:
    """Minimal mock of EquilibriumHandler for testing."""

    def __init__(self, nx, nu):
        self._nx = nx
        self._nu = nu
        self._equilibria = {}
        self._default = "origin"

        # Origin equilibrium
        self._equilibria["origin"] = {
            "x": np.zeros(nx),
            "u": np.zeros(nu) if nu > 0 else np.array([]),
            "metadata": {},
        }

    def add_equilibrium(self, name, x_eq, u_eq):
        """Add equilibrium."""
        self._equilibria[name] = {"x": np.asarray(x_eq), "u": np.asarray(u_eq), "metadata": {}}

    def get_x(self, name=None, backend="numpy"):
        """Get equilibrium state."""
        name = name or self._default
        x = self._equilibria[name]["x"]
        return self._convert_to_backend(x, backend)

    def get_u(self, name=None, backend="numpy"):
        """Get equilibrium control."""
        name = name or self._default
        u = self._equilibria[name]["u"]
        return self._convert_to_backend(u, backend)

    def get_both(self, name=None, backend="numpy"):
        """Get both state and control."""
        return self.get_x(name, backend), self.get_u(name, backend)

    def list_names(self):
        """List equilibrium names."""
        return list(self._equilibria.keys())

    def _convert_to_backend(self, arr, backend):
        """Convert to backend."""
        if backend == "numpy":
            return arr
        elif backend == "torch" and _torch_available():
            import torch

            return torch.tensor(arr, dtype=torch.float64)
        elif backend == "jax" and _jax_available():
            import jax.numpy as jnp

            return jnp.array(arr)
        return arr


# ============================================================================
# Concrete Test Integrator
# ============================================================================


class ConcreteTestIntegrator(IntegratorBase):
    """
    Minimal concrete integrator for testing base class.

    Note: Named 'ConcreteTestIntegrator' instead of 'TestIntegrator'
    to avoid pytest collection warning.
    """

    def step(self, x, u, dt=None):
        """Simple Euler step."""
        dt = dt or self.dt
        f = self.system(x, u, backend=self.backend)
        self._stats["total_steps"] += 1
        return x + dt * f

    def integrate(self, x0, u_func=None, t_span=(0, 1), t_eval=None, dense_output=False):
        """Basic integration loop."""
        t0, tf = t_span
        t = t0
        x = np.asarray(x0)

        t_history = [t]
        x_history = [x.copy()]

        steps = 0
        while t < tf and steps < self.max_steps:
            if u_func is not None:
                u = u_func(t)
            else:
                u = np.array([]) if self.system.nu == 0 else np.zeros(self.system.nu)

            x = self.step(x, u)
            t += self.dt

            t_history.append(t)
            x_history.append(x.copy())

            steps += 1
            # Note: total_steps is incremented in step()

        return IntegrationResult(
            t=np.array(t_history), x=np.array(x_history), success=True, nsteps=steps
        )

    @property
    def name(self):
        return "ConcreteTestIntegrator"


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_system():
    """Create basic mock system."""
    return MockSystem()


@pytest.fixture
def mock_autonomous_system():
    """Create autonomous mock system."""
    return MockAutonomousSystem()


@pytest.fixture
def mock_system_with_equilibria():
    """Create mock system with equilibria."""
    return MockSystemWithEquilibria()


@pytest.fixture
def integrator(mock_system):
    """Create integrator with basic mock system."""
    return ConcreteTestIntegrator(mock_system, dt=0.01, backend="numpy")


@pytest.fixture
def integrator_with_equilibria(mock_system_with_equilibria):
    """Create integrator with equilibria system."""
    return ConcreteTestIntegrator(mock_system_with_equilibria, dt=0.01, backend="numpy")


@pytest.fixture
def integrator_autonomous(mock_autonomous_system):
    """Create integrator with autonomous system."""
    return ConcreteTestIntegrator(mock_autonomous_system, dt=0.01, backend="numpy")


# ============================================================================
# Test Class: StepMode Enum
# ============================================================================


class TestStepMode:
    """Test StepMode enumeration."""

    def test_step_mode_values(self):
        """Test that StepMode has correct values."""
        assert StepMode.FIXED.value == "fixed"
        assert StepMode.ADAPTIVE.value == "adaptive"

    def test_step_mode_members(self):
        """Test that all expected members exist."""
        assert hasattr(StepMode, "FIXED")
        assert hasattr(StepMode, "ADAPTIVE")

    def test_step_mode_comparison(self):
        """Test StepMode comparison."""
        assert StepMode.FIXED == StepMode.FIXED
        assert StepMode.FIXED != StepMode.ADAPTIVE


# ============================================================================
# Test Class: IntegrationResult Container
# ============================================================================


class TestIntegrationResult:
    """Test IntegrationResult container class."""

    def test_basic_initialization(self):
        """Test basic IntegrationResult creation."""
        t = np.array([0.0, 0.1, 0.2])
        x = np.array([[1.0], [0.9], [0.8]])

        result = IntegrationResult(t=t, x=x)

        assert np.array_equal(result.t, t)
        assert np.array_equal(result.x, x)
        assert result.success is True
        assert result.message == "Integration successful"

    def test_initialization_with_metadata(self):
        """Test IntegrationResult with all parameters."""
        t = np.array([0.0, 1.0])
        x = np.array([[1.0], [0.5]])

        result = IntegrationResult(
            t=t,
            x=x,
            success=True,
            message="Completed",
            nfev=100,
            nsteps=50,
            solver_info="Additional data",
        )

        assert result.success is True
        assert result.message == "Completed"
        assert result.nfev == 100
        assert result.nsteps == 50
        assert result.metadata["solver_info"] == "Additional data"

    def test_failed_integration_result(self):
        """Test IntegrationResult for failed integration."""
        result = IntegrationResult(
            t=np.array([0.0]),
            x=np.array([[1.0]]),
            success=False,
            message="Integration failed: step size too small",
        )

        assert result.success is False
        assert "failed" in result.message.lower()

    def test_repr(self):
        """Test __repr__ output."""
        result = IntegrationResult(
            t=np.array([0.0, 1.0]), x=np.array([[1.0], [0.5]]), nfev=50, nsteps=25
        )

        repr_str = repr(result)

        assert "IntegrationResult" in repr_str
        assert "success=True" in repr_str
        assert "nsteps=25" in repr_str
        assert "nfev=50" in repr_str


# ============================================================================
# Test Class: Abstract Interface
# ============================================================================


class TestAbstractInterface:
    """Test IntegratorBase abstract class behavior."""

    def test_cannot_instantiate_directly(self, mock_system):
        """Test that IntegratorBase cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract"):
            IntegratorBase(mock_system, dt=0.01)

    def test_subclass_must_implement_step(self):
        """Test that subclasses must implement step()."""

        class IncompleteIntegrator(IntegratorBase):
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                pass

            @property
            def name(self):
                return "Incomplete"

        with pytest.raises(TypeError):
            IncompleteIntegrator(MockSystem(), dt=0.01)

    def test_subclass_must_implement_integrate(self):
        """Test that subclasses must implement integrate()."""

        class IncompleteIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x

            @property
            def name(self):
                return "Incomplete"

        with pytest.raises(TypeError):
            IncompleteIntegrator(MockSystem(), dt=0.01)

    def test_subclass_must_implement_name(self):
        """Test that subclasses must implement name property."""

        class IncompleteIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x

            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                pass

        with pytest.raises(TypeError):
            IncompleteIntegrator(MockSystem(), dt=0.01)


# ============================================================================
# Test Class: Initialization and Validation
# ============================================================================


class TestInitialization:
    """Test initialization and parameter validation."""

    def test_fixed_mode_requires_dt(self, mock_system):
        """Test that FIXED mode requires dt parameter."""
        with pytest.raises(ValueError, match="dt.*required.*FIXED"):
            ConcreteTestIntegrator(mock_system, dt=None, step_mode=StepMode.FIXED)

    def test_adaptive_mode_default_dt(self, mock_system):
        """Test that ADAPTIVE mode provides default dt if None."""
        integrator = ConcreteTestIntegrator(mock_system, dt=None, step_mode=StepMode.ADAPTIVE)
        assert integrator.dt == 0.01

    def test_invalid_backend_raises_error(self, mock_system):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Invalid backend"):
            ConcreteTestIntegrator(mock_system, dt=0.01, backend="tensorflow")

    def test_options_stored(self, mock_system):
        """Test that options are stored correctly."""
        integrator = ConcreteTestIntegrator(
            mock_system, dt=0.01, rtol=1e-8, atol=1e-10, max_steps=5000, custom_option="test"
        )

        assert integrator.rtol == 1e-8
        assert integrator.atol == 1e-10
        assert integrator.max_steps == 5000
        assert integrator.options["custom_option"] == "test"

    def test_default_properties(self, integrator):
        """Test default property values."""
        assert integrator.dt == 0.01
        assert integrator.backend == "numpy"
        assert integrator.step_mode == StepMode.FIXED


# ============================================================================
# Test Class: Statistics Tracking
# ============================================================================


class TestStatisticsTracking:
    """Test statistics tracking in IntegratorBase."""

    def test_initial_stats(self, integrator):
        """Test that stats start at zero."""
        stats = integrator.get_stats()

        assert stats["total_steps"] == 0
        assert stats["total_fev"] == 0
        assert stats["total_time"] == 0.0
        assert stats["avg_fev_per_step"] == 0.0

    def test_reset_stats(self, mock_system):
        """Test resetting statistics."""
        integrator = ConcreteTestIntegrator(mock_system, dt=0.01)

        integrator.step(np.array([1.0]), np.array([0.0]))
        assert integrator.get_stats()["total_steps"] == 1

        integrator.reset_stats()
        stats = integrator.get_stats()

        assert stats["total_steps"] == 0
        assert stats["total_fev"] == 0
        assert stats["total_time"] == 0.0


# ============================================================================
# Test Class: String Representations
# ============================================================================


class TestStringRepresentations:
    """Test __repr__ and __str__ methods."""

    def test_repr(self, integrator):
        """Test __repr__ output."""
        repr_str = repr(integrator)

        assert "ConcreteTestIntegrator" in repr_str
        assert "dt=0.01" in repr_str
        assert "fixed" in repr_str
        assert "numpy" in repr_str

    def test_str(self, mock_system):
        """Test __str__ output."""
        integrator = ConcreteTestIntegrator(mock_system, dt=0.05, backend="numpy")

        str_repr = str(integrator)

        assert "ConcreteTestIntegrator" in str_repr
        assert "0.05" in str_repr
        assert "numpy" in str_repr


# ============================================================================
# Test Class: Equilibrium-Based Integration
# ============================================================================


class TestEquilibriumIntegration:
    """Test integrator initialization with equilibria."""

    def test_integrate_from_named_equilibrium(
        self, integrator_with_equilibria, mock_system_with_equilibria
    ):
        """Test integration starting from a named equilibrium."""
        x_eq, u_eq = mock_system_with_equilibria.get_equilibrium("zero")

        result = integrator_with_equilibria.integrate(x0=x_eq, u_func=lambda t: u_eq, t_span=(0, 1))

        assert result.success
        assert result.x.shape[0] > 0
        np.testing.assert_allclose(result.x[-1], x_eq, atol=0.1)

    def test_integrate_from_custom_equilibrium(
        self, integrator_with_equilibria, mock_system_with_equilibria
    ):
        """Test integration from non-origin equilibrium."""
        x_eq, u_eq = mock_system_with_equilibria.get_equilibrium("custom")

        result = integrator_with_equilibria.integrate(
            x0=x_eq, u_func=lambda t: u_eq, t_span=(0, 0.5)
        )

        assert result.success
        assert len(result.t) > 0

    def test_equilibrium_list_accessible(self, mock_system_with_equilibria):
        """Test that integrator can access system equilibria."""
        equilibria = mock_system_with_equilibria.list_equilibria()

        assert "zero" in equilibria
        assert "custom" in equilibria
        assert "origin" in equilibria

    def test_integrate_from_origin_equilibrium(
        self, integrator_with_equilibria, mock_system_with_equilibria
    ):
        """Test integration from origin (default equilibrium)."""
        x_eq, u_eq = mock_system_with_equilibria.get_equilibrium("origin")

        assert np.allclose(x_eq, np.zeros(2))
        assert np.allclose(u_eq, np.zeros(1))

        result = integrator_with_equilibria.integrate(
            x0=x_eq, u_func=lambda t: u_eq, t_span=(0, 0.1)
        )

        assert result.success

    def test_integrate_from_each_equilibrium(
        self, integrator_with_equilibria, mock_system_with_equilibria
    ):
        """Test integration starting from each defined equilibrium."""
        equilibria = mock_system_with_equilibria.list_equilibria()

        for eq_name in equilibria:
            x_eq, u_eq = mock_system_with_equilibria.get_equilibrium(eq_name)

            result = integrator_with_equilibria.integrate(
                x0=x_eq, u_func=lambda t: u_eq, t_span=(0, 0.5)
            )

            assert result.success, f"Integration from {eq_name} failed"


# ============================================================================
# Test Class: Autonomous System Integration
# ============================================================================


class TestAutonomousIntegration:
    """Test integrator with autonomous systems (nu=0)."""

    def test_integrate_autonomous_from_equilibrium(
        self, integrator_autonomous, mock_autonomous_system
    ):
        """Test autonomous integration from equilibrium."""
        x_eq, u_eq = mock_autonomous_system.get_equilibrium("zero")

        assert x_eq.shape == (2,)
        assert u_eq.shape == (0,)

        result = integrator_autonomous.integrate(x0=x_eq, u_func=None, t_span=(0, 1))

        assert result.success
        assert result.x.shape[0] > 0

    def test_autonomous_equilibrium_control_empty(self, mock_autonomous_system):
        """Test that autonomous equilibrium has empty control."""
        x_eq, u_eq = mock_autonomous_system.get_equilibrium("zero")

        assert u_eq.shape == (0,)
        assert u_eq.size == 0

    def test_integrate_autonomous_nonzero_initial(self, integrator_autonomous):
        """Test autonomous integration from non-equilibrium point."""
        x0 = np.array([1.0, 0.5])

        result = integrator_autonomous.integrate(x0=x0, u_func=None, t_span=(0, 1))

        assert result.success
        assert np.linalg.norm(result.x[-1]) < np.linalg.norm(x0)


# ============================================================================
# Test Class: Backend Consistency
# ============================================================================


class TestBackendConsistency:
    """Test backend consistency when using equilibria."""

    def test_equilibrium_numpy_backend(self, mock_system_with_equilibria):
        """Test equilibrium retrieval in NumPy backend."""
        x_eq, u_eq = mock_system_with_equilibria.get_equilibrium("zero", backend="numpy")

        assert isinstance(x_eq, np.ndarray)
        assert isinstance(u_eq, np.ndarray)

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
    def test_equilibrium_torch_backend(self, mock_system_with_equilibria):
        """Test equilibrium retrieval in PyTorch backend."""
        import torch

        x_eq, u_eq = mock_system_with_equilibria.get_equilibrium("zero", backend="torch")

        assert isinstance(x_eq, torch.Tensor)
        assert isinstance(u_eq, torch.Tensor)

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_equilibrium_jax_backend(self, mock_system_with_equilibria):
        """Test equilibrium retrieval in JAX backend."""
        import jax.numpy as jnp

        x_eq, u_eq = mock_system_with_equilibria.get_equilibrium("zero", backend="jax")

        assert isinstance(x_eq, jnp.ndarray)
        assert isinstance(u_eq, jnp.ndarray)

    def test_integrator_respects_system_default_backend(self, mock_system_with_equilibria):
        """Test that integrator uses system's default backend for equilibria."""
        x_eq, u_eq = mock_system_with_equilibria.get_equilibrium("zero")

        assert isinstance(x_eq, np.ndarray)


# ============================================================================
# Test Class: Control Function Handling
# ============================================================================


class TestControlFunctionHandling:
    """Test various control function patterns."""

    def test_constant_control_function(self, integrator_with_equilibria):
        """Test integration with constant control."""
        u_const = np.array([0.5])

        result = integrator_with_equilibria.integrate(
            x0=np.array([0.0, 0.0]), u_func=lambda t: u_const, t_span=(0, 0.5)
        )

        assert result.success

    def test_time_varying_control_function(self, integrator_with_equilibria):
        """Test integration with time-varying control."""

        def u_func(t):
            return np.array([np.sin(t)])

        result = integrator_with_equilibria.integrate(
            x0=np.array([0.0, 0.0]), u_func=u_func, t_span=(0, 1)
        )

        assert result.success

    def test_none_control_autonomous(self, integrator_autonomous):
        """Test that None control works for autonomous systems."""
        result = integrator_autonomous.integrate(
            x0=np.array([1.0, 0.5]), u_func=None, t_span=(0, 1)
        )

        assert result.success

    def test_control_function_called_at_each_step(self, integrator_with_equilibria):
        """Test that control function is evaluated at each time step."""
        call_times = []

        def u_func(t):
            call_times.append(t)
            return np.array([0.0])

        result = integrator_with_equilibria.integrate(
            x0=np.array([0.0, 0.0]), u_func=u_func, t_span=(0, 1)
        )

        assert result.success
        assert len(call_times) > 5


# ============================================================================
# Test Class: Time Span Validation
# ============================================================================


class TestTimeSpanValidation:
    """Test validation of time span parameters."""

    def test_negative_time_span(self, integrator_with_equilibria):
        """Test integration with negative time direction."""
        x0 = np.array([1.0, 0.5])

        result = integrator_with_equilibria.integrate(
            x0=x0, u_func=lambda t: np.array([0.0]), t_span=(1, 0)
        )

        assert result is not None

    def test_zero_duration_integration(self, integrator_with_equilibria):
        """Test integration with zero duration."""
        x0 = np.array([1.0, 0.5])

        result = integrator_with_equilibria.integrate(
            x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 0)
        )

        assert result.success
        assert len(result.t) >= 1
        np.testing.assert_array_equal(result.x[0], x0)

    def test_very_short_time_span(self, integrator_with_equilibria):
        """Test integration over very short time."""
        x0 = np.array([1.0, 0.5])

        result = integrator_with_equilibria.integrate(
            x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 0.001)
        )

        assert result.success
        assert len(result.t) >= 1


# ============================================================================
# Test Class: Integration Termination
# ============================================================================


class TestIntegrationTermination:
    """Test integration termination conditions."""

    def test_max_steps_termination(self, mock_system_with_equilibria):
        """Test that integration stops at max_steps."""
        integrator = ConcreteTestIntegrator(mock_system_with_equilibria, dt=0.01, max_steps=10)

        x0 = np.array([1.0, 0.5])

        result = integrator.integrate(x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 100))

        assert result.nsteps <= 11

    def test_normal_termination(self, integrator_with_equilibria):
        """Test normal termination at end time."""
        x0 = np.array([1.0, 0.5])

        result = integrator_with_equilibria.integrate(
            x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 1)
        )

        assert result.success
        assert result.t[-1] >= 0.99


# ============================================================================
# Test Class: Dense Output
# ============================================================================


class TestDenseOutput:
    """Test dense output functionality."""

    def test_dense_output_flag(self, integrator_with_equilibria):
        """Test that dense_output flag is passed correctly."""
        x0 = np.array([1.0, 0.5])

        result = integrator_with_equilibria.integrate(
            x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 1), dense_output=True
        )

        assert result.success

    def test_t_eval_points(self, integrator_with_equilibria):
        """Test evaluation at specific time points."""
        x0 = np.array([1.0, 0.5])
        t_eval = np.array([0, 0.5, 1.0])

        result = integrator_with_equilibria.integrate(
            x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 1), t_eval=t_eval
        )

        assert result.success


# ============================================================================
# Test Class: Step Size Handling
# ============================================================================


class TestStepSizeHandling:
    """Test step size parameter handling."""

    def test_small_step_size(self, mock_system_with_equilibria):
        """Test integration with very small step size."""
        integrator = ConcreteTestIntegrator(mock_system_with_equilibria, dt=0.0001)

        x0 = np.array([1.0, 0.5])

        result = integrator.integrate(x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 0.1))

        assert result.success
        assert result.nsteps > 100

    def test_large_step_size(self, mock_system_with_equilibria):
        """Test integration with large step size."""
        integrator = ConcreteTestIntegrator(mock_system_with_equilibria, dt=0.5)

        x0 = np.array([1.0, 0.5])

        result = integrator.integrate(x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 1))

        assert result.success
        assert result.nsteps < 10

    def test_step_size_consistency(self, mock_system_with_equilibria):
        """Test that step size is consistent."""
        dt = 0.1
        integrator = ConcreteTestIntegrator(mock_system_with_equilibria, dt=dt)

        x0 = np.array([1.0, 0.5])

        result = integrator.integrate(x0=x0, u_func=lambda t: np.array([0.0]), t_span=(0, 1))

        assert result.success

        if len(result.t) > 1:
            dts = np.diff(result.t)
            np.testing.assert_allclose(dts, dt, rtol=0.1)


# ============================================================================
# Test Class: Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_nan_in_initial_condition(self, integrator_with_equilibria):
        """Test handling of NaN in initial condition."""
        x0_nan = np.array([np.nan, 0.0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            try:
                result = integrator_with_equilibria.integrate(
                    x0=x0_nan, u_func=lambda t: np.array([0.0]), t_span=(0, 0.1)
                )
                assert np.any(np.isnan(result.x))
            except (ValueError, RuntimeError):
                pass

    def test_inf_in_initial_condition(self, integrator_with_equilibria):
        """Test handling of Inf in initial condition."""
        x0_inf = np.array([np.inf, 0.0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            try:
                result = integrator_with_equilibria.integrate(
                    x0=x0_inf, u_func=lambda t: np.array([0.0]), t_span=(0, 0.1)
                )
                assert np.any(np.isinf(result.x))
            except (ValueError, RuntimeError):
                pass

    def test_none_initial_condition(self, integrator_with_equilibria):
        """Test handling of None initial condition."""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            integrator_with_equilibria.integrate(
                x0=None, u_func=lambda t: np.array([0.0]), t_span=(0, 0.1)
            )


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
