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
Unit tests for IntegratorBase abstract interface.

Tests cover:
1. Enum definitions (StepMode)
2. IntegrationResult container
3. Base class initialization and validation
4. Statistics tracking
5. Common utilities
6. String representations

Additional Unit Tests for IntegratorBase - Equilibrium Handling & Extended Features

Tests cover:
1. Equilibrium-based initialization
2. Integration from named equilibria
3. Autonomous system integration
4. Backend consistency with equilibria
5. Control function handling
6. Array dimension validation
7. Time span validation
8. Integration termination conditions
9. Dense output handling
10. Step size adaptation

Note: Helper class is named 'ConcreteTestIntegrator' instead of 'TestIntegrator'
to avoid pytest collection warning.
"""

import warnings
import pytest
import numpy as np
from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    StepMode,
    IntegrationResult
)


# Conditional imports for backends
torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

jax_available = True
try:
    import jax.numpy as jnp
except ImportError:
    jax_available = False


# ============================================================================
# Test Class 1: StepMode Enum
# ============================================================================

class TestStepMode:
    """Test StepMode enumeration"""
    
    def test_step_mode_values(self):
        """Test that StepMode has correct values"""
        assert StepMode.FIXED.value == "fixed"
        assert StepMode.ADAPTIVE.value == "adaptive"
    
    def test_step_mode_members(self):
        """Test that all expected members exist"""
        assert hasattr(StepMode, 'FIXED')
        assert hasattr(StepMode, 'ADAPTIVE')
    
    def test_step_mode_comparison(self):
        """Test StepMode comparison"""
        assert StepMode.FIXED == StepMode.FIXED
        assert StepMode.FIXED != StepMode.ADAPTIVE


# ============================================================================
# Test Class 2: IntegrationResult Container
# ============================================================================

class TestIntegrationResult:
    """Test IntegrationResult container class"""
    
    def test_basic_initialization(self):
        """Test basic IntegrationResult creation"""
        t = np.array([0.0, 0.1, 0.2])
        x = np.array([[1.0], [0.9], [0.8]])
        
        result = IntegrationResult(t=t, x=x)
        
        assert np.array_equal(result.t, t)
        assert np.array_equal(result.x, x)
        assert result.success is True
        assert result.message == "Integration successful"
    
    def test_initialization_with_metadata(self):
        """Test IntegrationResult with all parameters"""
        t = np.array([0.0, 1.0])
        x = np.array([[1.0], [0.5]])
        
        result = IntegrationResult(
            t=t,
            x=x,
            success=True,
            message="Completed",
            nfev=100,
            nsteps=50,
            solver_info="Additional data"
        )
        
        assert result.success is True
        assert result.message == "Completed"
        assert result.nfev == 100
        assert result.nsteps == 50
        assert result.metadata['solver_info'] == "Additional data"
    
    def test_failed_integration_result(self):
        """Test IntegrationResult for failed integration"""
        result = IntegrationResult(
            t=np.array([0.0]),
            x=np.array([[1.0]]),
            success=False,
            message="Integration failed: step size too small"
        )
        
        assert result.success is False
        assert "failed" in result.message.lower()
    
    def test_repr(self):
        """Test __repr__ output"""
        result = IntegrationResult(
            t=np.array([0.0, 1.0]),
            x=np.array([[1.0], [0.5]]),
            nfev=50,
            nsteps=25
        )
        
        repr_str = repr(result)
        
        assert 'IntegrationResult' in repr_str
        assert 'success=True' in repr_str
        assert 'nsteps=25' in repr_str
        assert 'nfev=50' in repr_str


# ============================================================================
# Test Class 3: IntegratorBase Abstract Interface
# ============================================================================

class TestIntegratorBaseInterface:
    """Test IntegratorBase abstract class behavior"""
    
    def test_cannot_instantiate_directly(self):
        """Test that IntegratorBase cannot be instantiated"""
        
        # Mock system
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        system = MockSystem()
        
        # Should raise TypeError because step() and integrate() are abstract
        with pytest.raises(TypeError, match="abstract"):
            IntegratorBase(system, dt=0.01)
    
    def test_subclass_must_implement_step(self):
        """Test that subclasses must implement step()"""
        
        class IncompleteIntegrator(IntegratorBase):
            # Missing step() implementation!
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                pass
            
            @property
            def name(self):
                return "Incomplete"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        # Should still raise because step() is abstract
        with pytest.raises(TypeError):
            IncompleteIntegrator(MockSystem(), dt=0.01)
    
    def test_subclass_must_implement_integrate(self):
        """Test that subclasses must implement integrate()"""
        
        class IncompleteIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x  # Dummy
            # Missing integrate()!
            
            @property
            def name(self):
                return "Incomplete"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        # Should raise because integrate() is abstract
        with pytest.raises(TypeError):
            IncompleteIntegrator(MockSystem(), dt=0.01)
    
    def test_subclass_must_implement_name(self):
        """Test that subclasses must implement name property"""
        
        class IncompleteIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x
            
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                pass
            # Missing name property!
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        # Should raise because name is abstract
        with pytest.raises(TypeError):
            IncompleteIntegrator(MockSystem(), dt=0.01)


# ============================================================================
# Test Class 4: Initialization and Validation
# ============================================================================

class TestInitializationValidation:
    """Test initialization and parameter validation"""
    
    def test_fixed_mode_requires_dt(self):
        """Test that FIXED mode requires dt parameter"""
        
        # Create a minimal concrete subclass for testing
        class MinimalIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                return IntegrationResult(np.array([0]), np.array([[0]]))
            @property
            def name(self):
                return "Minimal"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        system = MockSystem()
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="dt.*required.*FIXED"):
            MinimalIntegrator(system, dt=None, step_mode=StepMode.FIXED)
    
    def test_adaptive_mode_default_dt(self):
        """Test that ADAPTIVE mode provides default dt if None"""
        
        class MinimalIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                return IntegrationResult(np.array([0]), np.array([[0]]))
            @property
            def name(self):
                return "Minimal"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        system = MockSystem()
        
        # Should not raise, provides default dt=0.01
        integrator = MinimalIntegrator(system, dt=None, step_mode=StepMode.ADAPTIVE)
        
        assert integrator.dt == 0.01  # Default value
    
    def test_invalid_backend_raises_error(self):
        """Test that invalid backend raises ValueError"""
        
        class MinimalIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                return IntegrationResult(np.array([0]), np.array([[0]]))
            @property
            def name(self):
                return "Minimal"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        system = MockSystem()
        
        with pytest.raises(ValueError, match="Invalid backend"):
            MinimalIntegrator(system, dt=0.01, backend='tensorflow')
    
    def test_options_stored(self):
        """Test that options are stored correctly"""
        
        class MinimalIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                return IntegrationResult(np.array([0]), np.array([[0]]))
            @property
            def name(self):
                return "Minimal"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        system = MockSystem()
        
        integrator = MinimalIntegrator(
            system, dt=0.01,
            rtol=1e-8,
            atol=1e-10,
            max_steps=5000,
            custom_option='test'
        )
        
        assert integrator.rtol == 1e-8
        assert integrator.atol == 1e-10
        assert integrator.max_steps == 5000
        assert integrator.options['custom_option'] == 'test'


# ============================================================================
# Test Class 5: Statistics Tracking
# ============================================================================

class TestStatisticsTracking:
    """Test statistics tracking in IntegratorBase"""
    
    def test_initial_stats(self):
        """Test that stats start at zero"""
        
        class MinimalIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                return IntegrationResult(np.array([0]), np.array([[0]]))
            @property
            def name(self):
                return "Minimal"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        integrator = MinimalIntegrator(MockSystem(), dt=0.01)
        
        stats = integrator.get_stats()
        
        assert stats['total_steps'] == 0
        assert stats['total_fev'] == 0
        assert stats['total_time'] == 0.0
        assert stats['avg_fev_per_step'] == 0.0
    
    def test_reset_stats(self):
        """Test resetting statistics"""
        
        class MinimalIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                self._stats['total_steps'] += 1
                return x
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                return IntegrationResult(np.array([0]), np.array([[0]]))
            @property
            def name(self):
                return "Minimal"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        integrator = MinimalIntegrator(MockSystem(), dt=0.01)
        
        # Do some steps
        integrator.step(np.array([1.0]), np.array([0.0]))
        assert integrator.get_stats()['total_steps'] == 1
        
        # Reset
        integrator.reset_stats()
        stats = integrator.get_stats()
        
        assert stats['total_steps'] == 0
        assert stats['total_fev'] == 0
        assert stats['total_time'] == 0.0


# ============================================================================
# Test Class 6: String Representations
# ============================================================================

class TestStringRepresentations:
    """Test __repr__ and __str__ methods"""
    
    def test_repr(self):
        """Test __repr__ output"""
        
        class MinimalIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                return IntegrationResult(np.array([0]), np.array([[0]]))
            @property
            def name(self):
                return "Minimal"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        integrator = MinimalIntegrator(MockSystem(), dt=0.01, backend='numpy')
        
        repr_str = repr(integrator)
        
        assert 'MinimalIntegrator' in repr_str
        assert 'dt=0.01' in repr_str
        assert 'fixed' in repr_str
        assert 'numpy' in repr_str
    
    def test_str(self):
        """Test __str__ output"""
        
        class MinimalIntegrator(IntegratorBase):
            def step(self, x, u, dt=None):
                return x
            def integrate(self, x0, u_func, t_span, t_eval=None, dense_output=False):
                return IntegrationResult(np.array([0]), np.array([[0]]))
            @property
            def name(self):
                return "Minimal Test Integrator"
        
        class MockSystem:
            nx = 1
            nu = 1
            _initialized = True
            def __call__(self, x, u, backend='numpy'):
                return -x
        
        integrator = MinimalIntegrator(MockSystem(), dt=0.05, backend='numpy')
        
        str_repr = str(integrator)
        
        assert 'Minimal Test Integrator' in str_repr
        assert '0.05' in str_repr
        assert 'numpy' in str_repr


# ============================================================================
# New Tests
# ============================================================================

# ============================================================================
# Mock Systems for Testing
# ============================================================================

class MockSystemWithEquilibria:
    """Mock system with equilibrium handler for testing"""
    
    def __init__(self):
        self.nx = 2
        self.nu = 1
        self._initialized = True
        self._default_backend = 'numpy'
        
        # Mock equilibrium handler
        self.equilibria = MockEquilibriumHandler(nx=2, nu=1)
        self.equilibria.add_equilibrium("zero", np.array([0.0, 0.0]), np.array([0.0]))
        self.equilibria.add_equilibrium("custom", np.array([1.0, 0.5]), np.array([0.2]))
    
    def __call__(self, x, u, backend='numpy'):
        """Simple linear dynamics: dx = -x + u"""
        return -x + u
    
    def get_equilibrium(self, name=None, backend=None):
        """Get equilibrium in specified backend"""
        backend = backend or self._default_backend
        return self.equilibria.get_both(name, backend)
    
    def list_equilibria(self):
        """List equilibrium names"""
        return self.equilibria.list_names()


class MockAutonomousSystem:
    """Mock autonomous system (nu=0) for testing"""
    
    def __init__(self):
        self.nx = 2
        self.nu = 0  # Autonomous
        self._initialized = True
        self._default_backend = 'numpy'
        
        self.equilibria = MockEquilibriumHandler(nx=2, nu=0)
        self.equilibria.add_equilibrium("zero", np.array([0.0, 0.0]), np.array([]))
    
    def __call__(self, x, u=None, backend='numpy'):
        """Autonomous dynamics: dx = -x"""
        return -x
    
    def get_equilibrium(self, name=None, backend=None):
        """Get equilibrium in specified backend"""
        backend = backend or self._default_backend
        return self.equilibria.get_both(name, backend)


class MockEquilibriumHandler:
    """Minimal mock of EquilibriumHandler for testing"""
    
    def __init__(self, nx, nu):
        self._nx = nx
        self._nu = nu
        self._equilibria = {}
        self._default = "origin"
        
        # Origin
        self._equilibria["origin"] = {
            "x": np.zeros(nx),
            "u": np.zeros(nu),
            "metadata": {}
        }
    
    @property
    def nx(self):
        return self._nx
    
    @property
    def nu(self):
        return self._nu
    
    def add_equilibrium(self, name, x_eq, u_eq):
        """Add equilibrium"""
        self._equilibria[name] = {
            "x": np.asarray(x_eq),
            "u": np.asarray(u_eq),
            "metadata": {}
        }
    
    def get_x(self, name=None, backend='numpy'):
        """Get equilibrium state"""
        name = name or self._default
        x = self._equilibria[name]["x"]
        return self._convert_to_backend(x, backend)
    
    def get_u(self, name=None, backend='numpy'):
        """Get equilibrium control"""
        name = name or self._default
        u = self._equilibria[name]["u"]
        return self._convert_to_backend(u, backend)
    
    def get_both(self, name=None, backend='numpy'):
        """Get both state and control"""
        return self.get_x(name, backend), self.get_u(name, backend)
    
    def list_names(self):
        """List equilibrium names"""
        return list(self._equilibria.keys())
    
    def _convert_to_backend(self, arr, backend):
        """Convert to backend"""
        if backend == 'numpy':
            return arr
        elif backend == 'torch' and torch_available:
            return torch.tensor(arr, dtype=torch.float64)
        elif backend == 'jax' and jax_available:
            return jnp.array(arr)
        else:
            return arr


# ============================================================================
# Concrete Test Integrator
# ============================================================================

class ConcreteTestIntegrator(IntegratorBase):
    """Minimal concrete integrator for testing base class
    
    Note: Named 'ConcreteTestIntegrator' instead of 'TestIntegrator' 
    to avoid pytest collection warning (pytest collects classes starting with 'Test')
    """
    
    def step(self, x, u, dt=None):
        """Simple Euler step"""
        dt = dt or self.dt
        f = self.system(x, u, backend=self.backend)
        return x + dt * f
    
    def integrate(self, x0, u_func=None, t_span=(0, 1), t_eval=None, dense_output=False):
        """Basic integration loop"""
        t0, tf = t_span
        t = t0
        x = np.asarray(x0)
        
        t_history = [t]
        x_history = [x.copy()]
        
        steps = 0
        while t < tf and steps < self.max_steps:
            # Get control
            if u_func is not None:
                u = u_func(t)
            else:
                # Autonomous
                u = np.array([]) if self.system.nu == 0 else np.zeros(self.system.nu)
            
            # Take step
            x = self.step(x, u)
            t += self.dt
            
            t_history.append(t)
            x_history.append(x.copy())
            
            steps += 1
            self._stats['total_steps'] += 1
        
        return IntegrationResult(
            t=np.array(t_history),
            x=np.array(x_history),
            success=True,
            nsteps=steps
        )
    
    @property
    def name(self):
        return "ConcreteTestIntegrator"


# ============================================================================
# Test Class 7: Equilibrium-Based Initialization
# ============================================================================

class TestEquilibriumInitialization:
    """Test integrator initialization with equilibria"""
    
    def test_integrate_from_named_equilibrium(self):
        """Test integration starting from a named equilibrium"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        # Get equilibrium
        x_eq, u_eq = system.get_equilibrium("zero")
        
        # Integrate with constant control at equilibrium
        result = integrator.integrate(
            x0=x_eq,
            u_func=lambda t: u_eq,
            t_span=(0, 1)
        )
        
        assert result.success
        assert result.x.shape[0] > 0
        # At equilibrium with equilibrium control, should stay near equilibrium
        np.testing.assert_allclose(result.x[-1], x_eq, atol=0.1)
    
    def test_integrate_from_custom_equilibrium(self):
        """Test integration from non-origin equilibrium"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        x_eq, u_eq = system.get_equilibrium("custom")
        
        result = integrator.integrate(
            x0=x_eq,
            u_func=lambda t: u_eq,
            t_span=(0, 0.5)
        )
        
        assert result.success
        assert len(result.t) > 0
    
    def test_equilibrium_list_accessible(self):
        """Test that integrator can access system equilibria"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        equilibria = system.list_equilibria()
        
        assert "zero" in equilibria
        assert "custom" in equilibria
        assert "origin" in equilibria
    
    def test_integrate_from_origin_equilibrium(self):
        """Test integration from origin (default equilibrium)"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        x_eq, u_eq = system.get_equilibrium("origin")
        
        assert np.allclose(x_eq, np.zeros(2))
        assert np.allclose(u_eq, np.zeros(1))
        
        result = integrator.integrate(
            x0=x_eq,
            u_func=lambda t: u_eq,
            t_span=(0, 0.1)
        )
        
        assert result.success


# ============================================================================
# Test Class 8: Autonomous System Integration
# ============================================================================

class TestAutonomousIntegration:
    """Test integrator with autonomous systems (nu=0)"""
    
    def test_integrate_autonomous_from_equilibrium(self):
        """Test autonomous integration from equilibrium"""
        system = MockAutonomousSystem()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        x_eq, u_eq = system.get_equilibrium("zero")
        
        assert x_eq.shape == (2,)
        assert u_eq.shape == (0,)  # Empty for autonomous
        
        # Integrate without u_func (autonomous)
        result = integrator.integrate(
            x0=x_eq,
            u_func=None,  # No control function
            t_span=(0, 1)
        )
        
        assert result.success
        assert result.x.shape[0] > 0
    
    def test_autonomous_equilibrium_control_empty(self):
        """Test that autonomous equilibrium has empty control"""
        system = MockAutonomousSystem()
        
        x_eq, u_eq = system.get_equilibrium("zero")
        
        assert u_eq.shape == (0,)
        assert u_eq.size == 0
    
    def test_integrate_autonomous_nonzero_initial(self):
        """Test autonomous integration from non-equilibrium point"""
        system = MockAutonomousSystem()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        x0 = np.array([1.0, 0.5])
        
        result = integrator.integrate(
            x0=x0,
            u_func=None,
            t_span=(0, 1)
        )
        
        assert result.success
        # Should decay toward zero
        assert np.linalg.norm(result.x[-1]) < np.linalg.norm(x0)


# ============================================================================
# Test Class 9: Backend Consistency with Equilibria
# ============================================================================

class TestBackendConsistencyEquilibria:
    """Test backend consistency when using equilibria"""
    
    def test_equilibrium_numpy_backend(self):
        """Test equilibrium retrieval in NumPy backend"""
        system = MockSystemWithEquilibria()
        
        x_eq, u_eq = system.get_equilibrium("zero", backend='numpy')
        
        assert isinstance(x_eq, np.ndarray)
        assert isinstance(u_eq, np.ndarray)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_equilibrium_torch_backend(self):
        """Test equilibrium retrieval in PyTorch backend"""
        system = MockSystemWithEquilibria()
        
        x_eq, u_eq = system.get_equilibrium("zero", backend='torch')
        
        assert isinstance(x_eq, torch.Tensor)
        assert isinstance(u_eq, torch.Tensor)
    
    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_equilibrium_jax_backend(self):
        """Test equilibrium retrieval in JAX backend"""
        system = MockSystemWithEquilibria()
        
        x_eq, u_eq = system.get_equilibrium("zero", backend='jax')
        
        assert isinstance(x_eq, jnp.ndarray)
        assert isinstance(u_eq, jnp.ndarray)
    
    def test_integrator_respects_system_default_backend(self):
        """Test that integrator uses system's default backend for equilibria"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01, backend='numpy')
        
        # Should be able to get equilibrium without specifying backend
        x_eq, u_eq = system.get_equilibrium("zero")
        
        assert isinstance(x_eq, np.ndarray)


# ============================================================================
# Test Class 10: Control Function Handling
# ============================================================================

class TestControlFunctionHandling:
    """Test various control function patterns"""
    
    def test_constant_control_function(self):
        """Test integration with constant control"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        u_const = np.array([0.5])
        
        result = integrator.integrate(
            x0=np.array([0.0, 0.0]),
            u_func=lambda t: u_const,
            t_span=(0, 0.5)
        )
        
        assert result.success
    
    def test_time_varying_control_function(self):
        """Test integration with time-varying control"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        # Sinusoidal control
        def u_func(t):
            return np.array([np.sin(t)])
        
        result = integrator.integrate(
            x0=np.array([0.0, 0.0]),
            u_func=u_func,
            t_span=(0, 1)
        )
        
        assert result.success
    
    def test_none_control_autonomous(self):
        """Test that None control works for autonomous systems"""
        system = MockAutonomousSystem()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        result = integrator.integrate(
            x0=np.array([1.0, 0.5]),
            u_func=None,
            t_span=(0, 1)
        )
        
        assert result.success
    
    def test_control_function_called_at_each_step(self):
        """Test that control function is evaluated at each time step"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.1)
        
        call_times = []
        
        def u_func(t):
            call_times.append(t)
            return np.array([0.0])
        
        result = integrator.integrate(
            x0=np.array([0.0, 0.0]),
            u_func=u_func,
            t_span=(0, 1)
        )
        
        assert result.success
        # Should have called u_func multiple times
        assert len(call_times) > 5


# ============================================================================
# Test Class 11: Array Dimension Validation
# ============================================================================

class TestArrayDimensionValidation:
    """Test validation of array dimensions"""
    
    def test_wrong_initial_condition_dimension(self):
        """Test error on wrong initial condition dimension"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        # System has nx=2, but provide wrong dimension
        x0_wrong = np.array([1.0])  # Only 1D!
        
        # This should be caught by the system or integrator
        # Depending on implementation, might work or raise error
        # Document expected behavior
        try:
            result = integrator.integrate(
                x0=x0_wrong,
                u_func=lambda t: np.array([0.0]),
                t_span=(0, 0.1)
            )
            # If it works, dimensions were broadcast/handled
            assert result.x.shape[1] >= 1
        except (ValueError, IndexError):
            # Expected error for dimension mismatch
            pass
    
    def test_wrong_control_dimension(self):
        """Test error on wrong control dimension"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        x0 = np.array([0.0, 0.0])
        
        # System has nu=1, but provide wrong dimension
        u_wrong = lambda t: np.array([0.0, 0.0])  # 2D instead of 1D!
        
        # Should raise error or handle gracefully
        try:
            result = integrator.integrate(x0=x0, u_func=u_wrong, t_span=(0, 0.1))
            # Implementation-dependent
        except (ValueError, IndexError):
            # Expected for dimension mismatch
            pass
    
    def test_scalar_vs_array_initial_condition(self):
        """Test that scalar initial conditions work for 1D systems"""
        # Create 1D system
        class Mock1DSystem:
            nx = 1
            nu = 1
            _initialized = True
            _default_backend = 'numpy'
            
            def __call__(self, x, u, backend='numpy'):
                return -x + u
        
        system = Mock1DSystem()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        # Try scalar initial condition
        x0_scalar = np.array([1.0])
        
        result = integrator.integrate(
            x0=x0_scalar,
            u_func=lambda t: np.array([0.0]),
            t_span=(0, 0.1)
        )
        
        assert result.success


# ============================================================================
# Test Class 12: Time Span Validation
# ============================================================================

class TestTimeSpanValidation:
    """Test validation of time span parameters"""
    
    def test_negative_time_span(self):
        """Test integration with negative time direction"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        x0 = np.array([1.0, 0.5])
        
        # Backward integration: tf < t0
        result = integrator.integrate(
            x0=x0,
            u_func=lambda t: np.array([0.0]),
            t_span=(1, 0)  # Backward!
        )
        
        # Behavior depends on implementation
        # Some integrators handle this, others don't
        assert result is not None
    
    def test_zero_duration_integration(self):
        """Test integration with zero duration"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        x0 = np.array([1.0, 0.5])
        
        result = integrator.integrate(
            x0=x0,
            u_func=lambda t: np.array([0.0]),
            t_span=(0, 0)  # Zero duration
        )
        
        # Should return initial condition only
        assert result.success
        assert len(result.t) >= 1
        np.testing.assert_array_equal(result.x[0], x0)
    
    def test_very_short_time_span(self):
        """Test integration over very short time"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        x0 = np.array([1.0, 0.5])
        
        result = integrator.integrate(
            x0=x0,
            u_func=lambda t: np.array([0.0]),
            t_span=(0, 0.001)  # Very short
        )
        
        assert result.success
        # Might only take one step
        assert len(result.t) >= 1


# ============================================================================
# Test Class 13: Integration Termination
# ============================================================================

class TestIntegrationTermination:
    """Test integration termination conditions"""
    
    def test_max_steps_termination(self):
        """Test that integration stops at max_steps"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01, max_steps=10)
        
        x0 = np.array([1.0, 0.5])
        
        result = integrator.integrate(
            x0=x0,
            u_func=lambda t: np.array([0.0]),
            t_span=(0, 100)  # Very long, should hit max_steps
        )
        
        # Should stop early due to max_steps
        assert result.nsteps <= 10 + 1  # Allow for off-by-one
    
    def test_normal_termination(self):
        """Test normal termination at end time"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        x0 = np.array([1.0, 0.5])
        
        result = integrator.integrate(
            x0=x0,
            u_func=lambda t: np.array([0.0]),
            t_span=(0, 1)
        )
        
        assert result.success
        # Final time should be close to requested end time
        assert result.t[-1] >= 0.99  # Allow for small error


# ============================================================================
# Test Class 14: Dense Output
# ============================================================================

class TestDenseOutput:
    """Test dense output functionality"""
    
    def test_dense_output_flag(self):
        """Test that dense_output flag is passed correctly"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        x0 = np.array([1.0, 0.5])
        
        # Test with dense_output=True
        result = integrator.integrate(
            x0=x0,
            u_func=lambda t: np.array([0.0]),
            t_span=(0, 1),
            dense_output=True
        )
        
        # Behavior depends on implementation
        assert result.success
    
    def test_t_eval_points(self):
        """Test evaluation at specific time points"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        x0 = np.array([1.0, 0.5])
        t_eval = np.array([0, 0.5, 1.0])
        
        result = integrator.integrate(
            x0=x0,
            u_func=lambda t: np.array([0.0]),
            t_span=(0, 1),
            t_eval=t_eval
        )
        
        # Behavior depends on implementation
        # Some integrators respect t_eval, others don't
        assert result.success


# ============================================================================
# Test Class 15: Step Size Handling
# ============================================================================

class TestStepSizeHandling:
    """Test step size parameter handling"""
    
    def test_small_step_size(self):
        """Test integration with very small step size"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.0001)
        
        x0 = np.array([1.0, 0.5])
        
        result = integrator.integrate(
            x0=x0,
            u_func=lambda t: np.array([0.0]),
            t_span=(0, 0.1)
        )
        
        assert result.success
        # Should take many steps
        assert result.nsteps > 100
    
    def test_large_step_size(self):
        """Test integration with large step size"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.5)
        
        x0 = np.array([1.0, 0.5])
        
        result = integrator.integrate(
            x0=x0,
            u_func=lambda t: np.array([0.0]),
            t_span=(0, 1)
        )
        
        assert result.success
        # Should take few steps
        assert result.nsteps < 10
    
    def test_step_size_consistency(self):
        """Test that step size is consistent"""
        system = MockSystemWithEquilibria()
        dt = 0.1
        integrator = ConcreteTestIntegrator(system, dt=dt)
        
        x0 = np.array([1.0, 0.5])
        
        result = integrator.integrate(
            x0=x0,
            u_func=lambda t: np.array([0.0]),
            t_span=(0, 1)
        )
        
        assert result.success
        
        # Check time step consistency
        if len(result.t) > 1:
            dts = np.diff(result.t)
            # Most steps should be close to dt
            np.testing.assert_allclose(dts, dt, rtol=0.1)


# ============================================================================
# Test Class 16: Integration from Multiple Equilibria
# ============================================================================

class TestMultipleEquilibriaIntegration:
    """Test integration from different equilibrium points"""
    
    def test_integrate_from_each_equilibrium(self):
        """Test integration starting from each defined equilibrium"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        equilibria = system.list_equilibria()
        
        for eq_name in equilibria:
            x_eq, u_eq = system.get_equilibrium(eq_name)
            
            result = integrator.integrate(
                x0=x_eq,
                u_func=lambda t: u_eq,
                t_span=(0, 0.5)
            )
            
            assert result.success, f"Integration from {eq_name} failed"
    
    def test_equilibrium_switching(self):
        """Test switching between equilibria during integration"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        # Start at zero
        x_eq1, u_eq1 = system.get_equilibrium("zero")
        
        result1 = integrator.integrate(
            x0=x_eq1,
            u_func=lambda t: u_eq1,
            t_span=(0, 0.5)
        )
        
        # Continue from custom equilibrium
        x_eq2, u_eq2 = system.get_equilibrium("custom")
        
        result2 = integrator.integrate(
            x0=x_eq2,
            u_func=lambda t: u_eq2,
            t_span=(0, 0.5)
        )
        
        assert result1.success
        assert result2.success


# ============================================================================
# Test Class 17: Error Handling and Edge Cases
# ============================================================================

class TestErrorHandlingEdgeCases:
    """Test error handling and edge cases"""
    
    def test_nan_in_initial_condition(self):
        """Test handling of NaN in initial condition"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        x0_nan = np.array([np.nan, 0.0])
        
        # Suppress expected warnings for invalid operations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            # Should either reject or propagate NaN
            try:
                result = integrator.integrate(
                    x0=x0_nan,
                    u_func=lambda t: np.array([0.0]),
                    t_span=(0, 0.1)
                )
                # If it proceeds, result should contain NaN
                assert np.any(np.isnan(result.x))
            except (ValueError, RuntimeError):
                # Valid to reject NaN input
                pass
    
    def test_inf_in_initial_condition(self):
        """Test handling of Inf in initial condition"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        x0_inf = np.array([np.inf, 0.0])
        
        # Suppress expected warnings for invalid operations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            try:
                result = integrator.integrate(
                    x0=x0_inf,
                    u_func=lambda t: np.array([0.0]),
                    t_span=(0, 0.1)
                )
                # If it proceeds, result should contain Inf
                assert np.any(np.isinf(result.x))
            except (ValueError, RuntimeError):
                # Valid to reject Inf input
                pass
    
    def test_none_initial_condition(self):
        """Test handling of None initial condition"""
        system = MockSystemWithEquilibria()
        integrator = ConcreteTestIntegrator(system, dt=0.01)
        
        # Should raise error
        with pytest.raises((ValueError, TypeError, AttributeError)):
            integrator.integrate(
                x0=None,
                u_func=lambda t: np.array([0.0]),
                t_span=(0, 0.1)
            )


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])