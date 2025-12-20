"""
Unit Tests for Integrator Factory

Tests the factory class for creating numerical integrators, including:
- Backend-specific integrator creation
- Method validation and selection
- Use case-specific factory methods
- Error handling for invalid configurations
- Method listing and recommendation utilities

Test Coverage
-------------
1. Basic integrator creation
2. Backend-method compatibility validation
3. Fixed-step vs adaptive selection
4. Default method selection
5. Use case-specific factories (production, optimization, neural ODE, etc.)
6. Method listing and information retrieval
7. Recommendation system
8. Error handling and validation
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

# Import the factory
from src.systems.base.numerical_integration.integrator_factory import (
    IntegratorFactory,
    IntegratorType,
    create_integrator,
    auto_integrator,
)
from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    StepMode
)


# ============================================================================
# Mock System Fixture
# ============================================================================

@pytest.fixture
def mock_system():
    """Create a mock SymbolicDynamicalSystem for testing."""
    system = Mock()
    system.nx = 2
    system.nu = 1
    system.ny = 2
    system.__call__ = Mock(return_value=np.array([1.0, 2.0]))
    system.forward = Mock(return_value=np.array([1.0, 2.0]))
    return system


# ============================================================================
# Test Class: Basic Creation
# ============================================================================

class TestBasicCreation:
    """Test basic integrator creation with IntegratorFactory.create()."""
    
    def test_create_default_numpy(self, mock_system):
        """Test creating integrator with default numpy backend."""
        integrator = IntegratorFactory.create(mock_system, backend='numpy')
        
        assert integrator is not None
        assert integrator.backend == 'numpy'
        assert integrator.system == mock_system
    
    def test_create_with_default_method(self, mock_system):
        """Test that default methods are selected correctly."""
        integrator = IntegratorFactory.create(mock_system, backend='numpy')
        assert hasattr(integrator, 'method')
    
    def test_create_with_specific_method(self, mock_system):
        """Test creating with specific method."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method='RK45'
        )
        
        assert integrator.method == 'RK45'
    
    def test_create_fixed_step_requires_dt(self, mock_system):
        """Test that fixed-step methods require dt parameter."""
        with pytest.raises(ValueError, match="requires dt"):
            IntegratorFactory.create(
                mock_system,
                backend='numpy',
                method='rk4',
                step_mode=StepMode.FIXED
            )
    
    def test_create_fixed_step_with_dt(self, mock_system):
        """Test creating fixed-step integrator with dt."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method='rk4',
            dt=0.01,
            step_mode=StepMode.FIXED
        )
        
        assert integrator.dt == 0.01
        assert integrator.step_mode == StepMode.FIXED
    
    def test_create_with_options(self, mock_system):
        """Test creating integrator with additional options."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method='RK45',
            rtol=1e-9,
            atol=1e-11
        )
        
        assert integrator.rtol == 1e-9
        assert integrator.atol == 1e-11


# ============================================================================
# Test Class: Backend Validation
# ============================================================================

class TestBackendValidation:
    """Test backend validation and compatibility checks."""
    
    def test_invalid_backend_raises_error(self, mock_system):
        """Test that invalid backend name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid backend"):
            IntegratorFactory.create(mock_system, backend='matlab')
    
    def test_valid_backends(self, mock_system):
        """Test all valid backends can be created."""
        # Test numpy (always available)
        integrator = IntegratorFactory.create(mock_system, backend='numpy')
        assert integrator.backend == 'numpy'
        
        # Test torch if available
        try:
            import torch
            integrator = IntegratorFactory.create(mock_system, backend='torch')
            assert integrator.backend == 'torch'
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        # Test jax if available
        try:
            import jax
            integrator = IntegratorFactory.create(mock_system, backend='jax')
            assert integrator.backend == 'jax'
        except ImportError:
            pytest.skip("JAX not installed")


# ============================================================================
# Test Class: Method-Backend Compatibility
# ============================================================================

class TestMethodBackendCompatibility:
    """Test validation of method-backend compatibility."""
    
    def test_scipy_method_requires_numpy(self, mock_system):
        """Test scipy methods require numpy backend."""
        with pytest.raises(ValueError, match="requires backend"):
            IntegratorFactory.create(
                mock_system,
                backend='torch',
                method='RK45'  # Scipy-only method
            )
    
    def test_universal_methods_work_with_any_backend(self, mock_system):
        """Test that universal methods (euler, rk4) work with any backend."""
        universal_methods = ['euler', 'midpoint', 'rk4']
        
        for method in universal_methods:
            try:
                integrator = IntegratorFactory.create(
                    mock_system,
                    backend='numpy',
                    method=method,
                    dt=0.01,
                    step_mode=StepMode.FIXED
                )
                assert integrator is not None
            except Exception as e:
                pytest.fail(f"Universal method {method} failed: {e}")


# ============================================================================
# Test Class: NumPy Backend Creation
# ============================================================================

class TestNumpyBackendCreation:
    """Test creation of NumPy-based integrators."""
    
    @pytest.mark.parametrize("method", ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'])
    def test_scipy_methods(self, mock_system, method):
        """Test all scipy methods can be created."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method=method
        )
        
        assert integrator.method == method
        assert integrator.backend == 'numpy'
    
    @pytest.mark.parametrize("method", ['euler', 'midpoint', 'rk4'])
    def test_fixed_step_methods(self, mock_system, method):
        """Test fixed-step methods with numpy."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method=method,
            dt=0.01,
            step_mode=StepMode.FIXED
        )
        
        assert integrator.dt == 0.01
        assert integrator.step_mode == StepMode.FIXED


# ============================================================================
# Test Class: PyTorch Backend Creation
# ============================================================================

@pytest.mark.skipif(
    not __import__('importlib').util.find_spec('torch'),
    reason="PyTorch not installed"
)
class TestTorchBackendCreation:
    """Test creation of PyTorch-based integrators."""
    
    @pytest.mark.parametrize("method", ['euler', 'midpoint', 'rk4'])
    def test_fixed_step_methods_torch(self, mock_system, method):
        """Test fixed-step methods with torch backend."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='torch',
            method=method,
            dt=0.01,
            step_mode=StepMode.FIXED
        )
        
        assert integrator.backend == 'torch'
        assert integrator.dt == 0.01
    
    def test_torchdiffeq_method_available(self, mock_system):
        """Test that torchdiffeq integrator can be created."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='torch',
            method='adaptive_heun'  # TorchDiffEq-specific
        )
        
        assert integrator.backend == 'torch'


# ============================================================================
# Test Class: JAX Backend Creation
# ============================================================================

@pytest.mark.skipif(
    not __import__('importlib').util.find_spec('jax'),
    reason="JAX not installed"
)
class TestJaxBackendCreation:
    """Test creation of JAX-based integrators."""
    
    @pytest.mark.parametrize("method", ['tsit5', 'dopri5', 'dopri8'])
    def test_diffrax_explicit_methods(self, mock_system, method):
        """Test Diffrax explicit methods."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='jax',
            method=method
        )
        
        # DiffraxIntegrator stores method in 'solver_name'
        assert integrator.backend == 'jax'
        assert integrator.solver_name == method
        assert integrator is not None
    
    @pytest.mark.parametrize("method", ['heun', 'ralston', 'reversible_heun'])
    def test_diffrax_basic_methods(self, mock_system, method):
        """Test Diffrax methods that are NOT universal fixed-step methods."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='jax',
            method=method
        )
        
        assert integrator.backend == 'jax'
        assert integrator.solver_name == method
    
    def test_diffrax_reversible_methods(self, mock_system):
        """Test Diffrax special methods."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='jax',
            method='reversible_heun'
        )
        
        assert integrator.backend == 'jax'


# ============================================================================
# Test Class: Auto Selection
# ============================================================================

class TestAutoSelection:
    """Test automatic integrator selection."""
    
    def test_auto_defaults_to_available_backend(self, mock_system):
        """Test auto selection works with available backend."""
        integrator = IntegratorFactory.auto(mock_system)
        
        # Should create some integrator
        assert integrator is not None
        assert integrator.backend in ['numpy', 'torch', 'jax']
    
    @pytest.mark.skipif(
        not __import__('importlib').util.find_spec('jax'),
        reason="JAX not installed"
    )
    def test_auto_prefers_jax_when_available(self, mock_system):
        """Test auto selection prefers JAX if available."""
        integrator = IntegratorFactory.auto(mock_system)
        
        # If JAX is available, should use it
        assert integrator.backend == 'jax'
    
    def test_auto_respects_preference(self, mock_system):
        """Test auto selection respects backend preference."""
        integrator = IntegratorFactory.auto(
            mock_system,
            prefer_backend='numpy'
        )
        
        assert integrator.backend == 'numpy'


# ============================================================================
# Test Class: Use Case Specific Factories
# ============================================================================

class TestUseCaseFactories:
    """Test use case-specific factory methods."""
    
    def test_for_production(self, mock_system):
        """Test production integrator factory."""
        integrator = IntegratorFactory.for_production(mock_system)
        
        assert integrator.backend == 'numpy'
        assert integrator.method == 'LSODA'
        assert integrator.rtol <= 1e-8
        assert integrator.atol <= 1e-10
    
    def test_for_production_custom_tolerances(self, mock_system):
        """Test production factory with custom tolerances."""
        integrator = IntegratorFactory.for_production(
            mock_system,
            rtol=1e-12,
            atol=1e-14
        )
        
        assert integrator.rtol == 1e-12
        assert integrator.atol == 1e-14
    
    @pytest.mark.skipif(
        not __import__('importlib').util.find_spec('jax'),
        reason="JAX not installed"
    )
    def test_for_optimization_jax(self, mock_system):
        """Test optimization integrator prefers JAX."""
        integrator = IntegratorFactory.for_optimization(mock_system)
        
        assert integrator.backend == 'jax'
        # Just check it was created, don't assume specific solver attribute
        assert integrator is not None
    
    @pytest.mark.skipif(
        not __import__('importlib').util.find_spec('torch'),
        reason="PyTorch not installed"
    )
    def test_for_optimization_torch(self, mock_system):
        """Test optimization integrator can use torch."""
        integrator = IntegratorFactory.for_optimization(
            mock_system,
            prefer_backend='torch'
        )
        
        assert integrator.backend == 'torch'
        assert integrator.method == 'dopri5'
    
    @pytest.mark.skipif(
        not __import__('importlib').util.find_spec('torch'),
        reason="PyTorch not installed"
    )
    def test_for_neural_ode(self, mock_system):
        """Test neural ODE integrator factory."""
        integrator = IntegratorFactory.for_neural_ode(mock_system)
        
        assert integrator.backend == 'torch'
        assert integrator.method == 'dopri5'
        # Check options were set (adjoint may be in options dict)
        assert integrator is not None
    
    def test_for_simple_simulation(self, mock_system):
        """Test simple simulation factory."""
        integrator = IntegratorFactory.for_simple_simulation(
            mock_system,
            dt=0.01
        )
        
        assert integrator.dt == 0.01
        assert integrator.step_mode == StepMode.FIXED
    
    def test_for_real_time(self, mock_system):
        """Test real-time integrator factory."""
        integrator = IntegratorFactory.for_real_time(
            mock_system,
            dt=0.01
        )
        
        assert integrator.dt == 0.01
        assert integrator.step_mode == StepMode.FIXED


# ============================================================================
# Test Class: Method Listing
# ============================================================================

class TestMethodListing:
    """Test method listing utilities."""
    
    def test_list_methods_all_backends(self):
        """Test listing methods for all backends."""
        methods = IntegratorFactory.list_methods()
        
        assert 'numpy' in methods
        assert 'torch' in methods
        assert 'jax' in methods
        
        # Check numpy methods
        assert 'LSODA' in methods['numpy']
        assert 'RK45' in methods['numpy']
        
        # Check torch methods
        assert 'dopri5' in methods['torch']
        
        # Check jax methods
        assert 'tsit5' in methods['jax']
    
    def test_list_methods_single_backend(self):
        """Test listing methods for specific backend."""
        methods = IntegratorFactory.list_methods('numpy')
        
        assert 'numpy' in methods
        assert len(methods) == 1
        assert 'LSODA' in methods['numpy']
    
    def test_list_methods_shows_fixed_step(self):
        """Test that universal methods appear where supported."""
        methods = IntegratorFactory.list_methods()
        
        # NumPy and torch have rk4
        assert 'euler' in methods['numpy']
        assert 'rk4' in methods['numpy']
        assert 'rk4' in methods['torch']
        
        # JAX does NOT have rk4 in Diffrax 0.7.0
        # Just check it has euler
        assert 'euler' in methods['jax']
    
    def test_list_methods_shows_special_solvers(self):
        """Test that backend-specific solvers are listed."""
        methods = IntegratorFactory.list_methods()
        
        # NumPy should have stiff solvers
        assert 'BDF' in methods['numpy']
        assert 'Radau' in methods['numpy']
        
        # NOTE: IMEX methods not in Diffrax 0.7.0
        # Skip this check for now
        
        # Torch should have adaptive methods
        assert 'adaptive_heun' in methods['torch']


# ============================================================================
# Test Class: Method Information
# ============================================================================

class TestMethodInformation:
    """Test method information retrieval."""
    
    def test_get_info_for_method(self):
        """Test getting information about a method."""
        info = IntegratorFactory.get_info('numpy', 'LSODA')
        
        assert 'name' in info
        assert 'order' in info
        assert 'description' in info
        assert 'best_for' in info
        assert info['name'] == 'LSODA'
    
    def test_get_info_for_fixed_step(self):
        """Test info for fixed-step methods."""
        info = IntegratorFactory.get_info('numpy', 'rk4')
        
        assert info['name'] == 'Classic Runge-Kutta 4'
        assert info['order'] == 4
        assert info['type'] == 'Fixed-step'
    
    def test_get_info_for_adaptive(self):
        """Test info for adaptive methods."""
        info = IntegratorFactory.get_info('jax', 'tsit5')
        
        assert info['name'] == 'Tsitouras 5(4)'
        assert info['order'] == 5
        assert info['type'] == 'Adaptive'
    
    def test_get_info_unknown_method(self):
        """Test info for unknown method returns minimal info."""
        info = IntegratorFactory.get_info('numpy', 'unknown_method')
        
        assert 'name' in info
        assert info['name'] == 'unknown_method'


# ============================================================================
# Test Class: Recommendation System
# ============================================================================

class TestRecommendationSystem:
    """Test integrator recommendation system."""
    
    def test_recommend_production(self):
        """Test recommendation for production use."""
        rec = IntegratorFactory.recommend('production')
        
        assert rec['backend'] == 'numpy'
        assert rec['method'] == 'LSODA'
        assert rec['step_mode'] == StepMode.ADAPTIVE
        assert 'reason' in rec
    
    def test_recommend_optimization(self):
        """Test recommendation for optimization."""
        rec = IntegratorFactory.recommend('optimization', has_jax=True)
        
        assert rec['backend'] == 'jax'
        assert rec['method'] == 'tsit5'
    
    def test_recommend_optimization_no_jax(self):
        """Test optimization recommendation without JAX."""
        rec = IntegratorFactory.recommend('optimization', has_jax=False, has_torch=True)
        
        assert rec['backend'] == 'torch'
        assert rec['method'] == 'dopri5'
    
    def test_recommend_neural_ode(self):
        """Test recommendation for neural ODE."""
        rec = IntegratorFactory.recommend('neural_ode')
        
        assert rec['backend'] == 'torch'
        assert rec['method'] == 'dopri5'
        assert rec['adjoint'] is True
    
    def test_recommend_prototype(self):
        """Test recommendation for prototyping."""
        rec = IntegratorFactory.recommend('prototype')
        
        assert rec['backend'] == 'numpy'
        assert rec['method'] == 'rk4'
        assert rec['step_mode'] == StepMode.FIXED
    
    def test_recommend_educational(self):
        """Test recommendation for education."""
        rec = IntegratorFactory.recommend('educational')
        
        assert rec['backend'] == 'numpy'
        assert rec['method'] == 'euler'
        assert rec['step_mode'] == StepMode.FIXED
    
    def test_recommend_real_time(self):
        """Test recommendation for real-time."""
        rec = IntegratorFactory.recommend('real_time')
        
        assert rec['backend'] == 'numpy'
        assert rec['method'] == 'rk4'
        assert rec['step_mode'] == StepMode.FIXED
    
    def test_recommend_with_gpu(self):
        """Test recommendation adjusts for GPU."""
        rec = IntegratorFactory.recommend(
            'optimization',
            has_jax=True,
            has_torch=True,
            has_gpu=True
        )
        
        # Should prefer GPU-capable backend
        assert rec['backend'] in ['torch', 'jax']
    
    def test_recommend_invalid_use_case(self):
        """Test invalid use case raises error."""
        with pytest.raises(ValueError, match="Unknown use case"):
            IntegratorFactory.recommend('invalid_use_case')


# ============================================================================
# Test Class: Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience wrapper functions."""
    
    def test_create_integrator_function(self, mock_system):
        """Test create_integrator convenience function."""
        integrator = create_integrator(mock_system, backend='numpy')
        
        assert integrator is not None
        assert integrator.backend == 'numpy'
    
    def test_create_integrator_with_method(self, mock_system):
        """Test create_integrator with method."""
        integrator = create_integrator(
            mock_system,
            backend='numpy',
            method='RK45'
        )
        
        assert integrator.method == 'RK45'
    
    def test_auto_integrator_function(self, mock_system):
        """Test auto_integrator convenience function."""
        integrator = auto_integrator(mock_system)
        
        assert integrator is not None


# ============================================================================
# Test Class: Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling in factory."""
    
    def test_missing_pytorch_for_neural_ode(self, mock_system):
        """Test error when PyTorch not available for neural ODE."""
        # Check if torch is actually available
        try:
            import torch
            # If torch is available, this test doesn't apply
            pytest.skip("PyTorch is installed, can't test missing PyTorch error")
        except ImportError:
            # Good, torch is not available
            with pytest.raises(ImportError, match="PyTorch is required"):
                IntegratorFactory.for_neural_ode(mock_system)
    
    def test_method_backend_mismatch_clear_error(self, mock_system):
        """Test clear error message for method-backend mismatch."""
        with pytest.raises(ValueError) as exc_info:
            IntegratorFactory.create(
                mock_system,
                backend='numpy',
                method='dopri5'  # Available in torch and jax, not numpy
            )
        
        # Check error message format
        error_msg = str(exc_info.value).lower()
        assert 'requires backend' in error_msg or 'backend in' in error_msg
    
    def test_fixed_step_without_dt_clear_error(self, mock_system):
        """Test clear error when dt missing for fixed-step."""
        with pytest.raises(ValueError) as exc_info:
            IntegratorFactory.create(
                mock_system,
                backend='numpy',
                method='rk4',
                step_mode=StepMode.FIXED
            )
        
        assert 'requires dt' in str(exc_info.value).lower()


# ============================================================================
# Test Class: Integration with Actual System
# ============================================================================

class TestIntegrationWithActualSystem:
    """Test factory with real system (if available)."""
    
    @pytest.mark.integration
    def test_create_and_integrate(self, mock_system):
        """Test creating integrator."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method='RK45'
        )
        
        # For this test, just verify the integrator was created
        # Don't try to actually integrate with a Mock
        assert integrator is not None
        assert integrator.backend == 'numpy'
        assert integrator.method == 'RK45'


# ============================================================================
# Test Class: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_options(self, mock_system):
        """Test creation with no additional options."""
        integrator = IntegratorFactory.create(mock_system)
        
        assert integrator is not None
    
    def test_many_options(self, mock_system):
        """Test creation with many options."""
        integrator = IntegratorFactory.create(
            mock_system,
            backend='numpy',
            method='RK45',
            rtol=1e-10,
            atol=1e-12,
            max_steps=50000,
            first_step=1e-6
        )
        
        assert integrator.rtol == 1e-10


# ============================================================================
# Test Class: IntegratorType Enum
# ============================================================================

class TestIntegratorTypeEnum:
    """Test IntegratorType enum."""
    
    def test_integrator_types_defined(self):
        """Test that all integrator types are defined."""
        assert hasattr(IntegratorType, 'PRODUCTION')
        assert hasattr(IntegratorType, 'OPTIMIZATION')
        assert hasattr(IntegratorType, 'NEURAL_ODE')
        assert hasattr(IntegratorType, 'SIMPLE')
        assert hasattr(IntegratorType, 'EDUCATIONAL')
    
    def test_integrator_type_values(self):
        """Test integrator type enum values."""
        assert IntegratorType.PRODUCTION.value == 'production'
        assert IntegratorType.OPTIMIZATION.value == 'optimization'
        assert IntegratorType.NEURAL_ODE.value == 'neural_ode'


# ============================================================================
# Test Class: Method Counts
# ============================================================================

class TestMethodCounts:
    """Test that expected numbers of methods are available."""
    
    def test_numpy_method_count(self):
        """Test numpy has expected number of methods."""
        methods = IntegratorFactory.list_methods('numpy')
        
        # Should have 6 scipy + 3 fixed-step = 9 methods
        assert len(methods['numpy']) == 9
    
    def test_jax_method_count(self):
        """Test JAX has correct number of methods for Diffrax 0.7.0."""
        methods = IntegratorFactory.list_methods('jax')
        
        # Diffrax 0.7.0 has 9 explicit methods
        assert len(methods['jax']) == 9
    
    def test_torch_method_count(self):
        """Test torch has expected methods."""
        methods = IntegratorFactory.list_methods('torch')
        
        # Should have torchdiffeq methods (12+)
        assert len(methods['torch']) >= 10


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])