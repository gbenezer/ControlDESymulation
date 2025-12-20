"""
Comprehensive unit tests for DynamicsEvaluator

Tests cover:
1. NumPy backend evaluation
2. PyTorch backend evaluation
3. JAX backend evaluation
4. Input validation
5. Shape handling (batched vs single)
6. Backend dispatch
7. Performance tracking
8. Error handling
"""

import pytest
import numpy as np
import sympy as sp

# Conditional imports
torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

jax_available = True
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax_available = False

from src.systems.base.utils.dynamics_evaluator import DynamicsEvaluator
from src.systems.base.utils.code_generator import CodeGenerator
from src.systems.base.utils.backend_manager import BackendManager


# ============================================================================
# Mock Systems for Testing
# ============================================================================


class MockLinearSystem:
    """Simple linear system: dx/dt = -a*x + u"""
    
    def __init__(self, a=2.0):
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        a_sym = sp.symbols('a', real=True, positive=True)
        
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([-a_sym * x + u])
        self.parameters = {a_sym: a}
        self.nx = 1
        self.nu = 1
        self.order = 1
    
    def substitute_parameters(self, expr):
        return expr.subs(self.parameters)


class MockSecondOrderSystem:
    """Harmonic oscillator: q̈ = -k*q - c*q̇ + u"""
    
    def __init__(self, k=10.0, c=0.5):
        q, q_dot = sp.symbols('q q_dot', real=True)
        u = sp.symbols('u', real=True)
        k_sym, c_sym = sp.symbols('k c', real=True, positive=True)
        
        self.state_vars = [q, q_dot]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([-k_sym * q - c_sym * q_dot + u])
        self.parameters = {k_sym: k, c_sym: c}
        self.nx = 2
        self.nu = 1
        self.nq = 1
        self.order = 2
    
    def substitute_parameters(self, expr):
        return expr.subs(self.parameters)


class MockMultiInputSystem:
    """System with multiple states and controls"""
    
    def __init__(self):
        x1, x2 = sp.symbols('x1 x2', real=True)
        u1, u2 = sp.symbols('u1 u2', real=True)
        
        self.state_vars = [x1, x2]
        self.control_vars = [u1, u2]
        self._f_sym = sp.Matrix([x2 + u1, -x1 + u2])
        self.parameters = {}
        self.nx = 2
        self.nu = 2
        self.order = 1
    
    def substitute_parameters(self, expr):
        return expr


# ============================================================================
# Test Class 1: NumPy Backend Evaluation
# ============================================================================


class TestNumpyEvaluation:
    """Test NumPy backend evaluation"""
    
    def test_evaluate_numpy_single(self):
        """Test single point evaluation with NumPy"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = np.array([1.0])
        u = np.array([0.5])
        
        dx = evaluator.evaluate(x, u, backend='numpy')
        
        # dx = -2*x + u = -2*1 + 0.5 = -1.5
        assert isinstance(dx, np.ndarray)
        assert dx.shape == (1,)
        assert np.allclose(dx, np.array([-1.5]))
    
    def test_evaluate_numpy_batched(self):
        """Test batched evaluation with NumPy"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = np.array([[1.0], [2.0], [3.0]])
        u = np.array([[0.5], [0.5], [0.5]])
        
        dx = evaluator.evaluate(x, u, backend='numpy')
        
        assert dx.shape == (3, 1)
        expected = np.array([[-1.5], [-3.5], [-5.5]])
        assert np.allclose(dx, expected)
    
    def test_evaluate_numpy_auto_detect(self):
        """Test auto-detection of NumPy backend"""
        system = MockLinearSystem(a=1.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        # Should auto-detect NumPy
        dx = evaluator.evaluate(x, u)
        
        assert isinstance(dx, np.ndarray)
    
    def test_evaluate_numpy_second_order(self):
        """Test second-order system evaluation"""
        system = MockSecondOrderSystem(k=10.0, c=0.5)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = np.array([0.1, 0.0])  # [q, q_dot]
        u = np.array([0.0])
        
        dx = evaluator.evaluate(x, u, backend='numpy')
        
        # dx = q_ddot = -10*0.1 - 0.5*0 + 0 = -1.0
        assert dx.shape == (1,)
        assert np.allclose(dx, np.array([-1.0]))
    
    def test_evaluate_numpy_multi_input(self):
        """Test system with multiple states and controls"""
        system = MockMultiInputSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = np.array([1.0, 2.0])
        u = np.array([0.5, -0.5])
        
        dx = evaluator.evaluate(x, u, backend='numpy')
        
        # dx = [x2 + u1, -x1 + u2] = [2.0 + 0.5, -1.0 + (-0.5)] = [2.5, -1.5]
        assert dx.shape == (2,)
        assert np.allclose(dx, np.array([2.5, -1.5]))


# ============================================================================
# Test Class 2: PyTorch Backend Evaluation
# ============================================================================


class TestTorchEvaluation:
    """Test PyTorch backend evaluation"""
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_evaluate_torch_single(self):
        """Test single point evaluation with PyTorch"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = torch.tensor([1.0])
        u = torch.tensor([0.5])
        
        dx = evaluator.evaluate(x, u, backend='torch')
        
        assert isinstance(dx, torch.Tensor)
        assert dx.shape == (1,)
        assert torch.allclose(dx, torch.tensor([-1.5]))
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_evaluate_torch_batched(self):
        """Test batched evaluation with PyTorch"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = torch.tensor([[1.0], [2.0], [3.0]])
        u = torch.tensor([[0.5], [0.5], [0.5]])
        
        dx = evaluator.evaluate(x, u, backend='torch')
        
        assert dx.shape == (3, 1)
        expected = torch.tensor([[-1.5], [-3.5], [-5.5]])
        assert torch.allclose(dx, expected)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_evaluate_torch_auto_detect(self):
        """Test auto-detection of PyTorch backend"""
        system = MockLinearSystem(a=1.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = torch.tensor([1.0])
        u = torch.tensor([0.0])
        
        # Should auto-detect torch
        dx = evaluator.evaluate(x, u)
        
        assert isinstance(dx, torch.Tensor)
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_evaluate_torch_gradients(self):
        """Test that PyTorch gradients work"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = torch.tensor([1.0], requires_grad=True)
        u = torch.tensor([0.5])
        
        dx = evaluator.evaluate(x, u, backend='torch')
        dx.backward()
        
        # df/dx = -a = -2
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.tensor([-2.0]))


# ============================================================================
# Test Class 3: JAX Backend Evaluation
# ============================================================================


class TestJaxEvaluation:
    """Test JAX backend evaluation"""
    
    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_evaluate_jax_single(self):
        """Test single point evaluation with JAX"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = jnp.array([1.0])
        u = jnp.array([0.5])
        
        dx = evaluator.evaluate(x, u, backend='jax')
        
        assert isinstance(dx, jnp.ndarray)
        assert jnp.allclose(dx, jnp.array([-1.5]))
    
    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_evaluate_jax_batched(self):
        """Test batched evaluation with JAX (uses vmap)"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = jnp.array([[1.0], [2.0], [3.0]])
        u = jnp.array([[0.5], [0.5], [0.5]])
        
        dx = evaluator.evaluate(x, u, backend='jax')
        
        assert dx.shape == (3, 1)
        expected = jnp.array([[-1.5], [-3.5], [-5.5]])
        assert jnp.allclose(dx, expected)
    
    @pytest.mark.skipif(not jax_available, reason="JAX not installed")
    def test_evaluate_jax_auto_detect(self):
        """Test auto-detection of JAX backend"""
        system = MockLinearSystem(a=1.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = jnp.array([1.0])
        u = jnp.array([0.0])
        
        # Should auto-detect JAX
        dx = evaluator.evaluate(x, u)
        
        assert isinstance(dx, jnp.ndarray)


# ============================================================================
# Test Class 4: Backend Conversion and Dispatch
# ============================================================================


class TestBackendDispatch:
    """Test backend dispatch and conversion"""
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_force_backend_conversion(self):
        """Test forcing backend with input conversion"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        # NumPy input, force torch backend
        x = np.array([1.0])
        u = np.array([0.5])
        
        dx = evaluator.evaluate(x, u, backend='torch')
        
        # Should return torch tensor
        assert isinstance(dx, torch.Tensor)
        assert torch.allclose(dx, torch.tensor([-1.5]))
    
    def test_default_backend(self):
        """Test using configured default backend"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        backend_mgr.set_default('numpy')
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = np.array([1.0])
        u = np.array([0.5])
        
        dx = evaluator.evaluate(x, u, backend='default')
        
        assert isinstance(dx, np.ndarray)
    
    def test_multi_backend_consistency(self):
        """Test that all backends give same numerical result"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x_np = np.array([1.0])
        u_np = np.array([0.5])
        
        # Evaluate with NumPy
        dx_np = evaluator.evaluate(x_np, u_np, backend='numpy')
        
        backends_to_test = ['numpy']
        if torch_available:
            backends_to_test.append('torch')
        if jax_available:
            backends_to_test.append('jax')
        
        # All backends should give same result
        for backend in backends_to_test:
            dx = evaluator.evaluate(x_np, u_np, backend=backend)
            dx_val = np.array(dx) if not isinstance(dx, np.ndarray) else dx
            
            assert np.allclose(dx_val, dx_np), f"{backend} doesn't match NumPy"


# ============================================================================
# Test Class 5: Input Validation
# ============================================================================


class TestInputValidation:
    """Test input validation and error handling"""
    
    def test_zero_dimensional_input_error(self):
        """Test error on scalar input"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        with pytest.raises(ValueError, match="at least 1D"):
            evaluator.evaluate(np.array(1.0), np.array(0.0))
    
    def test_wrong_state_dimension(self):
        """Test error on wrong state dimension"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = np.array([1.0, 2.0])  # 2D but system is 1D!
        u = np.array([0.0])
        
        with pytest.raises(ValueError, match="Expected state dimension"):
            evaluator.evaluate(x, u, backend='numpy')
    
    def test_wrong_control_dimension(self):
        """Test error on wrong control dimension"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = np.array([1.0])
        u = np.array([0.0, 0.0])  # 2D but system has 1 control!
        
        with pytest.raises(ValueError, match="Expected control dimension"):
            evaluator.evaluate(x, u, backend='numpy')
    
    @pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
    def test_torch_dimension_validation(self):
        """Test dimension validation for PyTorch"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = torch.tensor([1.0, 2.0])  # Wrong size
        u = torch.tensor([0.0])
        
        with pytest.raises(ValueError, match="Expected state dimension"):
            evaluator.evaluate(x, u, backend='torch')


# ============================================================================
# Test Class 6: Performance Tracking
# ============================================================================


class TestPerformanceTracking:
    """Test performance statistics tracking"""
    
    def test_initial_stats(self):
        """Test initial stats are zero"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        stats = evaluator.get_stats()
        
        assert stats['calls'] == 0
        assert stats['total_time'] == 0.0
        assert stats['avg_time'] == 0.0
    
    def test_stats_increment(self):
        """Test that stats increment with calls"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        # Make several calls
        evaluator.evaluate(x, u)
        evaluator.evaluate(x, u)
        evaluator.evaluate(x, u)
        
        stats = evaluator.get_stats()
        
        assert stats['calls'] == 3
        assert stats['total_time'] > 0
        assert stats['avg_time'] > 0
    
    def test_reset_stats(self):
        """Test resetting performance stats"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        # Make some calls
        evaluator.evaluate(x, u)
        assert evaluator.get_stats()['calls'] > 0
        
        # Reset
        evaluator.reset_stats()
        stats = evaluator.get_stats()
        
        assert stats['calls'] == 0
        assert stats['total_time'] == 0.0


# ============================================================================
# Test Class 7: Function Caching
# ============================================================================


class TestFunctionCaching:
    """Test that generated functions are properly cached"""
    
    def test_function_reuse(self):
        """Test that cached functions are reused"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = np.array([1.0])
        u = np.array([0.0])
        
        # First call generates function
        evaluator.evaluate(x, u, backend='numpy')
        func1 = code_gen.get_dynamics('numpy')
        
        # Second call reuses cached function
        evaluator.evaluate(x, u, backend='numpy')
        func2 = code_gen.get_dynamics('numpy')
        
        assert func1 is func2  # Same function object


# ============================================================================
# Test Class 8: String Representations
# ============================================================================


class TestStringRepresentations:
    """Test __repr__ and __str__ methods"""
    
    def test_repr(self):
        """Test __repr__ output"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        repr_str = repr(evaluator)
        
        assert 'DynamicsEvaluator' in repr_str
        assert 'nx=1' in repr_str
        assert 'nu=1' in repr_str
    
    def test_str(self):
        """Test __str__ output"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        str_repr = str(evaluator)
        
        assert 'DynamicsEvaluator' in str_repr
        assert 'calls=' in str_repr


# ============================================================================
# Test Class 9: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios"""
    
    def test_large_batch(self):
        """Test with large batch size"""
        system = MockLinearSystem()
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        batch_size = 100
        x = np.random.randn(batch_size, 1)
        u = np.random.randn(batch_size, 1)
        
        dx = evaluator.evaluate(x, u, backend='numpy')
        
        assert dx.shape == (batch_size, 1)
    
    def test_zero_state(self):
        """Test evaluation at zero state"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x = np.array([0.0])
        u = np.array([1.0])
        
        dx = evaluator.evaluate(x, u, backend='numpy')
        
        # dx = -2*0 + 1 = 1
        assert np.allclose(dx, np.array([1.0]))


# ============================================================================
# Test Class 10: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features"""
    
    def test_full_workflow(self):
        """Test complete evaluation workflow"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        # Test multiple evaluations
        for _ in range(5):
            x = np.random.randn(1)
            u = np.random.randn(1)
            dx = evaluator.evaluate(x, u)
            
            # Verify result shape
            assert dx.shape == (1,)
        
        # Check stats
        stats = evaluator.get_stats()
        assert stats['calls'] == 5
    
    @pytest.mark.skipif(not (torch_available and jax_available),
                        reason="Both PyTorch and JAX required")
    def test_multi_backend_workflow(self):
        """Test switching between backends"""
        system = MockLinearSystem(a=2.0)
        code_gen = CodeGenerator(system)
        backend_mgr = BackendManager()
        evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        
        x_np = np.array([1.0])
        u_np = np.array([0.5])
        
        # Evaluate with different backends
        dx_np = evaluator.evaluate(x_np, u_np, backend='numpy')
        dx_torch = evaluator.evaluate(x_np, u_np, backend='torch')
        dx_jax = evaluator.evaluate(x_np, u_np, backend='jax')
        
        # All should give same result
        assert np.allclose(dx_np, np.array(dx_torch))
        assert np.allclose(dx_np, np.array(dx_jax))


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])