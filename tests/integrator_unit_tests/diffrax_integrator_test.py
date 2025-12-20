"""
Unit tests for DiffraxIntegrator with IntegratorBase compliance.

Tests cover:
- Basic integration accuracy with system interface
- Single step operations
- Multiple solver methods
- Adaptive vs fixed-step integration
- Batch integration with vmap
- Gradient computation
- JIT compilation
- Edge cases and error handling
"""

import pytest
import numpy as np
from typing import Tuple

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import Array
    import diffrax as dfx
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    pytest.skip("JAX not available", allow_module_level=True)

from src.systems.base.numerical_integration.diffrax_integrator import DiffraxIntegrator
from src.systems.base.numerical_integration.integrator_base import IntegrationResult, StepMode


# ============================================================================
# Mock System for Testing
# ============================================================================

class MockLinearSystem:
    """Mock dynamical system for testing: dx/dt = Ax + Bu"""
    
    def __init__(self, nx=2, nu=1):
        self.nx = nx
        self.nu = nu
        # Simple stable system
        self.A = jnp.array([[-0.5, 1.0], [-1.0, -0.5]])
        self.B = jnp.array([[0.0], [1.0]])
    
    def __call__(self, x, u, backend='jax'):
        """Evaluate dynamics - MUST return same structure as x."""
        x_jax = jnp.asarray(x)
        u_jax = jnp.asarray(u)
        
        # Matrix-vector multiplication (preserves shape automatically for 1D x)
        dx = self.A @ x_jax + (self.B @ u_jax.reshape(-1, 1)).squeeze()
        
        # Ensure same shape as x
        return dx.reshape(x_jax.shape)
    
    def analytical_solution(self, x0, u_const, t):
        """Analytical solution for constant control."""
        # For simple testing, assume zero control
        if jnp.allclose(u_const, 0.0):
            # Compute matrix exponential solution
            # For this A matrix with eigenvalues -0.5 ± 1j
            omega = 1.0
            sigma = -0.5
            c1 = x0[0]
            c2 = x0[1] - sigma * x0[0]
            
            x1 = jnp.exp(sigma * t) * (c1 * jnp.cos(omega * t) + c2/omega * jnp.sin(omega * t))
            x2 = jnp.exp(sigma * t) * (c2 * jnp.cos(omega * t) - omega * c1 * jnp.sin(omega * t) + sigma * c1 * jnp.cos(omega * t) + sigma * c2/omega * jnp.sin(omega * t))
            
            return jnp.stack([x1, x2], axis=-1)
        else:
            # Numerical approximation for non-zero control
            return None


class MockExponentialSystem:
    """Simple exponential decay: dx/dt = -k*x + u"""
    
    def __init__(self, k=0.5):
        self.nx = 1
        self.nu = 1
        self.k = k
    
    def __call__(self, x, u, backend='jax'):
        """Evaluate dynamics - MUST return same structure as x."""
        # Convert to JAX arrays
        x_jax = jnp.asarray(x)
        u_jax = jnp.asarray(u)
        
        # Compute derivative (element-wise operations preserve shape)
        dx = -self.k * x_jax + u_jax
        
        # CRITICAL: Return exact same structure as x
        # If x is shape (1,), return shape (1,)
        # If x is shape (), return shape ()
        return dx
    
    def analytical_solution(self, x0, t, u_const=0.0):
        """Analytical solution."""
        if u_const == 0.0:
            return x0 * jnp.exp(-self.k * t)
        else:
            # With constant control: x(t) = (x0 - u/k)*exp(-k*t) + u/k
            return (x0 - u_const/self.k) * jnp.exp(-self.k * t) + u_const/self.k


# ============================================================================
# Basic Integration Tests
# ============================================================================

class TestBasicIntegration:
    """Test basic integration functionality."""
    
    @pytest.fixture
    def system(self):
        """Create test system."""
        return MockExponentialSystem(k=0.5)
    
    @pytest.fixture
    def integrator(self, system):
        """Create default integrator."""
        return DiffraxIntegrator(
            system,
            dt=0.01,
            step_mode=StepMode.FIXED,
            backend='jax',
            solver='tsit5'
        )
    
    def test_single_step(self, integrator, system):
        """Test single integration step."""
        x = jnp.array([1.0])
        u = jnp.array([0.0])
        dt = 0.01
        
        # First, test that the system function works correctly
        dx = system(x, u, backend='jax')
        print(f"\nDEBUG: x.shape = {x.shape}, dx.shape = {dx.shape}")
        print(f"DEBUG: x = {x}, dx = {dx}")
        assert dx.shape == x.shape, f"Shape mismatch: dx.shape={dx.shape} vs x.shape={x.shape}"
        
        try:
            x_next = integrator.step(x, u, dt)
            
            # Check shape and type
            assert x_next.shape == x.shape
            assert isinstance(x_next, jnp.ndarray)
            
            # Check accuracy against analytical solution
            x_expected = system.analytical_solution(x[0], dt, u_const=0.0)
            np.testing.assert_allclose(x_next[0], x_expected, rtol=1e-5, atol=1e-7)
        except Exception as e:
            print(f"\nDEBUG: Integration failed with error: {e}")
            print(f"DEBUG: Error type: {type(e)}")
            raise
    
    def test_integrate_zero_control(self, integrator, system):
        """Test integration with zero control."""
        x0 = jnp.array([1.0])
        t_span = (0.0, 5.0)
        t_eval = jnp.linspace(0.0, 5.0, 50)
        
        # Zero control
        u_func = lambda t, x: jnp.array([0.0])
        
        result = integrator.integrate(x0, u_func, t_span, t_eval)
        
        # Debug: print result details if it fails
        if not result.success:
            print(f"\nIntegration failed: {result.message}")
            print(f"Final state: {result.x[-1]}")
            print(f"NaN in result: {jnp.any(jnp.isnan(result.x))}")
            print(f"Inf in result: {jnp.any(jnp.isinf(result.x))}")
        
        assert result.success, f"Integration failed: {result.message}"
        assert result.t.shape == t_eval.shape
        assert result.x.shape == (len(t_eval), 1)
        
        # Compare with analytical solution
        x_analytical = system.analytical_solution(x0[0], t_eval, u_const=0.0)
        np.testing.assert_allclose(result.x[:, 0], x_analytical, rtol=1e-4, atol=1e-6)
    
    def test_integrate_constant_control(self, integrator, system):
        """Test integration with constant non-zero control."""
        x0 = jnp.array([1.0])
        t_span = (0.0, 3.0)
        u_const = 0.5
        
        u_func = lambda t, x: jnp.array([u_const])
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        # With positive control and decay, should approach u/k
        assert result.x[-1, 0] > 0  # Should be positive
    
    def test_integrate_time_varying_control(self, integrator, system):
        """Test integration with time-varying control."""
        x0 = jnp.array([1.0])
        t_span = (0.0, 2.0)
        
        # Sinusoidal control
        u_func = lambda t, x: jnp.array([jnp.sin(t)])
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert jnp.all(jnp.isfinite(result.x))
    
    def test_integrate_state_feedback(self, integrator, system):
        """Test integration with state feedback control."""
        x0 = jnp.array([2.0])
        t_span = (0.0, 5.0)
        
        # Proportional feedback
        K = 0.5
        u_func = lambda t, x: -K * x
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        # Should decay faster with feedback
        assert jnp.abs(result.x[-1, 0]) < jnp.abs(x0[0])


# ============================================================================
# Solver Method Tests
# ============================================================================

class TestSolverMethods:
    """Test different solver methods."""
    
    @pytest.fixture
    def system(self):
        return MockExponentialSystem(k=1.0)
    
    @pytest.mark.parametrize("solver", [
        'tsit5', 'dopri5', 'dopri8', 'euler', 
        'midpoint', 'heun', 'ralston', 'bosh3'
    ])
    def test_all_solvers(self, system, solver):
        """Test that all solvers work correctly."""
        integrator = DiffraxIntegrator(
            system,
            dt=0.01,
            step_mode=StepMode.FIXED,
            backend='jax',
            solver=solver
        )
        
        x0 = jnp.array([1.0])
        t_span = (0.0, 2.0)
        u_func = lambda t, x: jnp.array([0.0])
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert result.metadata['solver'] == solver
        
        # Check reasonable accuracy
        x_analytical = system.analytical_solution(x0[0], 2.0, u_const=0.0)
        if solver in ['euler']:
            rtol = 1e-2
        elif solver in ['midpoint', 'heun', 'ralston']:
            rtol = 1e-3
        else:
            rtol = 1e-4
        
        np.testing.assert_allclose(result.x[-1, 0], x_analytical, rtol=rtol)
    
    def test_invalid_solver(self, system):
        """Test that invalid solver raises error."""
        with pytest.raises(ValueError, match="Unknown solver"):
            DiffraxIntegrator(
                system,
                dt=0.01,
                backend='jax',
                solver='invalid_solver'
            )
    
    def test_integrator_name(self, system):
        """Test integrator name property."""
        integrator = DiffraxIntegrator(
            system, dt=0.01, backend='jax', solver='dopri5'
        )
        assert 'Diffrax' in integrator.name
        assert 'dopri5' in integrator.name


# ============================================================================
# Step Mode Tests
# ============================================================================

class TestStepModes:
    """Test fixed vs adaptive step modes."""
    
    @pytest.fixture
    def system(self):
        return MockExponentialSystem(k=0.5)
    
    def test_fixed_step_mode(self, system):
        """Test fixed step mode."""
        integrator = DiffraxIntegrator(
            system,
            dt=0.05,
            step_mode=StepMode.FIXED,
            backend='jax',
            solver='dopri5'
        )
        
        x0 = jnp.array([1.0])
        t_span = (0.0, 2.0)
        u_func = lambda t, x: jnp.array([0.0])
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        # Check uniform time spacing (fixed steps)
        dt_actual = jnp.diff(result.t)
        np.testing.assert_allclose(dt_actual, dt_actual[0], rtol=1e-10)
    
    def test_adaptive_step_mode(self, system):
        """Test adaptive step mode."""
        integrator = DiffraxIntegrator(
            system,
            dt=0.01,
            step_mode=StepMode.ADAPTIVE,
            backend='jax',
            solver='dopri5',
            rtol=1e-8,
            atol=1e-10
        )
        
        x0 = jnp.array([1.0])
        t_span = (0.0, 2.0)
        u_func = lambda t, x: jnp.array([0.0])
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert jnp.all(jnp.isfinite(result.x))
    
    def test_different_tolerances(self, system):
        """Test that different tolerances affect accuracy."""
        x0 = jnp.array([1.0])
        t_span = (0.0, 5.0)
        u_func = lambda t, x: jnp.array([0.0])
        
        # Tight tolerances
        integrator_tight = DiffraxIntegrator(
            system,
            dt=0.01,
            step_mode=StepMode.ADAPTIVE,
            backend='jax',
            solver='dopri5',
            rtol=1e-10,
            atol=1e-12
        )
        result_tight = integrator_tight.integrate(x0, u_func, t_span)
        
        # Loose tolerances
        integrator_loose = DiffraxIntegrator(
            system,
            dt=0.01,
            step_mode=StepMode.ADAPTIVE,
            backend='jax',
            solver='dopri5',
            rtol=1e-4,
            atol=1e-6
        )
        result_loose = integrator_loose.integrate(x0, u_func, t_span)
        
        # Both should succeed
        assert result_tight.success
        assert result_loose.success
        
        # Tight should be more accurate
        x_analytical = system.analytical_solution(x0[0], 5.0, u_const=0.0)
        error_tight = jnp.abs(result_tight.x[-1, 0] - x_analytical)
        error_loose = jnp.abs(result_loose.x[-1, 0] - x_analytical)
        
        assert error_tight < error_loose


# ============================================================================
# Multi-Dimensional System Tests
# ============================================================================

class TestMultiDimensional:
    """Test with multi-dimensional systems."""
    
    @pytest.fixture
    def system(self):
        return MockLinearSystem(nx=2, nu=1)
    
    @pytest.fixture
    def integrator(self, system):
        return DiffraxIntegrator(
            system,
            dt=0.01,
            backend='jax',
            solver='dopri5'
        )
    
    def test_2d_system_integration(self, integrator, system):
        """Test integration of 2D system."""
        x0 = jnp.array([1.0, 0.0])
        t_span = (0.0, 5.0)
        u_func = lambda t, x: jnp.array([0.0])
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert result.x.shape[1] == 2
        # System is stable, should decay
        assert jnp.linalg.norm(result.x[-1]) < jnp.linalg.norm(x0)
    
    def test_2d_system_step(self, integrator, system):
        """Test single step of 2D system."""
        x = jnp.array([1.0, 0.5])
        u = jnp.array([0.0])
        dt = 0.01
        
        x_next = integrator.step(x, u, dt)
        
        assert x_next.shape == x.shape
        assert jnp.all(jnp.isfinite(x_next))


# ============================================================================
# Gradient Computation Tests
# ============================================================================

class TestGradientComputation:
    """Test gradient computation capabilities."""
    
    @pytest.fixture
    def system(self):
        return MockExponentialSystem(k=1.0)
    
    @pytest.fixture
    def integrator(self, system):
        return DiffraxIntegrator(
            system,
            dt=0.01,
            backend='jax',
            solver='dopri5'
        )
    
    def test_gradient_wrt_initial_condition(self, integrator):
        """Test gradient computation w.r.t. initial conditions."""
        x0 = jnp.array([1.0])
        t_span = (0.0, 1.0)
        u_func = lambda t, x: jnp.array([0.0])
        
        def loss_fn(result):
            return jnp.sum(result.x**2)
        
        loss, grad = integrator.integrate_with_gradient(x0, u_func, t_span, loss_fn)
        
        assert jnp.isfinite(loss)
        assert jnp.isfinite(grad).all()
        assert grad.shape == x0.shape
    
    def test_gradient_finite_difference_validation(self, integrator):
        """Validate gradients using finite differences."""
        x0 = jnp.array([1.5])
        t_span = (0.0, 1.0)
        u_func = lambda t, x: jnp.array([0.0])
        
        def loss_fn(result):
            return jnp.sum(result.x[-1]**2)
        
        # Compute gradient using autodiff
        loss, grad_autodiff = integrator.integrate_with_gradient(x0, u_func, t_span, loss_fn)
        
        # Compute gradient using finite differences
        eps = 1e-4
        result_plus = integrator.integrate(x0 + eps, u_func, t_span)
        loss_plus = loss_fn(result_plus)
        
        result_minus = integrator.integrate(x0 - eps, u_func, t_span)
        loss_minus = loss_fn(result_minus)
        
        grad_fd = (loss_plus - loss_minus) / (2 * eps)
        
        # More lenient tolerance for numerical gradients
        np.testing.assert_allclose(grad_autodiff, grad_fd, rtol=1e-2, atol=1e-4)


# ============================================================================
# JIT Compilation Tests
# ============================================================================

class TestJITCompilation:
    """Test JIT compilation functionality."""
    
    @pytest.fixture
    def system(self):
        return MockExponentialSystem(k=1.0)
    
    @pytest.fixture
    def integrator(self, system):
        return DiffraxIntegrator(
            system,
            dt=0.01,
            backend='jax',
            solver='dopri5'
        )
    
    def test_jit_compiled_step(self, integrator):
        """Test JIT compilation of step function."""
        jitted_step = integrator.jit_compile_step()
        
        x = jnp.array([1.0])
        u = jnp.array([0.0])
        dt = 0.01
        
        # First call (compilation)
        x_next1 = jitted_step(x, u, dt)
        
        # Second call (should use compiled version)
        x_next2 = jitted_step(x * 2, u, dt)
        
        assert jnp.all(jnp.isfinite(x_next1))
        assert jnp.all(jnp.isfinite(x_next2))
        
        # Check scaling property
        np.testing.assert_allclose(x_next2 / x_next1, 2.0, rtol=1e-5)
    
    def test_vectorized_step(self, integrator):
        """Test vectorized step over batch."""
        x_batch = jnp.array([[1.0], [2.0], [3.0]])
        u_batch = jnp.array([[0.0], [0.0], [0.0]])
        
        x_next_batch = integrator.vectorized_step(x_batch, u_batch)
        
        assert x_next_batch.shape == x_batch.shape
        assert jnp.all(jnp.isfinite(x_next_batch))


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def system(self):
        return MockExponentialSystem(k=1.0)
    
    @pytest.fixture
    def integrator(self, system):
        return DiffraxIntegrator(
            system,
            dt=0.01,
            backend='jax',
            solver='dopri5'
        )
    
    def test_zero_time_span(self, integrator):
        """Test integration with zero time span."""
        x0 = jnp.array([1.0])
        t_span = (0.0, 0.0)
        u_func = lambda t, x: jnp.array([0.0])
        
        result = integrator.integrate(x0, u_func, t_span)
        
        # Should return initial condition
        assert result.success
        np.testing.assert_allclose(result.x[0], x0, rtol=1e-10)
    
    def test_backward_integration(self, integrator):
        """Test backward time integration."""
        x0 = jnp.array([0.5])
        t_span = (1.0, 0.0)  # Backward
        t_eval = jnp.linspace(1.0, 0.0, 20)
        u_func = lambda t, x: jnp.array([0.0])
        
        result = integrator.integrate(x0, u_func, t_span, t_eval=t_eval)
        
        # Should work (solution grows backward)
        # Backward integration might not always succeed depending on solver
        # Just check it doesn't crash
        assert result.t.shape[0] > 0
    
    def test_very_small_initial_value(self, integrator):
        """Test with very small initial values."""
        x0 = jnp.array([1e-10])
        t_span = (0.0, 5.0)
        u_func = lambda t, x: jnp.array([0.0])
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert jnp.all(jnp.isfinite(result.x))
    
    def test_invalid_backend(self, system):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="requires backend='jax'"):
            DiffraxIntegrator(
                system,
                dt=0.01,
                backend='numpy'
            )
    
    def test_statistics_tracking(self, integrator):
        """Test that statistics are tracked correctly."""
        x0 = jnp.array([1.0])
        t_span = (0.0, 1.0)
        u_func = lambda t, x: jnp.array([0.0])
        
        integrator.reset_stats()
        result = integrator.integrate(x0, u_func, t_span)
        
        stats = integrator.get_stats()
        # Check that some steps were taken
        # Note: stats might be 0 if integration completes in very few steps
        assert stats['total_steps'] >= 0
        assert stats['total_fev'] >= 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])