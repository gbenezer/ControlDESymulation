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
Comprehensive unit tests for DiscreteSymbolicSystem

Tests cover:
- System initialization and validation
- Forward dynamics evaluation
- Linearization (numerical and symbolic)
- Backend compatibility (NumPy, PyTorch, JAX)
- Autonomous vs controlled systems
- Edge cases and error handling
- Equilibrium management
- Comparison with continuous systems
"""

from typing import Optional

import numpy as np
import pytest
import sympy as sp

# Conditional imports for optional backends
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from src.systems.base.discrete_symbolic_system import DiscreteSymbolicSystem

# ============================================================================
# Test Fixtures - Example Systems
# ============================================================================


class DiscreteLinearSystem(DiscreteSymbolicSystem):
    """Simple 2D discrete linear system for testing."""

    def define_system(self, a11=0.9, a12=0.1, a21=-0.1, a22=0.8, b=1.0):
        x1, x2 = sp.symbols("x1 x2", real=True)
        u = sp.symbols("u", real=True)

        a11_sym, a12_sym = sp.symbols("a11 a12", real=True)
        a21_sym, a22_sym = sp.symbols("a21 a22", real=True)
        b_sym = sp.symbols("b", real=True)

        # x[k+1] = A*x[k] + B*u[k]
        self.state_vars = [x1, x2]
        self.control_vars = [u]
        self._f_sym = sp.Matrix(
            [a11_sym * x1 + a12_sym * x2, a21_sym * x1 + a22_sym * x2 + b_sym * u]
        )
        self.parameters = {a11_sym: a11, a12_sym: a12, a21_sym: a21, a22_sym: a22, b_sym: b}
        self.order = 1


class DiscreteLogisticMap(DiscreteSymbolicSystem):
    """Nonlinear autonomous discrete system (logistic map)."""

    def define_system(self, r=3.5):
        x = sp.symbols("x", real=True, positive=True)
        r_sym = sp.symbols("r", positive=True)

        # x[k+1] = r*x[k]*(1 - x[k])
        self.state_vars = [x]
        self.control_vars = []
        self._f_sym = sp.Matrix([r_sym * x * (1 - x)])
        self.parameters = {r_sym: r}
        self.order = 1


class DiscreteIntegrator(DiscreteSymbolicSystem):
    """Simple discrete integrator: x[k+1] = x[k] + dt*u[k]"""

    def define_system(self, dt=0.1):
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)
        dt_sym = sp.symbols("dt", positive=True)

        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([x + dt_sym * u])
        self.parameters = {dt_sym: dt}
        self.order = 1


class DiscreteDoubleIntegrator(DiscreteSymbolicSystem):
    """2nd order discrete system: [p, v] with acceleration control."""

    def define_system(self, dt=0.1):
        p, v = sp.symbols("p v", real=True)
        a = sp.symbols("a", real=True)
        dt_sym = sp.symbols("dt", positive=True)

        # Discrete double integrator with Euler discretization
        # p[k+1] = p[k] + dt*v[k]
        # v[k+1] = v[k] + dt*a[k]
        self.state_vars = [p, v]
        self.control_vars = [a]
        self._f_sym = sp.Matrix([p + dt_sym * v, v + dt_sym * a])
        self.parameters = {dt_sym: dt}
        self.order = 1


class DiscreteSystemWithOutput(DiscreteSymbolicSystem):
    """Discrete system with custom output function."""

    def define_system(self):
        x1, x2 = sp.symbols("x1 x2", real=True)
        u = sp.symbols("u", real=True)

        # Dynamics: simple rotation
        self.state_vars = [x1, x2]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([0.9 * x1 - 0.1 * x2 + u, 0.1 * x1 + 0.9 * x2])
        self.parameters = {}
        self.order = 1

        # Output: radius (nonlinear)
        self._h_sym = sp.Matrix([sp.sqrt(x1**2 + x2**2)])


# ============================================================================
# Test System Initialization
# ============================================================================


class TestInitialization:
    """Test proper initialization and validation."""

    def test_basic_initialization(self):
        """Test basic system creation."""
        system = DiscreteLinearSystem()

        assert system._is_discrete is True
        assert system.nx == 2
        assert system.nu == 1
        assert system.ny == 2
        assert system.order == 1

    def test_autonomous_system(self):
        """Test autonomous system (no control)."""
        system = DiscreteLogisticMap()

        assert system.nu == 0
        assert system.nx == 1
        assert len(system.control_vars) == 0

    def test_parameter_assignment(self):
        """Test parameter values are correctly stored."""
        system = DiscreteLinearSystem(a11=0.95, b=2.0)

        # Get symbolic parameter
        a11_sym = sp.symbols("a11", real=True)
        b_sym = sp.symbols("b", real=True)

        assert system.parameters[a11_sym] == 0.95
        assert system.parameters[b_sym] == 2.0

    def test_default_backend(self):
        """Test default backend is numpy."""
        system = DiscreteLinearSystem()
        assert system._default_backend == "numpy"

    def test_repr_str(self):
        """Test string representations."""
        system = DiscreteLinearSystem()

        repr_str = repr(system)
        assert "DiscreteLinearSystem" in repr_str
        assert "discrete=True" in repr_str

        str_str = str(system)
        assert "discrete-time" in str_str


# ============================================================================
# Test Forward Dynamics
# ============================================================================


class TestForwardDynamics:
    """Test forward dynamics evaluation."""

    def test_forward_returns_next_state(self):
        """Test that forward() returns x[k+1] directly."""
        system = DiscreteLinearSystem(a11=0.9, a12=0.0, a21=0.0, a22=0.8, b=0.0)

        x_k = np.array([1.0, 1.0])
        u_k = np.array([0.0])

        x_next = system.forward(x_k, u_k)

        # Should be [0.9, 0.8] (A*x with B*u=0)
        np.testing.assert_allclose(x_next, np.array([0.9, 0.8]), rtol=1e-10)

    def test_forward_with_control(self):
        """Test forward dynamics with control input."""
        system = DiscreteIntegrator(dt=0.1)

        x_k = np.array([1.0])
        u_k = np.array([2.0])

        x_next = system(x_k, u_k)

        # x[k+1] = x[k] + dt*u = 1.0 + 0.1*2.0 = 1.2
        np.testing.assert_allclose(x_next, np.array([1.2]), rtol=1e-10)

    def test_autonomous_forward(self):
        """Test forward dynamics for autonomous system."""
        system = DiscreteLogisticMap(r=3.5)

        x_k = np.array([0.5])
        x_next = system(x_k)  # No control input

        # x[k+1] = 3.5 * 0.5 * (1 - 0.5) = 0.875
        np.testing.assert_allclose(x_next, np.array([0.875]), rtol=1e-10)

    def test_autonomous_with_none_control(self):
        """Test autonomous system accepts u=None explicitly."""
        system = DiscreteLogisticMap(r=3.5)

        x_k = np.array([0.5])
        x_next = system(x_k, u=None)

        np.testing.assert_allclose(x_next, np.array([0.875]), rtol=1e-10)

    def test_batched_forward(self):
        """Test batched forward evaluation."""
        system = DiscreteIntegrator(dt=0.1)

        x_batch = np.array([[1.0], [2.0], [3.0]])
        u_batch = np.array([[1.0], [2.0], [3.0]])

        x_next_batch = system(x_batch, u_batch)

        expected = np.array([[1.1], [2.2], [3.3]])
        np.testing.assert_allclose(x_next_batch, expected, rtol=1e-10)

    def test_callable_interface(self):
        """Test that system is callable."""
        system = DiscreteLinearSystem()

        x = np.array([1.0, 0.0])
        u = np.array([0.0])

        # Should work as callable
        x_next = system(x, u)
        assert x_next.shape == (2,)

    def test_nonlinear_dynamics(self):
        """Test nonlinear discrete dynamics."""
        system = DiscreteLogisticMap(r=2.0)

        # Test trajectory
        x = np.array([0.1])
        trajectory = [x.copy()]

        for _ in range(5):
            x = system(x)
            trajectory.append(x.copy())

        # Should converge to fixed point for r=2.0
        assert len(trajectory) == 6
        # After 5 iterations, should be approaching 0.5
        assert 0.4 < trajectory[-1][0] < 0.6


# ============================================================================
# Test Linearization
# ============================================================================


class TestLinearization:
    """Test linearization capabilities."""

    def test_linearized_dynamics_linear_system(self):
        """Test linearization of linear system returns A, B."""
        system = DiscreteLinearSystem(a11=0.9, a12=0.1, a21=-0.1, a22=0.8, b=1.0)

        x = np.array([0.0, 0.0])
        u = np.array([0.0])

        A, B = system.linearized_dynamics(x, u)

        # For linear system, A and B should be constant
        expected_A = np.array([[0.9, 0.1], [-0.1, 0.8]])
        expected_B = np.array([[0.0], [1.0]])

        np.testing.assert_allclose(A, expected_A, rtol=1e-10)
        np.testing.assert_allclose(B, expected_B, rtol=1e-10)

    def test_linearized_dynamics_nonlinear(self):
        """Test linearization of nonlinear system."""
        system = DiscreteLogisticMap(r=3.5)

        x = np.array([0.5])
        A, B = system.linearized_dynamics(x)

        # A = ∂f/∂x = r*(1 - 2*x) evaluated at x=0.5
        # A = 3.5*(1 - 2*0.5) = 3.5*0 = 0
        np.testing.assert_allclose(A, np.array([[0.0]]), atol=1e-10)

        # B should be empty (0 columns) for autonomous system
        assert B.shape == (1, 0)

    def test_linearized_dynamics_symbolic(self):
        """Test symbolic linearization."""
        system = DiscreteLinearSystem(a11=0.9, a12=0.1, a21=-0.1, a22=0.8, b=1.0)

        x_eq = sp.Matrix([0, 0])
        u_eq = sp.Matrix([0])

        A_sym, B_sym = system.linearized_dynamics_symbolic(x_eq, u_eq)

        # Convert to numpy for comparison
        A_np = np.array(A_sym, dtype=float)
        B_np = np.array(B_sym, dtype=float)

        expected_A = np.array([[0.9, 0.1], [-0.1, 0.8]])
        expected_B = np.array([[0.0], [1.0]])

        np.testing.assert_allclose(A_np, expected_A, rtol=1e-10)
        np.testing.assert_allclose(B_np, expected_B, rtol=1e-10)

    def test_linearization_state_dependent(self):
        """Test linearization at different operating points."""
        system = DiscreteLogisticMap(r=3.5)

        x1 = np.array([0.2])
        x2 = np.array([0.8])

        A1, _ = system.linearized_dynamics(x1)
        A2, _ = system.linearized_dynamics(x2)

        # A should be different at different points
        # A = r*(1 - 2*x)
        # At x=0.2: A = 3.5*(1 - 0.4) = 2.1
        # At x=0.8: A = 3.5*(1 - 1.6) = -2.1
        np.testing.assert_allclose(A1, np.array([[2.1]]), rtol=1e-10)
        np.testing.assert_allclose(A2, np.array([[-2.1]]), rtol=1e-10)


# ============================================================================
# Test Backend Compatibility
# ============================================================================


class TestBackendCompatibility:
    """Test multi-backend support."""

    def test_numpy_backend(self):
        """Test NumPy backend."""
        system = DiscreteLinearSystem()

        x = np.array([1.0, 0.0])
        u = np.array([0.0])

        x_next = system(x, u, backend="numpy")

        assert isinstance(x_next, np.ndarray)
        assert x_next.shape == (2,)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_torch_backend(self):
        """Test PyTorch backend."""
        system = DiscreteLinearSystem()

        x = torch.tensor([1.0, 0.0])
        u = torch.tensor([0.0])

        x_next = system(x, u, backend="torch")

        assert isinstance(x_next, torch.Tensor)
        assert x_next.shape == (2,)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_backend(self):
        """Test JAX backend."""
        system = DiscreteLinearSystem()

        x = jnp.array([1.0, 0.0])
        u = jnp.array([0.0])

        x_next = system(x, u, backend="jax")

        assert isinstance(x_next, jnp.ndarray)
        assert x_next.shape == (2,)

    def test_backend_auto_detection(self):
        """Test automatic backend detection from input type."""
        system = DiscreteLinearSystem()

        x_np = np.array([1.0, 0.0])
        u_np = np.array([0.0])

        x_next = system(x_np, u_np)  # No backend specified
        assert isinstance(x_next, np.ndarray)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_backend_conversion(self):
        """Test backend conversion."""
        system = DiscreteLinearSystem()

        # NumPy input, request torch output
        x = np.array([1.0, 0.0])
        u = np.array([0.0])

        x_next = system(x, u, backend="torch")
        assert isinstance(x_next, torch.Tensor)


# ============================================================================
# Test Output Functions
# ============================================================================


class TestOutputFunctions:
    """Test output function evaluation."""

    def test_identity_output(self):
        """Test default identity output."""
        system = DiscreteLinearSystem()

        x = np.array([1.0, 2.0])
        y = system.h(x)

        # Default output is identity
        np.testing.assert_array_equal(y, x)

    def test_custom_output(self):
        """Test custom output function."""
        system = DiscreteSystemWithOutput()

        x = np.array([3.0, 4.0])
        y = system.h(x)

        # Output should be radius = sqrt(3^2 + 4^2) = 5
        np.testing.assert_allclose(y, np.array([5.0]), rtol=1e-10)

    def test_linearized_observation(self):
        """Test output linearization."""
        system = DiscreteSystemWithOutput()

        x = np.array([3.0, 4.0])
        C = system.linearized_observation(x)

        # C = ∂h/∂x = [x1/r, x2/r] where r = sqrt(x1^2 + x2^2) = 5
        # C = [3/5, 4/5] = [0.6, 0.8]
        expected_C = np.array([[0.6, 0.8]])
        np.testing.assert_allclose(C, expected_C, rtol=1e-10)


# ============================================================================
# Test Equilibrium Management
# ============================================================================


class TestEquilibrium:
    """Test equilibrium point management."""

    def test_add_equilibrium(self):
        """Test adding equilibrium point."""
        system = DiscreteLinearSystem()

        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])

        system.add_equilibrium("origin", x_eq, u_eq, verify=True)

        assert "origin" in system.list_equilibria()

    def test_get_equilibrium(self):
        """Test retrieving equilibrium."""
        system = DiscreteLinearSystem()

        x_eq = np.array([1.0, 2.0])
        u_eq = np.array([0.5])

        system.add_equilibrium("test", x_eq, u_eq, verify=False)

        x_retrieved, u_retrieved = system.get_equilibrium("test")

        np.testing.assert_array_equal(x_retrieved, x_eq)
        np.testing.assert_array_equal(u_retrieved, u_eq)

    def test_equilibrium_verification_pass(self):
        """Test equilibrium verification for valid equilibrium."""
        system = DiscreteLinearSystem()

        # For this system, origin is an equilibrium
        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])

        # Should not raise warning
        system.add_equilibrium("origin", x_eq, u_eq, verify=True, tol=1e-6)

    def test_equilibrium_verification_fail(self):
        """Test equilibrium verification for non-equilibrium."""
        system = DiscreteLinearSystem()

        # This is NOT an equilibrium
        x_eq = np.array([1.0, 0.0])
        u_eq = np.array([0.0])

        # Should issue warning but still add
        with pytest.warns(UserWarning):
            system.add_equilibrium("not_eq", x_eq, u_eq, verify=True, tol=1e-6)

    def test_linearize_at_equilibrium_name(self):
        """Test linearization using equilibrium name."""
        system = DiscreteLinearSystem()

        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])
        system.add_equilibrium("test_eq", x_eq, u_eq, verify=False)

        # Linearize using name
        A, B = system.linearized_dynamics("test_eq")

        expected_A = np.array([[0.9, 0.1], [-0.1, 0.8]])
        expected_B = np.array([[0.0], [1.0]])

        np.testing.assert_allclose(A, expected_A, rtol=1e-10)
        np.testing.assert_allclose(B, expected_B, rtol=1e-10)


# ============================================================================
# Test Error Handling
# ============================================================================


class TestErrorHandling:
    """Test proper error handling."""

    def test_wrong_state_dimension(self):
        """Test error for wrong state dimension."""
        system = DiscreteLinearSystem()

        x_wrong = np.array([1.0])  # Should be 2D
        u = np.array([0.0])

        with pytest.raises((ValueError, IndexError)):
            system(x_wrong, u)

    def test_wrong_control_dimension(self):
        """Test error for wrong control dimension."""
        system = DiscreteLinearSystem()

        x = np.array([1.0, 0.0])
        u_wrong = np.array([0.0, 0.0])  # Should be 1D

        with pytest.raises((ValueError, IndexError)):
            system(x, u_wrong)

    def test_autonomous_with_control_error(self):
        """Test error when providing control to autonomous system."""
        system = DiscreteLogisticMap()

        x = np.array([0.5])
        u = np.array([1.0])  # Autonomous system shouldn't take control

        with pytest.raises(ValueError):
            system(x, u)

    def test_controlled_without_control_error(self):
        """Test error when omitting control for non-autonomous system."""
        system = DiscreteLinearSystem()

        x = np.array([1.0, 0.0])

        with pytest.raises(ValueError):
            system(x)  # Missing required control input


# ============================================================================
# Test Verification Against Known Solutions
# ============================================================================


class TestVerification:
    """Test against known analytical solutions."""

    def test_discrete_integrator_trajectory(self):
        """Test discrete integrator produces correct trajectory."""
        system = DiscreteIntegrator(dt=0.1)

        x = np.array([0.0])
        u = np.array([1.0])  # Constant control

        # Simulate 10 steps
        trajectory = [x.copy()]
        for _ in range(10):
            x = system(x, u)
            trajectory.append(x.copy())

        # x[k] = x[0] + k*dt*u = 0 + k*0.1*1.0 = 0.1*k
        expected = np.array([0.1 * i for i in range(11)]).reshape(-1, 1)
        actual = np.array(trajectory)

        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_logistic_map_fixed_point(self):
        """Test logistic map converges to known fixed point."""
        # For r=2.0, fixed point is (r-1)/r = 1/2 = 0.5
        system = DiscreteLogisticMap(r=2.0)

        x = np.array([0.1])

        # Iterate many times
        for _ in range(100):
            x = system(x)

        # Should converge to 0.5
        np.testing.assert_allclose(x, np.array([0.5]), rtol=1e-6)

    def test_double_integrator_trajectory(self):
        """Test double integrator with constant acceleration."""
        system = DiscreteDoubleIntegrator(dt=0.1)

        # Start at rest at origin
        x = np.array([0.0, 0.0])  # [position, velocity]
        a = np.array([1.0])  # Constant acceleration

        # Simulate 10 steps
        positions = []
        for _ in range(10):
            positions.append(x[0])
            x = system(x, a)

        # Position should follow: p[k] = 0.5*a*dt^2*k^2 approximately
        # But with discrete Euler: p[k] = sum(dt*v[i]) where v[k] = k*dt*a
        # This gives p[k] = dt*a * sum(i) = dt*a * k*(k-1)/2
        # For dt=0.1, a=1.0: p[k] = 0.1 * k*(k-1)/2 = 0.05*k*(k-1)

        # Just verify it's monotonically increasing
        positions = np.array(positions)
        assert np.all(np.diff(positions) >= 0)


# ============================================================================
# Test Comparison with Continuous Systems
# ============================================================================


class TestContinuousComparison:
    """Test differences from continuous systems."""

    def test_discrete_flag_set(self):
        """Test that discrete flag is properly set."""
        discrete_system = DiscreteLinearSystem()

        assert hasattr(discrete_system, "_is_discrete")
        assert discrete_system._is_discrete is True

    def test_forward_semantics_difference(self):
        """Test that forward returns next state, not derivative."""
        system = DiscreteIntegrator(dt=0.1)

        x = np.array([1.0])
        u = np.array([1.0])

        result = system.forward(x, u)

        # For discrete: result should be x[k+1] = 1.1
        # For continuous: result would be dx/dt = 1.0
        np.testing.assert_allclose(result, np.array([1.1]), rtol=1e-10)

        # Verify it's the next state, not a rate
        assert result > x  # Next state should be different from current


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_state(self):
        """Test with zero state."""
        system = DiscreteLinearSystem()

        x = np.array([0.0, 0.0])
        u = np.array([0.0])

        x_next = system(x, u)
        np.testing.assert_array_equal(x_next, np.array([0.0, 0.0]))

    def test_large_values(self):
        """Test with large state values."""
        system = DiscreteLinearSystem()

        x = np.array([1e6, 1e6])
        u = np.array([1e6])

        x_next = system(x, u)

        # Should not overflow or produce NaN
        assert not np.any(np.isnan(x_next))
        assert not np.any(np.isinf(x_next))

    def test_negative_values(self):
        """Test with negative values."""
        system = DiscreteIntegrator(dt=0.1)

        x = np.array([1.0])
        u = np.array([-10.0])

        x_next = system(x, u)

        # x[k+1] = 1.0 + 0.1*(-10) = 0.0
        np.testing.assert_allclose(x_next, np.array([0.0]), rtol=1e-10)

    def test_single_state_system(self):
        """Test 1D system."""
        system = DiscreteLogisticMap()

        x = np.array([0.5])
        x_next = system(x)

        assert x_next.shape == (1,)

    def test_empty_batch(self):
        """Test with empty batch - should fail gracefully with clear error."""
        system = DiscreteLinearSystem()

        x_batch = np.array([]).reshape(0, 2)
        u_batch = np.array([]).reshape(0, 1)

        # Empty batches are not supported and should raise clear, helpful error
        with pytest.raises(
            ValueError, match=r"Empty batch detected.*batch_size=0.*at least one sample"
        ):
            x_next_batch = system(x_batch, u_batch)


# ============================================================================
# Test Configuration and Utilities
# ============================================================================


class TestConfiguration:
    """Test configuration and utility methods."""

    def test_print_equations(self, capsys):
        """Test equation printing."""
        system = DiscreteLinearSystem()

        system.print_equations()

        captured = capsys.readouterr()
        assert "x[k+1]" in captured.out
        assert "Discrete-Time" in captured.out

    def test_get_config_dict(self):
        """Test configuration dictionary."""
        system = DiscreteLinearSystem()

        config = system.get_config_dict()

        assert config["is_discrete"] is True
        assert config["nx"] == 2
        assert config["nu"] == 1
        assert config["class_name"] == "DiscreteLinearSystem"

    def test_compile(self):
        """Test compilation for all backends."""
        system = DiscreteLinearSystem()

        timings = system.compile(backends=["numpy"], verbose=False)

        assert "numpy" in timings
        assert timings["numpy"] is not None


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self):
        """Test complete workflow: create, evaluate, linearize."""
        # Create system
        system = DiscreteLinearSystem(a11=0.95, a12=0.05, a21=-0.05, a22=0.9, b=1.0)

        # Add equilibrium
        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])
        system.add_equilibrium("origin", x_eq, u_eq)

        # Evaluate dynamics
        x = np.array([1.0, 0.0])
        u = np.array([0.5])
        x_next = system(x, u)
        assert x_next.shape == (2,)

        # Linearize
        A, B = system.linearized_dynamics(x, u)
        assert A.shape == (2, 2)
        assert B.shape == (2, 1)

        # Evaluate output
        y = system.h(x)
        assert y.shape == (2,)

    def test_multi_step_simulation(self):
        """Test multi-step simulation."""
        system = DiscreteDoubleIntegrator(dt=0.1)

        x = np.array([0.0, 0.0])
        a = np.array([1.0])

        trajectory = [x.copy()]
        for _ in range(20):
            x = system(x, a)
            trajectory.append(x.copy())

        trajectory = np.array(trajectory)

        # Verify trajectory properties
        assert trajectory.shape == (21, 2)

        # Position should be increasing
        assert np.all(np.diff(trajectory[:, 0]) >= 0)

        # Velocity should be increasing (constant acceleration)
        assert np.all(np.diff(trajectory[:, 1]) >= -1e-10)  # Allow small numerical errors

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_backend_switching(self):
        """Test switching between backends."""
        system = DiscreteLinearSystem()

        x_np = np.array([1.0, 0.0])
        u_np = np.array([0.0])

        # NumPy evaluation
        x_next_np = system(x_np, u_np, backend="numpy")

        # PyTorch evaluation
        x_torch = torch.tensor([1.0, 0.0])
        u_torch = torch.tensor([0.0])
        x_next_torch = system(x_torch, u_torch, backend="torch")

        # Results should match
        np.testing.assert_allclose(x_next_np, x_next_torch.numpy(), rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
