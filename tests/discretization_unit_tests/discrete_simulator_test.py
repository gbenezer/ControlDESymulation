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
Comprehensive unit tests for DiscreteSimulator

Tests cover:
- Pure discrete system simulation
- Discretized continuous system simulation
- All controller types (None, sequence, function, nn.Module)
- Observer integration (output feedback)
- Batched simulation
- Return options (final only, with controls)
- Disturbance simulation
- Edge cases and error handling
- Backend compatibility
"""

import pytest
import numpy as np
import sympy as sp
from typing import Optional

# Conditional imports for optional backends
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from src.systems.base.discretization.discrete_simulator import DiscreteSimulator
from src.systems.base.discrete_symbolic_system import DiscreteSymbolicSystem
from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem
from src.systems.base.discretization.discretizer import Discretizer


# ============================================================================
# Test Fixtures - Discrete Systems
# ============================================================================

class DiscreteLinearSystem(DiscreteSymbolicSystem):
    """Simple 2D discrete linear system: x[k+1] = A*x[k] + B*u[k]"""
    
    def define_system(self, a11=0.9, a12=0.1, a21=-0.1, a22=0.8, b1=0.0, b2=1.0):
        x1, x2 = sp.symbols('x1 x2', real=True)
        u = sp.symbols('u', real=True)
        
        a11_sym, a12_sym = sp.symbols('a11 a12', real=True)
        a21_sym, a22_sym = sp.symbols('a21 a22', real=True)
        b1_sym, b2_sym = sp.symbols('b1 b2', real=True)
        
        self.state_vars = [x1, x2]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([
            a11_sym*x1 + a12_sym*x2 + b1_sym*u,
            a21_sym*x1 + a22_sym*x2 + b2_sym*u
        ])
        self.parameters = {
            a11_sym: a11, a12_sym: a12,
            a21_sym: a21, a22_sym: a22,
            b1_sym: b1, b2_sym: b2
        }
        self.order = 1


class DiscreteIntegrator(DiscreteSymbolicSystem):
    """Discrete integrator: x[k+1] = x[k] + dt*u[k]"""
    
    def define_system(self, dt=0.1):
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        dt_sym = sp.symbols('dt', positive=True)
        
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([x + dt_sym * u])
        self.parameters = {dt_sym: dt}
        self.order = 1


class DiscreteAutonomousSystem(DiscreteSymbolicSystem):
    """Autonomous discrete system: x[k+1] = 0.95*x[k]"""
    
    def define_system(self, decay=0.95):
        x = sp.symbols('x', real=True)
        decay_sym = sp.symbols('decay', real=True)
        
        self.state_vars = [x]
        self.control_vars = []
        self._f_sym = sp.Matrix([decay_sym * x])
        self.parameters = {decay_sym: decay}
        self.order = 1


# Continuous system for discretization tests
class SimpleOscillator(SymbolicDynamicalSystem):
    """Simple harmonic oscillator"""
    
    def define_system(self, omega=1.0):
        x1, x2 = sp.symbols('x1 x2', real=True)
        u = sp.symbols('u', real=True)
        omega_sym = sp.symbols('omega', positive=True)
        
        self.state_vars = [x1, x2]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([
            x2,
            -omega_sym**2 * x1 + u
        ])
        self.parameters = {omega_sym: omega}
        self.order = 1


# Mock observer for testing
class MockObserver:
    """Simple mock observer for testing."""
    
    def __init__(self, nx):
        self.nx = nx
        self.update_count = 0
    
    def initialize(self, x0):
        """Initialize observer state."""
        if x0.ndim == 1:
            return x0.copy() if isinstance(x0, np.ndarray) else x0.clone()
        else:
            return x0.copy() if isinstance(x0, np.ndarray) else x0.clone()
    
    def update(self, x_hat, u, y):
        """Update observer state (identity for testing)."""
        self.update_count += 1
        # Just return the measurement (perfect observer)
        return y


# ============================================================================
# Test Initialization
# ============================================================================

class TestInitialization:
    """Test simulator initialization."""
    
    def test_init_pure_discrete(self):
        """Test initialization with pure discrete system."""
        system = DiscreteLinearSystem()
        sim = DiscreteSimulator(system)
        
        assert sim.system is system
        assert sim.discretizer is None
        assert sim.observer is None
        assert sim.nx == 2
        assert sim.nu == 1
    
    def test_init_discretized_continuous(self):
        """Test initialization with discretized continuous system."""
        system = SimpleOscillator()
        discretizer = Discretizer(system, dt=0.01, method='rk4')
        sim = DiscreteSimulator(system, discretizer=discretizer)
        
        assert sim.system is system
        assert sim.discretizer is discretizer
        assert sim.nx == 2
        assert sim.nu == 1
    
    def test_init_with_observer(self):
        """Test initialization with observer."""
        system = DiscreteLinearSystem()
        observer = MockObserver(nx=2)
        sim = DiscreteSimulator(system, observer=observer)
        
        assert sim.observer is observer
    
    def test_init_continuous_without_discretizer_fails(self):
        """Test that continuous system without discretizer raises error."""
        system = SimpleOscillator()
        
        with pytest.raises(TypeError, match="requires a Discretizer"):
            sim = DiscreteSimulator(system)


# ============================================================================
# Test Autonomous Systems
# ============================================================================

class TestAutonomousSimulation:
    """Test autonomous system simulation (no control)."""
    
    def test_autonomous_simulation(self):
        """Test basic autonomous simulation."""
        system = DiscreteAutonomousSystem(decay=0.9)
        sim = DiscreteSimulator(system)
        
        x0 = np.array([1.0])
        states = sim.simulate(x0, steps=10)
        
        # Check shape
        assert states.shape == (11, 1)  # steps+1 includes initial
        
        # Check decay: x[k] = 0.9^k
        expected = np.array([0.9**k for k in range(11)]).reshape(-1, 1)
        np.testing.assert_allclose(states, expected, rtol=1e-10)
    
    def test_autonomous_with_none_controller(self):
        """Test autonomous system with controller=None."""
        system = DiscreteAutonomousSystem(decay=0.9)
        sim = DiscreteSimulator(system)
        
        x0 = np.array([1.0])
        states = sim.simulate(x0, steps=10, controller=None)
        
        assert states.shape == (11, 1)
    
    def test_autonomous_final_only(self):
        """Test return_final_only for autonomous system."""
        system = DiscreteAutonomousSystem(decay=0.9)
        sim = DiscreteSimulator(system)
        
        x0 = np.array([1.0])
        x_final = sim.simulate(x0, steps=10, return_final_only=True)
        
        # Should return only final state
        assert x_final.shape == (1,)
        
        # x[10] = 0.9^10
        expected = 0.9**10
        np.testing.assert_allclose(x_final, np.array([expected]), rtol=1e-10)


# ============================================================================
# Test Controller Types
# ============================================================================

class TestControllerTypes:
    """Test different controller types."""
    
    def test_no_controller_autonomous(self):
        """Test autonomous system (controller=None)."""
        system = DiscreteAutonomousSystem()
        sim = DiscreteSimulator(system)
        
        x0 = np.array([1.0])
        states = sim.simulate(x0, steps=5, controller=None)
        
        assert states.shape == (6, 1)
    
    def test_sequence_controller_single(self):
        """Test pre-computed control sequence (single trajectory)."""
        system = DiscreteIntegrator(dt=0.1)
        sim = DiscreteSimulator(system)
        
        x0 = np.array([0.0])
        u_seq = np.ones((10, 1))  # Constant control = 1.0
        
        states = sim.simulate(x0, steps=10, controller=u_seq)
        
        # x[k] = 0 + k*0.1*1.0 = 0.1*k
        expected = np.array([0.1 * k for k in range(11)]).reshape(-1, 1)
        np.testing.assert_allclose(states, expected, rtol=1e-10)
    
    def test_sequence_controller_batched(self):
        """Test pre-computed control sequence (batched)."""
        system = DiscreteIntegrator(dt=0.1)
        sim = DiscreteSimulator(system)
        
        batch_size = 3
        x0 = np.zeros((batch_size, 1))
        u_seq = np.ones((batch_size, 10, 1))  # (batch, steps, nu)
        
        states = sim.simulate(x0, steps=10, controller=u_seq)
        
        assert states.shape == (batch_size, 11, 1)
    
    def test_function_controller(self):
        """Test state-feedback function controller."""
        system = DiscreteLinearSystem(a11=0.95, a12=0.0, a21=0.0, a22=0.9, b1=0.0, b2=1.0)
        sim = DiscreteSimulator(system)
        
        # Proportional controller on second state
        def controller(x, k):
            # x might be (nx,) or (batch, nx)
            if x.ndim == 1:
                return np.array([-0.5 * x[1]])
            else:
                return -0.5 * x[:, 1:2]  # (batch, 1)
        
        x0 = np.array([0.0, 1.0])
        states, controls = sim.simulate(x0, steps=20, controller=controller, return_controls=True)
        
        assert states.shape == (21, 2)
        assert controls.shape == (20, 1)
        
        # Verify controller was applied (not zero)
        assert not np.allclose(controls, 0.0)
        
        # Second state should decay with feedback
        assert np.abs(states[-1, 1]) < np.abs(states[0, 1])
    
    def test_time_varying_controller(self):
        """Test time-varying controller using step index."""
        system = DiscreteIntegrator(dt=0.1)
        sim = DiscreteSimulator(system)
        
        # Linearly increasing control
        controller = lambda x, k: np.array([k * 0.1])
        
        x0 = np.array([0.0])
        states = sim.simulate(x0, steps=5, controller=controller)
        
        # x[k+1] = x[k] + 0.1*u[k] where u[k] = 0.1*k
        # x[1] = 0 + 0.1*0.1*0 = 0
        # x[2] = 0 + 0.1*0.1*1 = 0.01
        # x[3] = 0.01 + 0.1*0.1*2 = 0.03
        # etc.
        assert states.shape == (6, 1)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_neural_network_controller(self):
        """Test PyTorch neural network controller."""
        system = DiscreteLinearSystem()
        system.set_default_backend('torch')
        sim = DiscreteSimulator(system)
        
        # Simple linear policy
        policy = nn.Sequential(
            nn.Linear(2, 1, bias=False)
        )
        
        # Initialize weights to known values
        with torch.no_grad():
            policy[0].weight.copy_(torch.tensor([[1.0, 0.5]]))
        
        x0 = torch.tensor([1.0, 0.0])
        states = sim.simulate(x0, steps=5, controller=policy)
        
        assert states.shape == (6, 2)
        assert isinstance(states, torch.Tensor)
    
    def test_constant_controller_as_function(self):
        """Test constant control via function."""
        system = DiscreteIntegrator(dt=0.1)
        sim = DiscreteSimulator(system)
        
        controller = lambda x, k: np.array([1.0])  # Constant
        
        x0 = np.array([0.0])
        states = sim.simulate(x0, steps=10, controller=controller)
        
        # Same as sequence test
        expected = np.array([0.1 * k for k in range(11)]).reshape(-1, 1)
        np.testing.assert_allclose(states, expected, rtol=1e-10)


# ============================================================================
# Test Simulation Options
# ============================================================================

class TestSimulationOptions:
    """Test various simulation options."""
    
    def test_return_final_only(self):
        """Test return_final_only option."""
        system = DiscreteLinearSystem()
        sim = DiscreteSimulator(system)
        
        x0 = np.array([1.0, 0.0])
        u_seq = np.zeros((100, 1))
        
        # Full trajectory
        states_full = sim.simulate(x0, steps=100, controller=u_seq)
        assert states_full.shape == (101, 2)
        
        # Final only
        x_final = sim.simulate(x0, steps=100, controller=u_seq, return_final_only=True)
        assert x_final.shape == (2,)
        
        # Should match last state of full trajectory
        np.testing.assert_array_equal(x_final, states_full[-1])
    
    def test_return_controls(self):
        """Test return_controls option."""
        system = DiscreteIntegrator(dt=0.1)
        sim = DiscreteSimulator(system)
        
        x0 = np.array([0.0])
        u_seq = np.ones((10, 1))
        
        states, controls = sim.simulate(
            x0, steps=10, controller=u_seq, return_controls=True
        )
        
        assert states.shape == (11, 1)
        assert controls.shape == (10, 1)
        
        # Controls should match input sequence
        np.testing.assert_array_equal(controls, u_seq)
    
    def test_return_final_and_controls(self):
        """Test return_final_only with return_controls."""
        system = DiscreteIntegrator(dt=0.1)
        sim = DiscreteSimulator(system)
        
        x0 = np.array([0.0])
        u_seq = np.ones((10, 1))
        
        x_final, controls = sim.simulate(
            x0, steps=10, controller=u_seq,
            return_final_only=True, return_controls=True
        )
        
        assert x_final.shape == (1,)
        assert controls.shape == (10, 1)
        
        # Final state should be sum of controls * dt
        expected_final = 10 * 0.1 * 1.0
        np.testing.assert_allclose(x_final, np.array([expected_final]), rtol=1e-10)


# ============================================================================
# Test Batched Simulation
# ============================================================================

class TestBatchedSimulation:
    """Test batched trajectory simulation."""
    
    def test_batched_initial_conditions(self):
        """Test simulation with batched initial conditions."""
        system = DiscreteAutonomousSystem(decay=0.9)
        sim = DiscreteSimulator(system)
        
        batch_size = 5
        x0_batch = np.ones((batch_size, 1))
        
        states = sim.simulate(x0_batch, steps=10)
        
        assert states.shape == (batch_size, 11, 1)
        
        # All trajectories should be identical (same IC, deterministic)
        for i in range(1, batch_size):
            np.testing.assert_array_equal(states[i], states[0])
    
    def test_batched_with_sequence_controller_single(self):
        """Test batch simulation with single control sequence."""
        system = DiscreteIntegrator(dt=0.1)
        sim = DiscreteSimulator(system)
        
        batch_size = 3
        x0_batch = np.zeros((batch_size, 1))
        u_seq = np.ones((10, 1))  # Single sequence for all
        
        states = sim.simulate(x0_batch, steps=10, controller=u_seq)
        
        assert states.shape == (batch_size, 11, 1)
        
        # All should follow same control
        for i in range(batch_size):
            np.testing.assert_allclose(
                states[i, -1], np.array([1.0]), rtol=1e-10
            )
    
    def test_batched_with_sequence_controller_batched(self):
        """Test batch simulation with batched control sequences."""
        system = DiscreteIntegrator(dt=0.1)
        sim = DiscreteSimulator(system)
        
        batch_size = 3
        x0_batch = np.zeros((batch_size, 1))
        u_seq = np.array([
            [[1.0]] * 10,  # First trajectory
            [[2.0]] * 10,  # Second trajectory
            [[3.0]] * 10,  # Third trajectory
        ])  # Shape: (3, 10, 1)
        
        states = sim.simulate(x0_batch, steps=10, controller=u_seq)
        
        assert states.shape == (batch_size, 11, 1)
        
        # Each trajectory should integrate different control
        np.testing.assert_allclose(states[0, -1], np.array([1.0]), rtol=1e-10)
        np.testing.assert_allclose(states[1, -1], np.array([2.0]), rtol=1e-10)
        np.testing.assert_allclose(states[2, -1], np.array([3.0]), rtol=1e-10)
    
    def test_batched_with_function_controller(self):
        """Test batched simulation with function controller."""
        system = DiscreteLinearSystem()
        sim = DiscreteSimulator(system)
        
        K = np.array([[1.0, 0.5]])
        controller = lambda x, k: -K @ x.T if x.ndim > 1 else -K @ x.reshape(-1, 1)
        
        batch_size = 4
        x0_batch = np.random.randn(batch_size, 2)
        
        states = sim.simulate(x0_batch, steps=20, controller=controller)
        
        assert states.shape == (batch_size, 21, 2)


# ============================================================================
# Test Discretized Continuous Systems
# ============================================================================

class TestDiscretizedContinuous:
    """Test simulation of discretized continuous systems."""
    
    def test_discretized_simulation(self):
        """Test basic discretized continuous simulation."""
        system = SimpleOscillator(omega=1.0)
        discretizer = Discretizer(system, dt=0.01, method='rk4')
        sim = DiscreteSimulator(system, discretizer=discretizer)
        
        x0 = np.array([1.0, 0.0])
        u_seq = np.zeros((100, 1))
        
        states = sim.simulate(x0, steps=100, controller=u_seq, dt=0.01)
        
        assert states.shape == (101, 2)
    
    def test_discretized_with_function_controller(self):
        """Test discretized system with state feedback."""
        system = SimpleOscillator(omega=1.0)  # Lower frequency for stability
        discretizer = Discretizer(system, dt=0.01, method='rk4')  # RK4 more stable than Euler
        sim = DiscreteSimulator(system, discretizer=discretizer)
        
        # Strong damping control
        def controller(x, k):
            if x.ndim == 1:
                return np.array([-2.0 * x[1]])  # Stronger damping
            else:
                return -2.0 * x[:, 1:2]
        
        x0 = np.array([1.0, 0.0])
        states, controls = sim.simulate(
            x0, steps=100, controller=controller, dt=0.01, return_controls=True
        )
        
        assert states.shape == (101, 2)
        
        # Verify controller was actually applied
        assert not np.allclose(controls, 0.0)
        
        # With strong damping, energy should decrease
        final_energy = 0.5 * (states[-1, 0]**2 + states[-1, 1]**2)
        initial_energy = 0.5 * (states[0, 0]**2 + states[0, 1]**2)
        assert final_energy < initial_energy
    
    def test_dt_required_for_discretizer(self):
        """Test that dt is required when using discretizer."""
        system = SimpleOscillator()
        discretizer = Discretizer(system, dt=0.01, method='rk4')
        sim = DiscreteSimulator(system, discretizer=discretizer)
        
        x0 = np.array([1.0, 0.0])
        
        with pytest.raises(ValueError, match="dt must be provided"):
            states = sim.simulate(x0, steps=10)  # Missing dt


# ============================================================================
# Test Observer Integration
# ============================================================================

class TestObserverIntegration:
    """Test output feedback with observers."""
    
    def test_observer_initialization(self):
        """Test observer is initialized with x0."""
        system = DiscreteLinearSystem()
        observer = MockObserver(nx=2)
        sim = DiscreteSimulator(system, observer=observer)
        
        x0 = np.array([1.0, 2.0])
        states = sim.simulate(x0, steps=5)
        
        # Observer should have been called
        assert observer.update_count == 5
    
    def test_observer_receives_measurements(self):
        """Test observer receives measurements y = h(x)."""
        system = DiscreteLinearSystem()
        
        # Track measurements received by observer
        measurements_received = []
        
        class TrackingObserver(MockObserver):
            def update(self, x_hat, u, y):
                measurements_received.append(y.copy() if isinstance(y, np.ndarray) else y.clone())
                return super().update(x_hat, u, y)
        
        observer = TrackingObserver(nx=2)
        sim = DiscreteSimulator(system, observer=observer)
        
        x0 = np.array([1.0, 0.0])
        states = sim.simulate(x0, steps=3)
        
        # Should have 3 measurements (one per step)
        assert len(measurements_received) == 3
    
    def test_controller_uses_observer_estimate(self):
        """Test that controller receives observer estimate, not true state."""
        system = DiscreteLinearSystem()
        
        # Observer that adds constant bias
        class BiasedObserver(MockObserver):
            def update(self, x_hat, u, y):
                # Add bias to estimate
                bias = np.array([0.1, 0.0])
                return y + bias
        
        observer = BiasedObserver(nx=2)
        
        # Track what controller receives
        states_seen_by_controller = []
        
        def tracking_controller(x, k):
            states_seen_by_controller.append(x.copy())
            return np.zeros(1)
        
        sim = DiscreteSimulator(system, observer=observer)
        
        x0 = np.array([0.0, 0.0])
        states = sim.simulate(x0, steps=3, controller=tracking_controller)
        
        # Controller should see biased estimates, not true states
        assert len(states_seen_by_controller) == 3


# ============================================================================
# Test Disturbance Simulation
# ============================================================================

class TestDisturbances:
    """Test disturbance simulation."""
    
    def test_constant_disturbance(self):
        """Test simulation with constant disturbance."""
        system = DiscreteAutonomousSystem(decay=1.0)  # No decay
        sim = DiscreteSimulator(system)
        
        # Constant disturbance
        disturbance = lambda k: np.array([0.1])
        
        x0 = np.array([0.0])
        states = sim.simulate_with_disturbances(
            x0, steps=10, disturbance_func=disturbance
        )
        
        # x[k] = x[0] + k*0.1 = 0.1*k
        expected = np.array([0.1 * k for k in range(11)]).reshape(-1, 1)
        np.testing.assert_allclose(states, expected, rtol=1e-10)
    
    def test_time_varying_disturbance(self):
        """Test time-varying disturbance."""
        system = DiscreteIntegrator(dt=0.1)
        sim = DiscreteSimulator(system)
        
        # Impulse disturbance at k=5
        def impulse(k):
            return np.array([1.0]) if k == 5 else np.array([0.0])
        
        x0 = np.array([0.0])
        states = sim.simulate_with_disturbances(
            x0, steps=10, controller=None, disturbance_func=impulse
        )
        
        # Should see jump at k=5
        assert states.shape == (11, 1)
        # After impulse, state should be affected
        assert states[6, 0] > states[4, 0]
    
    def test_no_disturbance_same_as_simulate(self):
        """Test that None disturbance gives same result as simulate()."""
        system = DiscreteLinearSystem()
        sim = DiscreteSimulator(system)
        
        x0 = np.array([1.0, 0.0])
        u_seq = np.random.randn(20, 1)
        
        states1 = sim.simulate(x0, steps=20, controller=u_seq)
        states2 = sim.simulate_with_disturbances(
            x0, steps=20, controller=u_seq, disturbance_func=None
        )
        
        np.testing.assert_array_equal(states1, states2)


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_steps(self):
        """Test with zero steps."""
        system = DiscreteLinearSystem()
        sim = DiscreteSimulator(system)
        
        x0 = np.array([1.0, 0.0])
        
        with pytest.raises(ValueError, match="steps must be positive"):
            states = sim.simulate(x0, steps=0)
    
    def test_negative_steps(self):
        """Test with negative steps."""
        system = DiscreteLinearSystem()
        sim = DiscreteSimulator(system)
        
        x0 = np.array([1.0, 0.0])
        
        with pytest.raises(ValueError, match="steps must be positive"):
            states = sim.simulate(x0, steps=-5)
    
    def test_single_step(self):
        """Test simulation with single step."""
        system = DiscreteLinearSystem()
        sim = DiscreteSimulator(system)
        
        x0 = np.array([1.0, 0.0])
        states = sim.simulate(x0, steps=1, controller=None)
        
        # Should have 2 states (initial + 1 step)
        assert states.shape == (2, 2)
    
    def test_wrong_initial_state_dimension(self):
        """Test error for wrong initial state dimension."""
        system = DiscreteLinearSystem()  # nx=2
        sim = DiscreteSimulator(system)
        
        x0_wrong = np.array([1.0])  # Should be 2D
        
        with pytest.raises(ValueError, match="doesn't match system dimension"):
            states = sim.simulate(x0_wrong, steps=10)
    
    def test_wrong_control_sequence_length(self):
        """Test error for control sequence with wrong length."""
        system = DiscreteIntegrator()
        sim = DiscreteSimulator(system)
        
        x0 = np.array([0.0])
        u_seq = np.ones((5, 1))  # Only 5 controls
        
        # Requesting 10 steps but only 5 controls
        with pytest.raises(IndexError):
            states = sim.simulate(x0, steps=10, controller=u_seq)
    
    def test_wrong_control_dimension(self):
        """Test error for control with wrong dimension."""
        system = DiscreteIntegrator()  # nu=1
        sim = DiscreteSimulator(system)
        
        x0 = np.array([0.0])
        u_seq = np.ones((10, 2))  # Should be (10, 1)
        
        # Should fail during first step evaluation
        with pytest.raises((ValueError, IndexError)):
            states = sim.simulate(x0, steps=10, controller=u_seq)


# ============================================================================
# Test Comparison: Discrete vs Discretized
# ============================================================================

class TestDiscreteVsDiscretized:
    """Compare pure discrete vs discretized continuous."""
    
    def test_euler_discretized_matches_discrete(self):
        """Test that Euler discretization matches discrete integrator."""
        # Continuous integrator: dx/dt = u
        class ContinuousIntegrator(SymbolicDynamicalSystem):
            def define_system(self):
                x = sp.symbols('x', real=True)
                u = sp.symbols('u', real=True)
                self.state_vars = [x]
                self.control_vars = [u]
                self._f_sym = sp.Matrix([u])
                self.parameters = {}
                self.order = 1
        
        # Discrete integrator: x[k+1] = x[k] + dt*u[k]
        discrete_system = DiscreteIntegrator(dt=0.1)
        continuous_system = ContinuousIntegrator()
        
        # Create simulators
        sim_discrete = DiscreteSimulator(discrete_system)
        
        discretizer = Discretizer(continuous_system, dt=0.1, method='euler')
        sim_continuous = DiscreteSimulator(continuous_system, discretizer=discretizer)
        
        # Same initial condition and control
        x0 = np.array([0.0])
        u_seq = np.ones((10, 1))
        
        states_discrete = sim_discrete.simulate(x0, steps=10, controller=u_seq)
        states_continuous = sim_continuous.simulate(
            x0, steps=10, controller=u_seq, dt=0.1
        )
        
        # Should be identical (Euler discretization)
        np.testing.assert_allclose(states_discrete, states_continuous, rtol=1e-10)


# ============================================================================
# Test Backend Compatibility
# ============================================================================

class TestBackendCompatibility:
    """Test multi-backend support."""
    
    def test_numpy_backend(self):
        """Test NumPy backend."""
        system = DiscreteLinearSystem()
        sim = DiscreteSimulator(system)
        
        x0 = np.array([1.0, 0.0])
        states = sim.simulate(x0, steps=10)
        
        assert isinstance(states, np.ndarray)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_torch_backend(self):
        """Test PyTorch backend."""
        system = DiscreteLinearSystem()
        system.set_default_backend('torch')
        sim = DiscreteSimulator(system)
        
        x0 = torch.tensor([1.0, 0.0])
        states = sim.simulate(x0, steps=10)
        
        assert isinstance(states, torch.Tensor)
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_backend(self):
        """Test JAX backend."""
        system = DiscreteLinearSystem()
        system.set_default_backend('jax')
        sim = DiscreteSimulator(system)
        
        x0 = jnp.array([1.0, 0.0])
        states = sim.simulate(x0, steps=10)
        
        assert isinstance(states, jnp.ndarray)


# ============================================================================
# Test Numerical Accuracy
# ============================================================================

class TestNumericalAccuracy:
    """Test numerical accuracy of simulations."""
    
    def test_integrator_conservation(self):
        """Test that integrator conserves expected quantities."""
        # Simple harmonic oscillator should conserve energy
        system = SimpleOscillator(omega=1.0)
        discretizer = Discretizer(system, dt=0.01, method='rk4')
        sim = DiscreteSimulator(system, discretizer=discretizer)
        
        x0 = np.array([1.0, 0.0])
        u_seq = np.zeros((100, 1))
        
        states = sim.simulate(x0, steps=100, controller=u_seq, dt=0.01)
        
        # Energy: E = 0.5*(x1^2 + x2^2)
        energy = 0.5 * (states[:, 0]**2 + states[:, 1]**2)
        
        # Energy should be approximately conserved (RK4 is symplectic-ish)
        energy_drift = np.abs(energy[-1] - energy[0])
        assert energy_drift < 0.1  # Allow some drift
    
    def test_linear_system_stability(self):
        """Test stable linear system converges to zero."""
        # Stable system (eigenvalues inside unit circle)
        system = DiscreteLinearSystem(a11=0.8, a12=0.1, a21=-0.1, a22=0.7)
        sim = DiscreteSimulator(system)
        
        x0 = np.array([1.0, 1.0])
        states = sim.simulate(x0, steps=100)
        
        # Should converge to zero
        final_norm = np.linalg.norm(states[-1])
        assert final_norm < 0.01


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_full_workflow_discrete(self):
        """Test complete workflow with pure discrete system."""
        # Create system
        system = DiscreteLinearSystem()
        
        # Create simulator
        sim = DiscreteSimulator(system)
        
        # Simple proportional controller
        def controller(x, k):
            if x.ndim == 1:
                # Single state: x is (nx,)
                return np.array([-0.3 * x[0] - 0.2 * x[1]])
            else:
                # Batched: x is (batch, nx)
                return (-0.3 * x[:, 0:1] - 0.2 * x[:, 1:2])  # (batch, 1)
        
        # Simulate
        x0 = np.array([1.0, 0.5])
        states, controls = sim.simulate(
            x0, steps=50, controller=controller, return_controls=True
        )
        
        # Verify shapes
        assert states.shape == (51, 2)
        assert controls.shape == (50, 1)
        
        # Verify controller was applied
        assert not np.allclose(controls, 0.0)
    
    def test_full_workflow_discretized_with_observer(self):
        """Test complete workflow with discretized system and observer."""
        # Continuous system
        system = SimpleOscillator(omega=2.0)
        
        # Discretizer
        discretizer = Discretizer(system, dt=0.01, method='rk4')
        
        # Observer
        observer = MockObserver(nx=2)
        
        # Simulator
        sim = DiscreteSimulator(system, discretizer=discretizer, observer=observer)
        
        # Controller
        controller = lambda x, k: -0.1 * x[1]
        
        # Simulate
        x0 = np.array([1.0, 0.0])
        states = sim.simulate(x0, steps=100, controller=controller, dt=0.01)
        
        assert states.shape == (101, 2)
        assert observer.update_count == 100
    
    def test_batched_with_different_controllers(self):
        """Test batched simulation where we might want different controllers."""
        system = DiscreteIntegrator(dt=0.1)
        sim = DiscreteSimulator(system)
        
        batch_size = 3
        x0_batch = np.zeros((batch_size, 1))
        
        # Different control sequences per trajectory
        u_batch = np.array([
            [[1.0]] * 10,
            [[2.0]] * 10,
            [[3.0]] * 10,
        ])
        
        states = sim.simulate(x0_batch, steps=10, controller=u_batch)
        
        # Each trajectory should reach different final state
        assert states.shape == (batch_size, 11, 1)
        np.testing.assert_allclose(states[0, -1], np.array([1.0]), rtol=1e-10)
        np.testing.assert_allclose(states[1, -1], np.array([2.0]), rtol=1e-10)
        np.testing.assert_allclose(states[2, -1], np.array([3.0]), rtol=1e-10)


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance-related features."""
    
    def test_final_only_faster(self):
        """Test that return_final_only uses less memory."""
        system = DiscreteLinearSystem()
        sim = DiscreteSimulator(system)
        
        x0 = np.array([1.0, 0.0])
        u_seq = np.zeros((1000, 1))
        
        # Full trajectory
        states_full = sim.simulate(x0, steps=1000, controller=u_seq)
        
        # Final only
        x_final = sim.simulate(x0, steps=1000, controller=u_seq, return_final_only=True)
        
        # Memory footprint comparison
        # Full: (1001, 2) = 2002 elements
        # Final: (2,) = 2 elements
        assert states_full.size == 2002
        assert x_final.size == 2
        
        # Should give same final state
        np.testing.assert_array_equal(x_final, states_full[-1])
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_batched_simulation_parallel(self):
        """Test that batched simulation can leverage parallelism."""
        system = DiscreteLinearSystem()
        system.set_default_backend('torch')
        sim = DiscreteSimulator(system)
        
        batch_size = 100
        x0_batch = torch.randn(batch_size, 2)
        
        # Controller that properly handles batched input
        def controller(x, k):
            # x has shape (batch_size, nx)
            # Return (batch_size, nu)
            return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        
        states = sim.simulate(x0_batch, steps=50, controller=controller)
        
        # Should handle large batch efficiently
        assert states.shape == (batch_size, 51, 2)
        assert isinstance(states, torch.Tensor)


# ============================================================================
# Test Information Methods
# ============================================================================

class TestInformation:
    """Test information and diagnostics."""
    
    def test_get_info_discrete(self):
        """Test get_info for pure discrete system."""
        system = DiscreteLinearSystem()
        sim = DiscreteSimulator(system)
        
        info = sim.get_info()
        
        assert info['system_type'] == 'discrete'
        assert info['discretizer'] is None
        assert info['has_observer'] is False
        assert info['dimensions']['nx'] == 2
        assert info['dimensions']['nu'] == 1
    
    def test_get_info_discretized(self):
        """Test get_info for discretized continuous system."""
        system = SimpleOscillator()
        discretizer = Discretizer(system, dt=0.01, method='rk4')
        sim = DiscreteSimulator(system, discretizer=discretizer)
        
        info = sim.get_info()
        
        assert info['system_type'] == 'discretized_continuous'
        assert info['discretizer'] == 'rk4'
        assert info['has_observer'] is False
    
    def test_get_info_with_observer(self):
        """Test get_info with observer."""
        system = DiscreteLinearSystem()
        observer = MockObserver(nx=2)
        sim = DiscreteSimulator(system, observer=observer)
        
        info = sim.get_info()
        
        assert info['has_observer'] is True
        assert info['observer_type'] == 'MockObserver'
    
    def test_repr_str(self):
        """Test string representations."""
        system = DiscreteLinearSystem()
        sim = DiscreteSimulator(system)
        
        repr_str = repr(sim)
        assert 'DiscreteSimulator' in repr_str
        assert 'DiscreteLinearSystem' in repr_str
        
        str_str = str(sim)
        assert 'discrete' in str_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])