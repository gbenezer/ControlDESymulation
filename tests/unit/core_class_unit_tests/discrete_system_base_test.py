# Copyright (C) 2025 Gil Benezer
#
# Unit Tests for DiscreteSystemBase
# Place at: tests/unit/core_class_unit_tests/discrete_system_base_test.py

import unittest
from typing import Optional, Sequence, Union, Callable

import numpy as np

from src.types.core import ControlVector, StateVector
from src.types.linearization import DiscreteLinearization
from src.systems.base.discrete_system_base import DiscreteSystemBase


class SimpleDiscreteSystem(DiscreteSystemBase):
    """Concrete implementation for testing: x[k+1] = 0.9*x[k] + 0.1*u[k]"""

    def __init__(self, nx=2, nu=1, dt_val=0.1):
        self.nx = nx
        self.nu = nu
        self.ny = nx
        self._dt = dt_val
        self.Ad = 0.9 * np.eye(nx)
        self.Bd = 0.1 * np.ones((nx, nu))

    @property
    def dt(self) -> float:
        return self._dt

    def step(
        self,
        x: StateVector,
        u: Optional[ControlVector] = None,
        k: int = 0
    ) -> StateVector:
        """x[k+1] = Ad*x[k] + Bd*u[k]"""
        if u is None:
            u = np.zeros(self.nu)

        if x.ndim == 2:
            u = u if u.ndim == 2 else u.reshape(-1, 1)
            return self.Ad @ x + self.Bd @ u

        return self.Ad @ x + self.Bd @ u.flatten()

    def simulate(
        self,
        x0: StateVector,
        u_sequence: Optional[Union[ControlVector, Sequence[ControlVector],
                                   Callable[[int], ControlVector]]] = None,
        n_steps: int = 100,
        **kwargs
    ) -> dict:
        """Simulate for n_steps - returns plain dict."""
        states = np.zeros((self.nx, n_steps + 1))
        states[:, 0] = x0

        controls = [] if u_sequence is not None else None

        for k in range(n_steps):
            if u_sequence is None:
                u = None
            elif callable(u_sequence):
                u = u_sequence(k)
            elif isinstance(u_sequence, np.ndarray) and u_sequence.ndim == 1:
                u = u_sequence
            else:
                u = u_sequence[k]

            if controls is not None and u is not None:
                controls.append(u)

            states[:, k + 1] = self.step(states[:, k], u, k)

        controls_array = np.array(controls).T if controls else None

        return {
            "states": states,
            "controls": controls_array,
            "time_steps": np.arange(n_steps + 1),
            "dt": self.dt,
            "metadata": kwargs
        }

    def linearize(
        self,
        x_eq: StateVector,
        u_eq: Optional[ControlVector] = None
    ) -> DiscreteLinearization:
        """Already linear, return (Ad, Bd) tuple."""
        return (self.Ad, self.Bd)


class TestDiscreteSystemBase(unittest.TestCase):
    """Test suite for DiscreteSystemBase abstract class."""

    def setUp(self):
        """Create test system."""
        self.system = SimpleDiscreteSystem(nx=2, nu=1, dt_val=0.1)

    def test_cannot_instantiate_abstract_class(self):
        """DiscreteSystemBase cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            system = DiscreteSystemBase()

    def test_concrete_class_instantiation(self):
        """Concrete implementation can be instantiated."""
        system = SimpleDiscreteSystem()
        self.assertIsInstance(system, DiscreteSystemBase)

    def test_dt_property_exists(self):
        """dt property is accessible."""
        self.assertEqual(self.system.dt, 0.1)

    def test_step_with_control(self):
        """Step forward with control input."""
        x = np.array([1.0, 2.0])
        u = np.array([0.5])

        x_next = self.system.step(x, u)

        self.assertEqual(x_next.shape, (2,))
        expected = np.array([0.95, 1.85])
        np.testing.assert_array_almost_equal(x_next, expected)

    def test_simulate_basic(self):
        """Basic simulation test."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, u_sequence=None, n_steps=10)

        # Check it's a dict (can't use isinstance with TypedDict)
        self.assertIsInstance(result, dict)
        self.assertIn("states", result)
        self.assertIn("time_steps", result)

    def test_simulate_result_structure(self):
        """Simulation result has correct structure."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, n_steps=10)

        # Check all required keys exist
        self.assertIn("states", result)
        self.assertIn("controls", result)
        self.assertIn("time_steps", result)
        self.assertIn("dt", result)
        self.assertIn("metadata", result)

    def test_simulate_constant_control(self):
        """Simulate with constant control."""
        x0 = np.array([1.0, 1.0])
        u = np.array([0.5])

        result = self.system.simulate(x0, u, n_steps=10)

        # Access via dict keys
        self.assertEqual(result["states"].shape[0], 2)  # nx
        self.assertEqual(result["states"].shape[1], 11)  # n_steps + 1

    def test_simulate_time_steps(self):
        """Time steps are correct."""
        x0 = np.array([1.0, 1.0])
        n_steps = 10
        result = self.system.simulate(x0, n_steps=n_steps)

        expected_steps = np.arange(n_steps + 1)
        np.testing.assert_array_equal(result["time_steps"], expected_steps)

    def test_simulate_includes_initial_state(self):
        """Simulation includes initial state x0."""
        x0 = np.array([1.0, 2.0])
        result = self.system.simulate(x0, n_steps=5)

        np.testing.assert_array_equal(result["states"][:, 0], x0)

    def test_simulate_autonomous(self):
        """Simulate autonomous system (no control)."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, u_sequence=None, n_steps=10)

        # Should decay to zero for stable system
        final_state = result["states"][:, -1]
        self.assertLess(np.linalg.norm(final_state), np.linalg.norm(x0))

    def test_simulate_zero_steps(self):
        """Simulate with n_steps=0 returns only initial state."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, n_steps=0)

        self.assertEqual(result["states"].shape[1], 1)
        np.testing.assert_array_equal(result["states"][:, 0], x0)

    def test_linearize_at_origin(self):
        """Linearize at origin."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)

        lin = self.system.linearize(x_eq, u_eq)

        # Check it's a tuple
        self.assertIsInstance(lin, tuple)
        self.assertEqual(len(lin), 2)
        
        Ad, Bd = lin
        self.assertEqual(Ad.shape, (2, 2))
        self.assertEqual(Bd.shape, (2, 1))

    def test_linearize_without_control(self):
        """Linearize with u_eq=None."""
        x_eq = np.zeros(2)

        lin = self.system.linearize(x_eq, u_eq=None)

        Ad, Bd = lin
        self.assertIsNotNone(Ad)
        self.assertIsNotNone(Bd)

    def test_linearize_correct_dimensions(self):
        """Linearization matrices have correct dimensions."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)

        lin = self.system.linearize(x_eq, u_eq)
        Ad, Bd = lin

        # Ad should be (nx, nx), Bd should be (nx, nu)
        self.assertEqual(Ad.shape, (self.system.nx, self.system.nx))
        self.assertEqual(Bd.shape, (self.system.nx, self.system.nu))

    def test_linearize_known_system(self):
        """Linearization of known system matches analytical result."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)

        lin = self.system.linearize(x_eq, u_eq)
        Ad, Bd = lin

        # For x[k+1] = 0.9*x[k] + 0.1*u[k]: Ad = 0.9*I, Bd = 0.1*I
        expected_Ad = 0.9 * np.eye(2)
        expected_Bd = 0.1 * np.ones((2, 1))

        np.testing.assert_array_almost_equal(Ad, expected_Ad)
        np.testing.assert_array_almost_equal(Bd, expected_Bd)

    def test_is_continuous_property(self):
        """is_continuous returns False."""
        self.assertFalse(self.system.is_continuous)

    def test_is_discrete_property(self):
        """is_discrete returns True."""
        self.assertTrue(self.system.is_discrete)

    def test_sampling_frequency(self):
        """Sampling frequency is 1/dt."""
        expected_freq = 1.0 / self.system.dt
        self.assertAlmostEqual(self.system.sampling_frequency, expected_freq)

    def test_repr(self):
        """String representation contains key info."""
        repr_str = repr(self.system)

        self.assertIn('SimpleDiscreteSystem', repr_str)
        self.assertIn('nx=2', repr_str)
        self.assertIn('nu=1', repr_str)
        self.assertIn('dt=0.1', repr_str)

    def test_polymorphic_usage(self):
        """System can be used polymorphically via base class."""
        def check_stability(sys: DiscreteSystemBase):
            """Generic function that works with any discrete system."""
            x_eq = np.zeros(sys.nx)
            u_eq = np.zeros(sys.nu)
            lin = sys.linearize(x_eq, u_eq)
            Ad, Bd = lin  # Unpack tuple
            eigenvalues = np.linalg.eigvals(Ad)
            return np.all(np.abs(eigenvalues) < 1)

        is_stable = check_stability(self.system)
        self.assertTrue(is_stable)

    def test_stability_check(self):
        """System with |eigenvalues| < 1 is stable."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)
        lin = self.system.linearize(x_eq, u_eq)
        
        Ad, Bd = lin
        eigenvalues = np.linalg.eigvals(Ad)
        is_stable = np.all(np.abs(eigenvalues) < 1)

        # 0.9 < 1, so system is stable
        self.assertTrue(is_stable)


if __name__ == '__main__':
    unittest.main()
