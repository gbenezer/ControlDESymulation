# Copyright (C) 2025 Gil Benezer
#
# Unit Tests for ContinuousSystemBase
# Place at: tests/unit/core_class_unit_tests/continuous_system_base_test.py

import unittest
from typing import Optional

import numpy as np

from src.types.core import ControlVector, StateVector
from src.types.linearization import ContinuousLinearization
from src.types.trajectories import IntegrationResult, SimulationResult
from src.systems.base.continuous_system_base import ContinuousSystemBase


class SimpleContinuousSystem(ContinuousSystemBase):
    """Concrete implementation for testing: dx/dt = -x + u"""

    def __init__(self, nx=2, nu=1):
        self.nx = nx
        self.nu = nu
        self.ny = nx

    def __call__(
        self, x: StateVector, u: Optional[ControlVector] = None, t: float = 0.0
    ) -> StateVector:
        """dx/dt = -x + u"""
        if u is None:
            u = np.zeros(self.nu)
        if x.ndim == 2:
            u = u if u.ndim == 2 else u.reshape(-1, 1)
        return -x + u if u is not None else -x

    def integrate(
        self, x0: StateVector, u=None, t_span=(0.0, 1.0), method="RK45", **kwargs
    ) -> IntegrationResult:
        """Simple Euler integration returning IntegrationResult."""
        t_start, t_end = t_span
        dt_integrator = kwargs.get("max_step", 0.01)

        t_points = []
        states_list = []
        
        t = t_start
        x = x0.copy()
        nfev = 0
        
        t_points.append(t)
        states_list.append(x.copy())

        while t < t_end:
            if u is None:
                u_val = None
            elif callable(u):
                u_val = u(t)
            else:
                u_val = u

            dxdt = self(x, u_val, t)
            nfev += 1
            
            x = x + dt_integrator * dxdt
            t = min(t + dt_integrator, t_end)
            
            t_points.append(t)
            states_list.append(x.copy())

        # Return IntegrationResult (as dict)
        return {
            "t": np.array(t_points),
            "y": np.array(states_list).T,  # (nx, n_points)
            "success": True,
            "message": "Integration successful",
            "nfev": nfev,
            "njev": 0,
            "nlu": 0,
            "status": 0
        }

    def linearize(
        self, x_eq: StateVector, u_eq: Optional[ControlVector] = None
    ) -> ContinuousLinearization:
        """Linearization: A = -I, B = I"""
        a = -np.eye(self.nx)
        b = np.ones((self.nx, self.nu))
        return (a, b)


class TestContinuousSystemBase(unittest.TestCase):
    """Test suite for ContinuousSystemBase abstract class."""

    def setUp(self):
        """Create test system."""
        self.system = SimpleContinuousSystem(nx=2, nu=1)

    # =========================================================================
    # Test Abstract Class
    # =========================================================================

    def test_cannot_instantiate_abstract_class(self):
        """ContinuousSystemBase cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            system = ContinuousSystemBase()

    def test_concrete_class_instantiation(self):
        """Concrete implementation can be instantiated."""
        system = SimpleContinuousSystem()
        self.assertIsInstance(system, ContinuousSystemBase)

    # =========================================================================
    # Test __call__ Method (Dynamics Evaluation)
    # =========================================================================

    def test_call_with_control(self):
        """Evaluate dynamics with control input."""
        x = np.array([1.0, 2.0])
        u = np.array([0.5])

        dxdt = self.system(x, u)

        self.assertEqual(dxdt.shape, (2,))
        expected = np.array([-0.5, -1.5])
        np.testing.assert_array_almost_equal(dxdt, expected)

    def test_call_without_control(self):
        """Evaluate autonomous dynamics (u=None)."""
        x = np.array([1.0, 2.0])

        dxdt = self.system(x, u=None)

        expected = np.array([-1.0, -2.0])
        np.testing.assert_array_almost_equal(dxdt, expected)

    # =========================================================================
    # Test integrate() Method (Low-level, returns IntegrationResult)
    # =========================================================================

    def test_integrate_returns_integration_result(self):
        """integrate() returns IntegrationResult with solver diagnostics."""
        x0 = np.array([1.0, 1.0])
        result = self.system.integrate(x0, u=None, t_span=(0.0, 1.0))

        # Check it's a dict (IntegrationResult is TypedDict)
        self.assertIsInstance(result, dict)
        
        # Check required IntegrationResult fields
        self.assertIn("t", result)
        self.assertIn("y", result)
        self.assertIn("success", result)
        self.assertIn("message", result)
        self.assertIn("nfev", result)

    def test_integrate_solver_diagnostics(self):
        """integrate() includes solver diagnostic information."""
        x0 = np.array([1.0, 1.0])
        result = self.system.integrate(x0, u=None, t_span=(0.0, 1.0))

        # Check solver diagnostics
        self.assertIsInstance(result["success"], bool)
        self.assertIsInstance(result["nfev"], int)
        self.assertGreater(result["nfev"], 0)

    def test_integrate_adaptive_time_points(self):
        """integrate() returns adaptive time points from solver."""
        x0 = np.array([1.0, 1.0])
        result = self.system.integrate(x0, u=None, t_span=(0.0, 1.0))

        # Time points are chosen by solver (may not be regular)
        t = result["t"]
        self.assertGreater(len(t), 1)
        self.assertAlmostEqual(t[0], 0.0)
        self.assertAlmostEqual(t[-1], 1.0, places=5)

    # =========================================================================
    # Test simulate() Method (High-level, returns SimulationResult)
    # =========================================================================

    def test_simulate_returns_simulation_result(self):
        """simulate() returns SimulationResult with regular time grid."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, t_span=(0.0, 1.0), dt=0.1)

        # Check it's a dict (SimulationResult is TypedDict)
        self.assertIsInstance(result, dict)
        
        # Check required SimulationResult fields (NOT IntegrationResult fields)
        self.assertIn("time", result)
        self.assertIn("states", result)
        self.assertIn("metadata", result)

    def test_simulate_regular_time_grid(self):
        """simulate() returns states on regular time grid."""
        x0 = np.array([1.0, 1.0])
        dt = 0.1
        result = self.system.simulate(x0, t_span=(0.0, 1.0), dt=dt)

        # Time should be regular grid
        time = result["time"]
        time_diff = np.diff(time)
        np.testing.assert_array_almost_equal(time_diff, dt * np.ones(len(time_diff)))

    def test_simulate_no_solver_diagnostics(self):
        """simulate() hides solver internals (cleaner output)."""
        x0 = np.array([1.0, 1.0])
        result = self.system.simulate(x0, t_span=(0.0, 1.0), dt=0.1)

        # Should NOT have nfev in top level (it's in metadata)
        self.assertNotIn("nfev", result)
        self.assertNotIn("njev", result)
        
        # Metadata may contain solver info
        self.assertIn("metadata", result)

    # =========================================================================
    # Test linearize() Method
    # =========================================================================

    def test_linearize_at_origin(self):
        """Linearize at origin."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)

        lin = self.system.linearize(x_eq, u_eq)

        # Returns tuple
        self.assertIsInstance(lin, tuple)
        self.assertEqual(len(lin), 2)
        
        A, B = lin
        self.assertEqual(A.shape, (2, 2))
        self.assertEqual(B.shape, (2, 1))

    def test_linearize_known_system(self):
        """Linearization of known system matches analytical result."""
        x_eq = np.zeros(2)
        u_eq = np.zeros(1)

        A, B = self.system.linearize(x_eq, u_eq)

        # For dx/dt = -x + u: A = -I, B = I
        expected_A = -np.eye(2)
        expected_B = np.ones((2, 1))

        np.testing.assert_array_almost_equal(A, expected_A)
        np.testing.assert_array_almost_equal(B, expected_B)

    # =========================================================================
    # Test Properties
    # =========================================================================

    def test_is_continuous_property(self):
        """is_continuous returns True."""
        self.assertTrue(self.system.is_continuous)

    def test_is_discrete_property(self):
        """is_discrete returns False."""
        self.assertFalse(self.system.is_discrete)

    def test_is_stochastic_property(self):
        """is_stochastic returns False by default."""
        self.assertFalse(self.system.is_stochastic)

    def test_is_time_varying_property(self):
        """is_time_varying returns False by default."""
        self.assertFalse(self.system.is_time_varying)

    # =========================================================================
    # Test __repr__
    # =========================================================================

    def test_repr(self):
        """String representation contains key info."""
        repr_str = repr(self.system)

        self.assertIn("SimpleContinuousSystem", repr_str)
        self.assertIn("nx=2", repr_str)
        self.assertIn("nu=1", repr_str)


if __name__ == "__main__":
    unittest.main()
