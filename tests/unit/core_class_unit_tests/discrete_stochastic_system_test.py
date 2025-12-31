# Location: tests/systems/test_discrete_stochastic_system.py

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
Comprehensive Unit Tests for DiscreteStochasticSystem
======================================================

Test suite covering:
1. System initialization and validation
2. Deterministic and stochastic evaluation
3. Noise type detection and classification
4. Linearization with diffusion
5. Monte Carlo simulation
6. Backend compatibility (NumPy, PyTorch, JAX)
7. Batched operations
8. Edge cases and error handling
9. Performance and optimization
10. Integration with existing framework

Authors
-------
Gil Benezer

License
-------
AGPL-3.0
"""

import unittest
from typing import Tuple

import numpy as np
import sympy as sp

from src.systems.discrete_stochastic_system import DiscreteStochasticSystem
from src.systems.base.utils.stochastic.sde_validator import ValidationError
from src.systems.base.utils.stochastic.noise_analysis import NoiseType, SDEType


# ============================================================================
# Test System Definitions
# ============================================================================


class DiscreteOU(DiscreteStochasticSystem):
    """Discrete-time Ornstein-Uhlenbeck process (AR(1) with additive noise)."""

    def define_system(self, alpha=1.0, sigma=0.5, dt=0.1):
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        alpha_sym = sp.symbols('alpha', positive=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        dt_sym = sp.symbols('dt', positive=True)

        # Deterministic: x[k+1] = (1 - α*dt)*x[k] + u[k]
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([(1 - alpha_sym * dt_sym) * x + u])
        self.parameters = {alpha_sym: alpha, sigma_sym: sigma, dt_sym: dt}
        self._dt = dt
        self.order = 1

        # Stochastic: additive noise
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = 'ito'


class GeometricRandomWalk(DiscreteStochasticSystem):
    """Discrete-time geometric random walk (multiplicative noise)."""

    def define_system(self, mu=0.1, sigma=0.2, dt=1.0):
        x = sp.symbols('x', positive=True)
        u = sp.symbols('u', real=True)
        mu_sym = sp.symbols('mu', real=True)
        sigma_sym = sp.symbols('sigma', positive=True)

        # Deterministic: x[k+1] = (1 + μ)*x[k] + u[k]
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([(1 + mu_sym) * x + u])
        self.parameters = {mu_sym: mu, sigma_sym: sigma}
        self._dt = dt
        self.order = 1

        # Stochastic: multiplicative noise
        self.diffusion_expr = sp.Matrix([[sigma_sym * x]])
        self.sde_type = 'ito'


class MultiDimensionalStochastic(DiscreteStochasticSystem):
    """2D system with coupled noise."""

    def define_system(self, a=0.9, b=0.1, sigma1=0.3, sigma2=0.2, dt=0.1):
        x1, x2 = sp.symbols('x1 x2', real=True)
        u = sp.symbols('u', real=True)
        a_sym, b_sym = sp.symbols('a b', real=True)
        sigma1_sym, sigma2_sym = sp.symbols('sigma1 sigma2', positive=True)

        # Deterministic: linear coupling
        self.state_vars = [x1, x2]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([
            a_sym * x1 + b_sym * x2,
            -b_sym * x1 + a_sym * x2 + u
        ])
        self.parameters = {
            a_sym: a, b_sym: b,
            sigma1_sym: sigma1, sigma2_sym: sigma2
        }
        self._dt = dt
        self.order = 1

        # Stochastic: diagonal noise
        self.diffusion_expr = sp.Matrix([
            [sigma1_sym, 0],
            [0, sigma2_sym]
        ])
        self.sde_type = 'ito'


class AutonomousStochastic(DiscreteStochasticSystem):
    """Autonomous system (no control input)."""

    def define_system(self, alpha=0.95, sigma=0.1, dt=0.1):
        x = sp.symbols('x', real=True)
        alpha_sym = sp.symbols('alpha', positive=True)
        sigma_sym = sp.symbols('sigma', positive=True)

        # Autonomous: no control
        self.state_vars = [x]
        self.control_vars = []  # No control!
        self._f_sym = sp.Matrix([alpha_sym * x])
        self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
        self._dt = dt
        self.order = 1

        # Additive noise
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = 'ito'


class InvalidDiffusionSystem(DiscreteStochasticSystem):
    """System that doesn't set diffusion_expr (should fail)."""

    def define_system(self, dt=0.1):
        x = sp.symbols('x')
        self.state_vars = [x]
        self.control_vars = []
        self._f_sym = sp.Matrix([0.9 * x])
        self.parameters = {}
        self._dt = dt
        self.order = 1
        # Missing: self.diffusion_expr!


class WrongDimensionDiffusion(DiscreteStochasticSystem):
    """System with diffusion dimension mismatch."""

    def define_system(self, dt=0.1):
        x1, x2 = sp.symbols('x1 x2')
        sigma = sp.symbols('sigma', positive=True)

        self.state_vars = [x1, x2]  # nx=2
        self.control_vars = []
        self._f_sym = sp.Matrix([0.9 * x1, 0.9 * x2])
        self.parameters = {sigma: 0.3}
        self._dt = dt
        self.order = 1

        # Wrong: diffusion has 1 row but should have 2
        self.diffusion_expr = sp.Matrix([[sigma]])  # Should be (2, nw)!
        self.sde_type = 'ito'


# ============================================================================
# Test Suite
# ============================================================================


class TestDiscreteStochasticSystemInitialization(unittest.TestCase):
    """Test system initialization and validation."""

    def test_basic_initialization(self):
        """Test basic system creation."""
        system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)

        # Check dimensions
        self.assertEqual(system.nx, 1)
        self.assertEqual(system.nu, 1)
        self.assertEqual(system.nw, 1)
        self.assertEqual(system.dt, 0.1)

        # Check stochastic flag
        self.assertTrue(system.is_stochastic)

        # Check components exist
        self.assertIsNotNone(system.diffusion_handler)
        self.assertIsNotNone(system.noise_characteristics)

    def test_multidimensional_initialization(self):
        """Test 2D system with diagonal noise."""
        system = MultiDimensionalStochastic(dt=0.05)

        self.assertEqual(system.nx, 2)
        self.assertEqual(system.nu, 1)
        self.assertEqual(system.nw, 2)
        self.assertEqual(system.dt, 0.05)

    def test_autonomous_initialization(self):
        """Test autonomous system (nu=0)."""
        system = AutonomousStochastic(dt=0.1)

        self.assertEqual(system.nx, 1)
        self.assertEqual(system.nu, 0)  # Autonomous!
        self.assertEqual(system.nw, 1)

    def test_missing_diffusion_expr(self):
        """Test that missing diffusion_expr raises error."""
        with self.assertRaises(ValueError) as cm:
            system = InvalidDiffusionSystem(dt=0.1)

        self.assertIn("must set self.diffusion_expr", str(cm.exception))

    def test_wrong_dimension_diffusion(self):
        """Test that dimension mismatch in diffusion raises error."""
        with self.assertRaises(ValidationError) as cm:
            system = WrongDimensionDiffusion(dt=0.1)

        # Should catch dimension mismatch during validation
        self.assertIn("dimension", str(cm.exception).lower())

    def test_sde_type_normalization(self):
        """Test that sde_type string is normalized to enum."""
        system = DiscreteOU(dt=0.1)

        # Should be converted to enum
        self.assertIsInstance(system.sde_type, SDEType)
        self.assertEqual(system.sde_type, SDEType.ITO)


class TestNoiseCharacterization(unittest.TestCase):
    """Test automatic noise type detection."""

    def test_additive_noise_detection(self):
        """Test additive noise is correctly detected."""
        system = DiscreteOU(sigma=0.3, dt=0.1)

        # Noise characteristics
        self.assertTrue(system.is_additive_noise())
        self.assertFalse(system.is_multiplicative_noise())
        self.assertEqual(system.get_noise_type(), NoiseType.ADDITIVE)

        # Dependencies
        self.assertFalse(system.depends_on_state())
        self.assertFalse(system.depends_on_control())
        self.assertFalse(system.depends_on_time())

    def test_multiplicative_noise_detection(self):
        """Test multiplicative noise is correctly detected."""
        system = GeometricRandomWalk(mu=0.1, sigma=0.2, dt=1.0)

        # Noise characteristics
        self.assertFalse(system.is_additive_noise())
        self.assertTrue(system.is_multiplicative_noise())
        self.assertEqual(system.get_noise_type(), NoiseType.MULTIPLICATIVE)

        # Dependencies
        self.assertTrue(system.depends_on_state())
        self.assertFalse(system.depends_on_control())

    def test_diagonal_noise_detection(self):
        """Test diagonal noise structure."""
        system = MultiDimensionalStochastic(dt=0.1)

        # Should detect diagonal structure
        self.assertTrue(system.is_diagonal_noise())

    def test_scalar_noise_detection(self):
        """Test scalar noise (nw=1)."""
        system = DiscreteOU(dt=0.1)

        self.assertTrue(system.is_scalar_noise())
        self.assertEqual(system.nw, 1)


class TestDeterministicEvaluation(unittest.TestCase):
    """Test deterministic part evaluation."""

    def setUp(self):
        """Set up test systems."""
        self.system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)

    def test_call_single_state(self):
        """Test __call__ with single state."""
        x_k = np.array([1.0])
        u_k = np.array([0.0])

        f = self.system(x_k, u_k)

        # Expected: (1 - 2.0*0.1)*1.0 + 0.0 = 0.8
        self.assertAlmostEqual(f[0], 0.8, places=10)

    def test_call_batched(self):
        """Test __call__ with batched inputs."""
        x_batch = np.array([[1.0], [2.0], [3.0]])
        u_batch = np.array([[0.0], [0.5], [1.0]])

        f = self.system(x_batch, u_batch)

        # Check shape
        self.assertEqual(f.shape, (3, 1))

        # Check values
        # f[0] = 0.8*1.0 + 0.0 = 0.8
        # f[1] = 0.8*2.0 + 0.5 = 2.1
        # f[2] = 0.8*3.0 + 1.0 = 3.4
        np.testing.assert_allclose(f[:, 0], [0.8, 2.1, 3.4], rtol=1e-10)

    def test_call_autonomous(self):
        """Test autonomous system (nu=0)."""
        system = AutonomousStochastic(alpha=0.95, dt=0.1)
        x_k = np.array([1.0])

        # Should work without u
        f = system(x_k)

        self.assertAlmostEqual(f[0], 0.95, places=10)


class TestDiffusionEvaluation(unittest.TestCase):
    """Test stochastic part (diffusion) evaluation."""

    def setUp(self):
        """Set up test systems."""
        self.additive_system = DiscreteOU(sigma=0.3, dt=0.1)
        self.multiplicative_system = GeometricRandomWalk(sigma=0.2, dt=1.0)

    def test_diffusion_additive(self):
        """Test diffusion evaluation for additive noise."""
        x_k = np.array([1.0])
        u_k = np.array([0.0])

        g = self.additive_system.diffusion(x_k, u_k)

        # Should be constant
        self.assertEqual(g.shape, (1, 1))
        self.assertAlmostEqual(g[0, 0], 0.3, places=10)

    def test_diffusion_multiplicative(self):
        """Test diffusion evaluation for multiplicative noise."""
        x_k = np.array([2.0])
        u_k = np.array([0.0])

        g = self.multiplicative_system.diffusion(x_k, u_k)

        # Should be state-dependent: σ*x = 0.2*2.0 = 0.4
        self.assertEqual(g.shape, (1, 1))
        self.assertAlmostEqual(g[0, 0], 0.4, places=10)

    def test_diffusion_batched_additive(self):
        """Test batched diffusion evaluation for additive noise."""
        x_batch = np.array([[1.0], [2.0], [3.0]])
        u_batch = np.array([[0.0], [0.0], [0.0]])

        g = self.additive_system.diffusion(x_batch, u_batch)

        # Additive: should return (nx, nw) constant
        self.assertEqual(g.shape, (1, 1))
        self.assertAlmostEqual(g[0, 0], 0.3, places=10)

    def test_diffusion_batched_multiplicative(self):
        """Test batched diffusion for multiplicative noise."""
        x_batch = np.array([[1.0], [2.0], [3.0]])
        u_batch = np.array([[0.0], [0.0], [0.0]])

        g = self.multiplicative_system.diffusion(x_batch, u_batch)

        # Multiplicative: should return (batch, nx, nw)
        self.assertEqual(g.shape, (3, 1, 1))

        # Check values: σ*x for each sample
        np.testing.assert_allclose(
            g[:, 0, 0],
            np.array([0.2, 0.4, 0.6]),  # 0.2*[1, 2, 3]
            rtol=1e-10
        )


class TestStochasticStep(unittest.TestCase):
    """Test full stochastic step evaluation."""

    def setUp(self):
        """Set up test systems."""
        self.system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)

    def test_step_stochastic_with_noise(self):
        """Test stochastic step with provided noise."""
        x_k = np.array([1.0])
        u_k = np.array([0.0])
        w_k = np.array([1.0])  # Unit noise

        x_next = self.system.step_stochastic(x_k, u_k, w_k)

        # Expected: f + g*w = 0.8 + 0.3*1.0 = 1.1
        self.assertAlmostEqual(x_next[0], 1.1, places=10)

    def test_step_stochastic_zero_noise(self):
        """Test that zero noise gives deterministic result."""
        x_k = np.array([1.0])
        u_k = np.array([0.0])
        w_k = np.zeros(1)

        x_next = self.system.step_stochastic(x_k, u_k, w_k)
        f = self.system(x_k, u_k)

        # Should match deterministic part
        np.testing.assert_allclose(x_next, f, rtol=1e-10)

    def test_step_stochastic_auto_noise_generation(self):
        """Test automatic noise generation."""
        np.random.seed(42)
        x_k = np.array([1.0])
        u_k = np.array([0.0])

        x_next = self.system.step_stochastic(x_k, u_k)  # w=None

        # Should generate noise internally
        self.assertEqual(x_next.shape, (1,))
        # Result should be stochastic (not deterministic)
        f = self.system(x_k, u_k)
        self.assertNotEqual(x_next[0], f[0])


class TestLinearization(unittest.TestCase):
    """Test linearization with diffusion."""

    def setUp(self):
        """Set up test system."""
        self.system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)

    def test_linearization_returns_three_matrices(self):
        """Test that stochastic linearization returns (Ad, Bd, Gd)."""
        x_eq = np.zeros(1)
        u_eq = np.zeros(1)

        result = self.system.linearize(x_eq, u_eq)

        # Should return 3-tuple
        self.assertEqual(len(result), 3)

        Ad, Bd, Gd = result

        # Check shapes
        self.assertEqual(Ad.shape, (1, 1))
        self.assertEqual(Bd.shape, (1, 1))
        self.assertEqual(Gd.shape, (1, 1))

    def test_linearization_values_additive(self):
        """Test linearization values for additive noise."""
        x_eq = np.zeros(1)
        u_eq = np.zeros(1)

        Ad, Bd, Gd = self.system.linearize(x_eq, u_eq)

        # Ad = ∂f/∂x = (1 - α*dt) = 1 - 2.0*0.1 = 0.8
        self.assertAlmostEqual(Ad[0, 0], 0.8, places=10)

        # Bd = ∂f/∂u = 1.0
        self.assertAlmostEqual(Bd[0, 0], 1.0, places=10)

        # Gd = g(x_eq, u_eq) = σ = 0.3
        self.assertAlmostEqual(Gd[0, 0], 0.3, places=10)


class TestStochasticSimulation(unittest.TestCase):
    """Test simulate_stochastic method."""

    def setUp(self):
        """Set up test system."""
        self.system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)

    def test_simulate_single_path(self):
        """Test single path simulation."""
        x0 = np.array([1.0])
        n_steps = 100

        result = self.system.simulate_stochastic(
            x0=x0,
            u_sequence=None,
            n_steps=n_steps,
            seed=42
        )

        # Check result structure
        self.assertIn('states', result)
        self.assertIn('time_steps', result)
        self.assertIn('dt', result)
        self.assertIn('metadata', result)

        # Check shapes
        self.assertEqual(result['states'].shape, (n_steps + 1, 1))
        self.assertEqual(len(result['time_steps']), n_steps + 1)

        # Check metadata
        self.assertEqual(result['metadata']['n_paths'], 1)
        self.assertEqual(result['metadata']['seed'], 42)

        # Check initial condition
        np.testing.assert_allclose(result['states'][0, :], x0, rtol=1e-10)

    def test_simulate_monte_carlo(self):
        """Test Monte Carlo simulation with multiple paths."""
        x0 = np.array([1.0])
        n_steps = 100
        n_paths = 50

        result = self.system.simulate_stochastic(
            x0=x0,
            u_sequence=None,
            n_steps=n_steps,
            n_paths=n_paths,
            seed=42
        )

        # Check shape: (n_paths, n_steps+1, nx)
        self.assertEqual(result['states'].shape, (n_paths, n_steps + 1, 1))

        # Check metadata
        self.assertEqual(result['metadata']['n_paths'], n_paths)

        # Check all paths start at x0
        for path in range(n_paths):
            np.testing.assert_allclose(
                result['states'][path, 0, :],
                x0,
                rtol=1e-10
            )

    def test_simulate_reproducibility(self):
        """Test that same seed gives same results."""
        x0 = np.array([1.0])
        n_steps = 100

        result1 = self.system.simulate_stochastic(x0, None, n_steps, seed=42)
        result2 = self.system.simulate_stochastic(x0, None, n_steps, seed=42)

        # Should be identical
        np.testing.assert_allclose(
            result1['states'],
            result2['states'],
            rtol=1e-10
        )


class TestConstantNoiseOptimization(unittest.TestCase):
    """Test constant noise precomputation for additive systems."""

    def test_get_constant_noise_additive(self):
        """Test getting constant noise matrix."""
        system = DiscreteOU(sigma=0.3, dt=0.1)

        G = system.get_constant_noise(backend='numpy')

        self.assertEqual(G.shape, (1, 1))
        self.assertAlmostEqual(G[0, 0], 0.3, places=10)

    def test_get_constant_noise_multiplicative_fails(self):
        """Test that multiplicative noise can't use constant optimization."""
        system = GeometricRandomWalk(sigma=0.2, dt=1.0)

        with self.assertRaises(ValueError):
            G = system.get_constant_noise()


class TestBackendCompatibility(unittest.TestCase):
    """Test multi-backend support."""

    def setUp(self):
        """Set up test system."""
        self.system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)
        self.x = np.array([1.0])
        self.u = np.array([0.0])

    def test_numpy_backend(self):
        """Test NumPy backend."""
        f = self.system(self.x, self.u, backend='numpy')
        g = self.system.diffusion(self.x, self.u, backend='numpy')

        self.assertIsInstance(f, np.ndarray)
        self.assertIsInstance(g, np.ndarray)


class TestPrintingAndInfo(unittest.TestCase):
    """Test printing and information methods."""

    def setUp(self):
        """Set up test system."""
        self.system = DiscreteOU(alpha=2.0, sigma=0.3, dt=0.1)

    def test_print_equations(self):
        """Test that print_equations doesn't crash."""
        # Should not raise
        self.system.print_equations(simplify=True)
        self.system.print_equations(simplify=False)

    def test_print_stochastic_info(self):
        """Test that print_stochastic_info doesn't crash."""
        # Should not raise
        self.system.print_stochastic_info()

    def test_get_info_structure(self):
        """Test get_info returns correct structure."""
        info = self.system.get_info()

        # Check required keys
        self.assertIn('system_type', info)
        self.assertIn('is_discrete', info)
        self.assertIn('is_stochastic', info)
        self.assertIn('dimensions', info)
        self.assertIn('noise', info)

        # Check values
        self.assertEqual(info['system_type'], 'DiscreteStochasticSystem')
        self.assertTrue(info['is_discrete'])
        self.assertTrue(info['is_stochastic'])


class TestStatisticalProperties(unittest.TestCase):
    """Test statistical properties of simulations."""

    def test_variance_accumulation(self):
        """Test that variance accumulates correctly for additive noise."""
        system = DiscreteOU(alpha=0.0, sigma=0.3, dt=0.1)  # Pure random walk
        x0 = np.zeros(1)
        n_steps = 100
        n_paths = 500

        result = system.simulate_stochastic(
            x0, None, n_steps, n_paths, seed=42
        )

        # Compute variance at each time step
        variance_traj = result['states'].var(axis=0)[:, 0]

        # For random walk: Var[x[k]] = k * σ²
        expected_variance = np.arange(n_steps + 1) * 0.3**2

        # Check at specific points (allow 30% error due to finite samples)
        for k in [10, 50, 100]:
            self.assertAlmostEqual(
                variance_traj[k] / expected_variance[k],
                1.0,
                delta=0.3
            )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_zero_noise(self):
        """Test system with zero noise."""
        system = DiscreteOU(alpha=2.0, sigma=0.0, dt=0.1)  # σ=0!

        x_k = np.array([1.0])
        u_k = np.array([0.0])

        # Diffusion should be zero
        g = system.diffusion(x_k, u_k)
        np.testing.assert_allclose(g, np.zeros((1, 1)), rtol=1e-10)

    def test_very_small_dt(self):
        """Test system with very small time step."""
        system = DiscreteOU(alpha=2.0, sigma=0.3, dt=1e-6)

        self.assertEqual(system.dt, 1e-6)

        # Should still work
        x_k = np.array([1.0])
        u_k = np.array([0.0])
        f = system(x_k, u_k)
        g = system.diffusion(x_k, u_k)

        # f ≈ (1 - α*dt)*x = (1 - 2e-6)*1 ≈ 1.0
        self.assertAlmostEqual(f[0], 1.0 - 2e-6, places=8)


# ============================================================================
# Test Runner
# ============================================================================


def run_tests(verbosity=2):
    """
    Run all tests with specified verbosity.

    Parameters
    ----------
    verbosity : int
        Verbosity level (0=quiet, 1=normal, 2=verbose)

    Returns
    -------
    unittest.TestResult
        Test results
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestDiscreteStochasticSystemInitialization,
        TestNoiseCharacterization,
        TestDeterministicEvaluation,
        TestDiffusionEvaluation,
        TestStochasticStep,
        TestLinearization,
        TestStochasticSimulation,
        TestConstantNoiseOptimization,
        TestBackendCompatibility,
        TestPrintingAndInfo,
        TestStatisticalProperties,
        TestEdgeCases,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if __name__ == '__main__':
    # Run with verbose output
    result = run_tests(verbosity=2)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED!")
    else:
        print("\n✗ SOME TESTS FAILED")

    print("=" * 70)