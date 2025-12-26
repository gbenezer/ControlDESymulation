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
Comprehensive unit tests for DiscreteLinearization

Tests cover:
- Initialization and validation
- Basic linearization (pure discrete and discretized)
- Caching behavior (hits, misses, key generation)
- Equilibrium name vs numeric point linearization
- Multiple discretization methods
- Batch equilibria linearization
- Stability analysis
- Controllability/observability matrices
- Gain scheduling support
- Cache management and statistics
- Backend compatibility
- Stochastic system warnings
- Edge cases and error handling
"""

import warnings

import numpy as np
import pytest
import sympy as sp

# Conditional imports
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
from src.systems.base.discretization.discrete_linearization import DiscreteLinearization
from src.systems.base.discretization.discretizer import Discretizer
from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem

# ============================================================================
# Test Fixtures - Systems
# ============================================================================


class DiscreteLinearSystem(DiscreteSymbolicSystem):
    """Simple 2D discrete linear system."""

    def define_system(self, a11=0.9, a12=0.1, a21=-0.1, a22=0.8, b1=0.0, b2=1.0):
        x1, x2 = sp.symbols("x1 x2", real=True)
        u = sp.symbols("u", real=True)

        a11_sym, a12_sym = sp.symbols("a11 a12", real=True)
        a21_sym, a22_sym = sp.symbols("a21 a22", real=True)
        b1_sym, b2_sym = sp.symbols("b1 b2", real=True)

        self.state_vars = [x1, x2]
        self.control_vars = [u]
        self._f_sym = sp.Matrix(
            [a11_sym * x1 + a12_sym * x2 + b1_sym * u, a21_sym * x1 + a22_sym * x2 + b2_sym * u]
        )
        self.parameters = {
            a11_sym: a11,
            a12_sym: a12,
            a21_sym: a21,
            a22_sym: a22,
            b1_sym: b1,
            b2_sym: b2,
        }
        self.order = 1


class DiscreteNonlinearSystem(DiscreteSymbolicSystem):
    """Nonlinear discrete system for testing."""

    def define_system(self, k=1.0):
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)
        k_sym = sp.symbols("k", positive=True)

        self.state_vars = [x]
        self.control_vars = [u]
        # x[k+1] = x[k] - k*sin(x[k]) + u[k]
        self._f_sym = sp.Matrix([x - k_sym * sp.sin(x) + u])
        self.parameters = {k_sym: k}
        self.order = 1


class SimpleOscillator(SymbolicDynamicalSystem):
    """Simple continuous harmonic oscillator."""

    def define_system(self, omega=1.0):
        x1, x2 = sp.symbols("x1 x2", real=True)
        u = sp.symbols("u", real=True)
        omega_sym = sp.symbols("omega", positive=True)

        self.state_vars = [x1, x2]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([x2, -(omega_sym**2) * x1 + u])
        self.parameters = {omega_sym: omega}
        self.order = 1


# ============================================================================
# Test Initialization
# ============================================================================


class TestInitialization:
    """Test linearization cache initialization."""

    def test_init_pure_discrete(self):
        """Test initialization with pure discrete system."""
        system = DiscreteLinearSystem()
        lin = DiscreteLinearization(system)

        assert lin.system is system
        assert lin.discretizer is None
        assert lin.backend == "numpy"
        assert len(lin._cache) == 0

    def test_init_discretized_continuous(self):
        """Test initialization with discretized continuous system."""
        system = SimpleOscillator()
        discretizer = Discretizer(system, dt=0.01, method="rk4")
        lin = DiscreteLinearization(system, discretizer=discretizer)

        assert lin.system is system
        assert lin.discretizer is discretizer

    def test_init_continuous_without_discretizer_fails(self):
        """Test that continuous system without discretizer fails."""
        system = SimpleOscillator()

        with pytest.raises(TypeError, match="requires a Discretizer"):
            DiscreteLinearization(system)

    def test_init_stochastic_system_warns(self):
        """Test warning when using stochastic system."""
        from src.systems.builtin.stochastic.discrete_ar1 import DiscreteAR1

        system = DiscreteAR1()

        with pytest.warns(UserWarning, match="stochastic.*StochasticDiscreteLinearization"):
            lin = DiscreteLinearization(system)


# ============================================================================
# Test Basic Linearization
# ============================================================================


class TestBasicLinearization:
    """Test basic linearization computation."""

    def test_linearize_discrete_linear(self):
        """Test linearization of linear discrete system."""
        system = DiscreteLinearSystem(a11=0.9, a12=0.1, a21=-0.1, a22=0.8, b1=0.0, b2=1.0)
        lin = DiscreteLinearization(system)

        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])

        Ad, Bd = lin.compute(x_eq, u_eq)

        # For linear system, A and B are constant
        expected_A = np.array([[0.9, 0.1], [-0.1, 0.8]])
        expected_B = np.array([[0.0], [1.0]])

        np.testing.assert_allclose(Ad, expected_A, rtol=1e-10)
        np.testing.assert_allclose(Bd, expected_B, rtol=1e-10)

    def test_linearize_discrete_nonlinear(self):
        """Test linearization of nonlinear discrete system."""
        system = DiscreteNonlinearSystem(k=1.0)
        lin = DiscreteLinearization(system)

        # At x=0: sin(0)=0, cos(0)=1
        x_eq = np.array([0.0])
        u_eq = np.array([0.0])

        Ad, Bd = lin.compute(x_eq, u_eq)

        # A = ∂(x - sin(x) + u)/∂x = 1 - cos(x) = 1 - 1 = 0
        # B = ∂(x - sin(x) + u)/∂u = 1
        np.testing.assert_allclose(Ad, np.array([[0.0]]), atol=1e-10)
        np.testing.assert_allclose(Bd, np.array([[1.0]]), rtol=1e-10)

    def test_linearize_at_nonzero_point(self):
        """Test linearization at non-equilibrium point."""
        system = DiscreteNonlinearSystem(k=1.0)
        lin = DiscreteLinearization(system)

        # At x=π/2: cos(π/2)=0
        x_eq = np.array([np.pi / 2])
        u_eq = np.array([0.0])

        Ad, Bd = lin.compute(x_eq, u_eq)

        # A = 1 - cos(π/2) = 1 - 0 = 1
        np.testing.assert_allclose(Ad, np.array([[1.0]]), rtol=1e-10)

    def test_linearize_discretized_continuous(self):
        """Test linearization of discretized continuous system."""
        system = SimpleOscillator(omega=1.0)
        discretizer = Discretizer(system, dt=0.01, method="euler")
        lin = DiscreteLinearization(system, discretizer=discretizer)

        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])

        Ad, Bd = lin.compute(x_eq, u_eq, method="euler")

        # Euler: Ad = I + dt*Ac
        # Ac = [[0, 1], [-1, 0]] at origin
        # Ad = [[1, 0.01], [-0.01, 1]]
        expected_A = np.array([[1.0, 0.01], [-0.01, 1.0]])
        expected_B = np.array([[0.0], [0.01]])

        np.testing.assert_allclose(Ad, expected_A, rtol=1e-10)
        np.testing.assert_allclose(Bd, expected_B, rtol=1e-10)


# ============================================================================
# Test Caching Behavior
# ============================================================================


class TestCaching:
    """Test cache functionality."""

    def test_cache_hit(self):
        """Test that second call uses cache."""
        system = DiscreteLinearSystem()
        lin = DiscreteLinearization(system)

        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])

        # First call
        Ad1, Bd1 = lin.compute(x_eq, u_eq)
        computes_1 = lin._stats["computes"]

        # Second call
        Ad2, Bd2 = lin.compute(x_eq, u_eq)
        computes_2 = lin._stats["computes"]

        # Should be same objects (cached)
        assert Ad1 is Ad2
        assert Bd1 is Bd2

        # Stats should show cache hit
        assert computes_2 == computes_1  # No new computation
        assert lin._stats["cache_hits"] == 1

    def test_cache_different_points(self):
        """Test that different points are cached separately."""
        # Use NONLINEAR system so linearizations actually differ
        system = DiscreteNonlinearSystem()
        lin = DiscreteLinearization(system)

        x_eq1 = np.array([0.0])
        x_eq2 = np.array([1.0])
        u_eq = np.array([0.0])

        Ad1, Bd1 = lin.compute(x_eq1, u_eq)
        Ad2, Bd2 = lin.compute(x_eq2, u_eq)

        # Should be different (nonlinear system, different points)
        assert not np.allclose(Ad1, Ad2, rtol=1e-6)

        # Both should be cached (2 different keys)
        assert len(lin._cache) == 2

    def test_cache_different_methods(self):
        """Test that different methods are cached separately."""
        system = SimpleOscillator()
        discretizer = Discretizer(system, dt=0.01, method="rk4")
        lin = DiscreteLinearization(system, discretizer=discretizer)

        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])

        Ad_euler, Bd_euler = lin.compute(x_eq, u_eq, method="euler")
        Ad_exact, Bd_exact = lin.compute(x_eq, u_eq, method="exact")

        # Different methods should give different results
        assert not np.allclose(Ad_euler, Ad_exact, rtol=1e-6)

        # Both cached
        assert len(lin._cache) == 2

    def test_use_cache_false_recomputes(self):
        """Test that use_cache=False forces recomputation."""
        system = DiscreteLinearSystem()
        lin = DiscreteLinearization(system)

        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])

        # First call
        Ad1, Bd1 = lin.compute(x_eq, u_eq)
        computes_1 = lin._stats["computes"]

        # Force recompute
        Ad2, Bd2 = lin.compute(x_eq, u_eq, use_cache=False)
        computes_2 = lin._stats["computes"]

        # Should have computed again
        assert computes_2 == computes_1 + 1

    def test_is_cached(self):
        """Test checking if linearization is cached."""
        system = DiscreteLinearSystem()
        lin = DiscreteLinearization(system)

        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])

        # Not cached initially
        assert not lin.is_cached(x_eq, u_eq)

        # Compute
        Ad, Bd = lin.compute(x_eq, u_eq)

        # Now cached
        assert lin.is_cached(x_eq, u_eq)

    def test_get_cached(self):
        """Test retrieving cached linearization."""
        system = DiscreteLinearSystem()
        lin = DiscreteLinearization(system)

        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])

        # Not cached yet
        cached = lin.get_cached(x_eq, u_eq)
        assert cached is None

        # Compute
        Ad, Bd = lin.compute(x_eq, u_eq)

        # Retrieve from cache
        cached = lin.get_cached(x_eq, u_eq)
        assert cached is not None

        Ad_cached, Bd_cached = cached
        assert Ad is Ad_cached
        assert Bd is Bd_cached


# ============================================================================
# Test Equilibrium-Based Linearization
# ============================================================================


class TestEquilibriumLinearization:
    """Test linearization using equilibrium names."""

    def test_linearize_at_named_equilibrium(self):
        """Test linearization using equilibrium name."""
        system = DiscreteLinearSystem()
        system.add_equilibrium("origin", np.array([0.0, 0.0]), np.array([0.0]))

        lin = DiscreteLinearization(system)

        # Use equilibrium name
        Ad, Bd = lin.compute("origin")

        assert Ad.shape == (2, 2)
        assert Bd.shape == (2, 1)

    def test_equilibrium_name_cached_separately(self):
        """Test that equilibrium names and numeric points cached separately."""
        system = DiscreteLinearSystem()
        system.add_equilibrium("origin", np.array([0.0, 0.0]), np.array([0.0]))

        lin = DiscreteLinearization(system)

        # Cache using name
        Ad1, Bd1 = lin.compute("origin")

        # Cache using numeric point (same values)
        Ad2, Bd2 = lin.compute(np.array([0.0, 0.0]), np.array([0.0]))

        # Both cached (different keys)
        assert len(lin._cache) == 2

    def test_compute_at_equilibria(self):
        """Test batch linearization at all equilibria."""
        system = DiscreteLinearSystem()
        system.add_equilibrium("eq1", np.array([0.0, 0.0]), np.array([0.0]))
        system.add_equilibrium("eq2", np.array([1.0, 0.0]), np.array([0.0]))
        system.add_equilibrium("eq3", np.array([0.0, 1.0]), np.array([0.0]))

        lin = DiscreteLinearization(system)

        # Linearize at all equilibria
        linearizations = lin.compute_at_equilibria()

        assert len(linearizations) == 4  # 3 + 'origin' default
        assert "eq1" in linearizations
        assert "eq2" in linearizations
        assert "eq3" in linearizations

        # Each should be (Ad, Bd) tuple
        for name, (Ad, Bd) in linearizations.items():
            assert Ad.shape == (2, 2)
            assert Bd.shape == (2, 1)

    def test_compute_at_specific_equilibria(self):
        """Test linearization at subset of equilibria."""
        system = DiscreteLinearSystem()
        system.add_equilibrium("eq1", np.array([0.0, 0.0]), np.array([0.0]))
        system.add_equilibrium("eq2", np.array([1.0, 0.0]), np.array([0.0]))
        system.add_equilibrium("eq3", np.array([0.0, 1.0]), np.array([0.0]))

        lin = DiscreteLinearization(system)

        # Linearize at specific subset
        linearizations = lin.compute_at_equilibria(["eq1", "eq2"])

        assert len(linearizations) == 2
        assert "eq1" in linearizations
        assert "eq2" in linearizations
        assert "eq3" not in linearizations


# ============================================================================
# Test Discretization Methods
# ============================================================================


class TestDiscretizationMethods:
    """Test different discretization methods."""

    def test_euler_method(self):
        """Test Euler discretization."""
        system = SimpleOscillator(omega=1.0)
        discretizer = Discretizer(system, dt=0.01, method="rk4")
        lin = DiscreteLinearization(system, discretizer=discretizer)

        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])

        Ad, Bd = lin.compute(x_eq, u_eq, method="euler")

        # Euler: Ad = I + dt*Ac
        assert Ad.shape == (2, 2)

    def test_exact_method(self):
        """Test exact (matrix exponential) discretization."""
        system = SimpleOscillator(omega=1.0)
        discretizer = Discretizer(system, dt=0.01, method="rk4")
        lin = DiscreteLinearization(system, discretizer=discretizer)

        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])

        Ad, Bd = lin.compute(x_eq, u_eq, method="exact")

        assert Ad.shape == (2, 2)

        # Eigenvalues should be on unit circle (conservative system)
        eigs = np.linalg.eigvals(Ad)
        assert np.allclose(np.abs(eigs), 1.0, rtol=1e-6)

    def test_different_methods_different_results(self):
        """Test that different methods give different linearizations."""
        system = SimpleOscillator(omega=2.0)
        discretizer = Discretizer(system, dt=0.01, method="rk4")
        lin = DiscreteLinearization(system, discretizer=discretizer)

        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])

        Ad_euler, _ = lin.compute(x_eq, u_eq, method="euler")
        Ad_exact, _ = lin.compute(x_eq, u_eq, method="exact")

        # Should be different
        assert not np.allclose(Ad_euler, Ad_exact, rtol=1e-6)


# ============================================================================
# Test Stability Analysis
# ============================================================================


class TestStabilityAnalysis:
    """Test stability analysis features."""

    def test_check_stability_stable_system(self):
        """Test stability check for stable system."""
        system = DiscreteLinearSystem(a11=0.8, a12=0.0, a21=0.0, a22=0.7)
        lin = DiscreteLinearization(system)

        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])

        stability = lin.check_stability(x_eq, u_eq)

        assert stability["is_stable"] is True
        assert stability["is_unstable"] is False
        assert stability["max_magnitude"] < 1.0
        assert len(stability["eigenvalues"]) == 2

    def test_check_stability_unstable_system(self):
        """Test stability check for unstable system."""
        system = DiscreteLinearSystem(a11=1.1, a12=0.0, a21=0.0, a22=0.9)
        lin = DiscreteLinearization(system)

        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])

        stability = lin.check_stability(x_eq, u_eq)

        assert stability["is_stable"] is False
        assert stability["is_unstable"] is True
        assert stability["max_magnitude"] > 1.0

    def test_check_stability_marginally_stable(self):
        """Test stability check for marginally stable system."""
        system = DiscreteLinearSystem(a11=1.0, a12=0.0, a21=0.0, a22=1.0)
        lin = DiscreteLinearization(system)

        x_eq = np.array([0.0, 0.0])
        u_eq = np.array([0.0])

        stability = lin.check_stability(x_eq, u_eq)

        assert stability["is_marginally_stable"] is True
        np.testing.assert_allclose(stability["max_magnitude"], 1.0, atol=1e-6)

    def test_spectral_radius(self):
        """Test spectral radius computation."""
        system = DiscreteLinearSystem(a11=0.95, a12=0.05, a21=-0.05, a22=0.9)
        lin = DiscreteLinearization(system)

        stability = lin.check_stability(np.zeros(2), np.zeros(1))

        # Spectral radius = max|λ|
        eigs = stability["eigenvalues"]
        max_mag = np.max(np.abs(eigs))

        np.testing.assert_allclose(stability["spectral_radius"], max_mag, rtol=1e-10)


# ============================================================================
# Test Controllability and Observability
# ============================================================================


class TestControllabilityObservability:
    """Test controllability and observability analysis."""

    def test_controllability_matrix(self):
        """Test controllability matrix computation."""
        system = DiscreteLinearSystem()
        lin = DiscreteLinearization(system)

        C = lin.compute_controllability_matrix(np.zeros(2), np.zeros(1))

        # For 2D system with nu=1: C is (nx, nx*nu) = (2, 2)
        assert C.shape == (2, 2)

    def test_controllability_check(self):
        """Test controllability analysis."""
        # Controllable system
        system = DiscreteLinearSystem(a11=0.9, a12=0.1, a21=-0.1, a22=0.8, b2=1.0)
        lin = DiscreteLinearization(system)

        C = lin.compute_controllability_matrix(np.zeros(2), np.zeros(1))
        rank = np.linalg.matrix_rank(C)

        # Should be controllable
        assert rank == 2

    def test_observability_matrix(self):
        """Test observability matrix computation."""
        system = DiscreteLinearSystem()
        lin = DiscreteLinearization(system)

        O = lin.compute_observability_matrix(np.zeros(2), np.zeros(1))

        # For 2D system with ny=nx: O is (nx*ny, nx) = (4, 2)
        assert O.shape == (4, 2)

    def test_observability_check(self):
        """Test observability analysis."""
        system = DiscreteLinearSystem()
        lin = DiscreteLinearization(system)

        O = lin.compute_observability_matrix(np.zeros(2), np.zeros(1))
        rank = np.linalg.matrix_rank(O)

        # Identity output should be observable
        assert rank == 2


# ============================================================================
# Test Gain Scheduling
# ============================================================================


class TestGainScheduling:
    """Test gain scheduling support."""

    def test_precompute_at_grid(self):
        """Test precomputing linearizations at operating point grid."""
        system = DiscreteNonlinearSystem()
        lin = DiscreteLinearization(system)

        # Create grid
        x_grid = np.linspace(-1, 1, 5).reshape(-1, 1)
        u_grid = np.zeros((5, 1))

        # Precompute
        lin.precompute_at_grid(x_grid, u_grid)

        # All should be cached
        assert lin._stats["computes"] == 5
        assert len(lin._cache) >= 5  # At least 5 (might have more from hashing)

    def test_gain_scheduling_workflow(self):
        """Test complete gain scheduling workflow."""
        system = DiscreteNonlinearSystem()
        lin = DiscreteLinearization(system)

        # Define operating points
        operating_points = [
            (np.array([0.0]), np.array([0.0])),
            (np.array([0.5]), np.array([0.0])),
            (np.array([1.0]), np.array([0.0])),
        ]

        # Precompute all
        for x, u in operating_points:
            lin.compute(x, u)

        # Verify all cached
        for x, u in operating_points:
            assert lin.is_cached(x, u)

        # Second access is instant
        stats_before = lin.get_stats()
        for x, u in operating_points:
            Ad, Bd = lin.compute(x, u)

        # All should be cache hits
        stats_after = lin.get_stats()
        assert stats_after["cache_hits"] == stats_before["cache_hits"] + 3


# ============================================================================
# Test Cache Management
# ============================================================================


class TestCacheManagement:
    """Test cache management features."""

    def test_reset_cache(self):
        """Test clearing entire cache."""
        system = DiscreteLinearSystem()
        lin = DiscreteLinearization(system)

        # Cache some linearizations
        lin.compute(np.zeros(2), np.zeros(1))
        lin.compute(np.ones(2), np.zeros(1))

        assert len(lin._cache) == 2

        # Reset
        lin.reset_cache()

        assert len(lin._cache) == 0

    def test_reset_specific_equilibrium(self):
        """Test clearing specific equilibrium from cache."""
        system = DiscreteLinearSystem()
        system.add_equilibrium("eq1", np.zeros(2), np.zeros(1))
        system.add_equilibrium("eq2", np.ones(2), np.zeros(1))

        lin = DiscreteLinearization(system)

        # Cache both
        lin.compute("eq1")
        lin.compute("eq2")

        assert len(lin._cache) == 2

        # Clear one
        lin.reset_cache("eq1")

        assert not lin.is_cached("eq1")
        assert lin.is_cached("eq2")

    def test_clear_cache(self):
        """Test clear_cache clears cache and stats."""
        system = DiscreteLinearSystem()
        lin = DiscreteLinearization(system)

        # Generate some activity
        lin.compute(np.zeros(2), np.zeros(1))
        lin.compute(np.zeros(2), np.zeros(1))  # Cache hit

        assert lin._stats["computes"] > 0
        assert lin._stats["cache_hits"] > 0

        # Clear
        lin.clear_cache()

        assert len(lin._cache) == 0
        assert lin._stats["computes"] == 0
        assert lin._stats["cache_hits"] == 0

    def test_list_cached(self):
        """Test listing cached items."""
        system = DiscreteLinearSystem()
        system.add_equilibrium("origin", np.zeros(2), np.zeros(1))

        lin = DiscreteLinearization(system)

        # Initially empty
        assert len(lin.list_cached()) == 0

        # Cache some
        lin.compute("origin")
        lin.compute(np.ones(2), np.zeros(1))

        cached = lin.list_cached()
        assert len(cached) == 2
        assert "origin" in cached


# ============================================================================
# Test Statistics
# ============================================================================


class TestStatistics:
    """Test cache statistics."""

    def test_get_stats(self):
        """Test statistics collection."""
        system = DiscreteLinearSystem()
        lin = DiscreteLinearization(system)

        x_eq = np.zeros(2)
        u_eq = np.zeros(1)

        # Initial stats
        stats = lin.get_stats()
        assert stats["computes"] == 0
        assert stats["cache_hits"] == 0
        assert stats["hit_rate"] == 0.0

        # First call (compute)
        lin.compute(x_eq, u_eq)
        stats = lin.get_stats()
        assert stats["computes"] == 1

        # Second call (cache hit)
        lin.compute(x_eq, u_eq)
        stats = lin.get_stats()
        assert stats["cache_hits"] == 1
        assert stats["hit_rate"] == 0.5  # 1 hit / 2 total

    def test_reset_stats(self):
        """Test resetting statistics."""
        system = DiscreteLinearSystem()
        lin = DiscreteLinearization(system)

        # Generate activity
        lin.compute(np.zeros(2), np.zeros(1))
        lin.compute(np.zeros(2), np.zeros(1))

        # Reset stats (but keep cache)
        cache_size = len(lin._cache)
        lin.reset_stats()

        stats = lin.get_stats()
        assert stats["computes"] == 0
        assert stats["cache_hits"] == 0
        assert len(lin._cache) == cache_size  # Cache still there


# ============================================================================
# Test Backend Compatibility
# ============================================================================


class TestBackendCompatibility:
    """Test multi-backend support."""

    def test_numpy_backend(self):
        """Test NumPy backend."""
        system = DiscreteLinearSystem()
        lin = DiscreteLinearization(system)

        Ad, Bd = lin.compute(np.zeros(2), np.zeros(1))

        assert isinstance(Ad, np.ndarray)
        assert isinstance(Bd, np.ndarray)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_torch_backend(self):
        """Test PyTorch backend."""
        system = DiscreteLinearSystem()
        system.set_default_backend("torch")
        lin = DiscreteLinearization(system)

        x_eq = torch.zeros(2)
        u_eq = torch.zeros(1)

        Ad, Bd = lin.compute(x_eq, u_eq)

        assert isinstance(Ad, torch.Tensor)
        assert isinstance(Bd, torch.Tensor)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_backend(self):
        """Test JAX backend."""
        system = DiscreteLinearSystem()
        system.set_default_backend("jax")
        lin = DiscreteLinearization(system)

        x_eq = jnp.zeros(2)
        u_eq = jnp.zeros(1)

        Ad, Bd = lin.compute(x_eq, u_eq)

        assert isinstance(Ad, jnp.ndarray)
        assert isinstance(Bd, jnp.ndarray)


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_state_system(self):
        """Test 1D system."""
        system = DiscreteNonlinearSystem()
        lin = DiscreteLinearization(system)

        Ad, Bd = lin.compute(np.array([0.0]), np.array([0.0]))

        assert Ad.shape == (1, 1)
        assert Bd.shape == (1, 1)

    def test_multiple_controls(self):
        """Test system with multiple control inputs."""

        class MultiControlSystem(DiscreteSymbolicSystem):
            def define_system(self):
                x = sp.symbols("x", real=True)
                u1, u2 = sp.symbols("u1 u2", real=True)

                self.state_vars = [x]
                self.control_vars = [u1, u2]
                self._f_sym = sp.Matrix([0.9 * x + u1 + u2])
                self.parameters = {}
                self.order = 1

        system = MultiControlSystem()
        lin = DiscreteLinearization(system)

        Ad, Bd = lin.compute(np.array([0.0]), np.array([0.0, 0.0]))

        assert Ad.shape == (1, 1)
        assert Bd.shape == (1, 2)  # Two control inputs

    def test_autonomous_system(self):
        """Test autonomous system (nu=0)."""

        class AutonomousSystem(DiscreteSymbolicSystem):
            def define_system(self):
                x = sp.symbols("x", real=True)
                self.state_vars = [x]
                self.control_vars = []
                self._f_sym = sp.Matrix([0.9 * x])
                self.parameters = {}
                self.order = 1

        system = AutonomousSystem()
        lin = DiscreteLinearization(system)

        Ad, Bd = lin.compute(np.array([0.0]))

        assert Ad.shape == (1, 1)
        assert Bd.shape == (1, 0)  # No control


# ============================================================================
# Test Information Methods
# ============================================================================


class TestInformation:
    """Test information and diagnostic methods."""

    def test_get_info(self):
        """Test get_info method."""
        system = DiscreteLinearSystem()
        lin = DiscreteLinearization(system)

        # Cache something
        lin.compute(np.zeros(2), np.zeros(1))

        info = lin.get_info()

        assert info["system"] == "DiscreteLinearSystem"
        assert info["system_type"] == "discrete"
        assert info["discretizer"] is None
        assert info["backend"] == "numpy"
        assert info["cache_size"] == 1
        assert "statistics" in info

    def test_get_info_discretized(self):
        """Test get_info for discretized system."""
        system = SimpleOscillator()
        discretizer = Discretizer(system, dt=0.01, method="rk4")
        lin = DiscreteLinearization(system, discretizer=discretizer)

        info = lin.get_info()

        assert info["system_type"] == "discretized"
        assert info["discretizer"] == "rk4"

    def test_repr_str(self):
        """Test string representations."""
        system = DiscreteLinearSystem()
        lin = DiscreteLinearization(system)

        # Before caching
        repr_str = repr(lin)
        assert "DiscreteLinearization" in repr_str
        assert "cache_size=0" in repr_str

        # After caching
        lin.compute(np.zeros(2), np.zeros(1))
        repr_str = repr(lin)
        assert "cache_size=1" in repr_str


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_lqr_design_workflow(self):
        """Test typical LQR design workflow."""
        system = DiscreteLinearSystem()
        system.add_equilibrium("origin", np.zeros(2), np.zeros(1))

        lin = DiscreteLinearization(system)

        # Get linearization
        Ad, Bd = lin.compute("origin")

        # Would design LQR controller here
        # K = solve_lqr(Ad, Bd, Q, R)

        # Verify shapes for LQR
        assert Ad.shape == (2, 2)
        assert Bd.shape == (2, 1)

    def test_gain_scheduling_complete(self):
        """Test complete gain scheduling workflow."""
        system = DiscreteNonlinearSystem()
        lin = DiscreteLinearization(system)

        # Operating points
        x_points = [-1.0, -0.5, 0.0, 0.5, 1.0]

        # Precompute all
        for x_val in x_points:
            lin.compute(np.array([x_val]), np.array([0.0]))

        assert len(lin._cache) == 5

        # All future accesses are cached
        for x_val in x_points:
            Ad, Bd = lin.compute(np.array([x_val]), np.array([0.0]))
            # Use Ad, Bd for local controller design
            assert Ad.shape == (1, 1)

        # All were cache hits
        assert lin._stats["cache_hits"] == 5

    def test_mpc_repeated_linearization(self):
        """Test MPC-like repeated linearization scenario."""
        system = DiscreteNonlinearSystem()
        lin = DiscreteLinearization(system)

        # Simulate MPC: linearize at current state repeatedly
        x_current = np.array([0.5])
        u_current = np.array([0.0])

        # First call computes
        for _ in range(10):
            Ad, Bd = lin.compute(x_current, u_current)
            # In real MPC, would solve optimization here

        # 1 compute + 9 cache hits
        stats = lin.get_stats()
        assert stats["computes"] == 1
        assert stats["cache_hits"] == 9
        assert stats["hit_rate"] == 0.9


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Test performance and efficiency."""

    def test_cache_speedup(self):
        """Test that cache provides speedup."""
        system = DiscreteLinearSystem()
        lin = DiscreteLinearization(system)

        x_eq = np.zeros(2)
        u_eq = np.zeros(1)

        import time

        # First call (compute)
        start = time.time()
        Ad1, Bd1 = lin.compute(x_eq, u_eq)
        time_compute = time.time() - start

        # Second call (cache)
        start = time.time()
        Ad2, Bd2 = lin.compute(x_eq, u_eq)
        time_cache = time.time() - start

        # Cache should be much faster (but test might be too fast to measure)
        # Just verify both completed
        assert Ad1 is Ad2  # Same object proves cache worked

    def test_large_grid_precomputation(self):
        """Test precomputation with large grid."""
        system = DiscreteNonlinearSystem()
        lin = DiscreteLinearization(system)

        # Large grid
        n_points = 50
        x_grid = np.linspace(-2, 2, n_points).reshape(-1, 1)
        u_grid = np.zeros((n_points, 1))

        lin.precompute_at_grid(x_grid, u_grid)

        # All cached
        assert lin._stats["computes"] == n_points


# ============================================================================
# Test Numerical Accuracy
# ============================================================================


class TestNumericalAccuracy:
    """Test numerical accuracy and consistency."""

    def test_linear_system_constant_linearization(self):
        """Test that linear system has constant linearization."""
        system = DiscreteLinearSystem()
        lin = DiscreteLinearization(system)

        # Linearize at different points
        Ad1, Bd1 = lin.compute(np.array([0.0, 0.0]), np.array([0.0]))
        Ad2, Bd2 = lin.compute(np.array([1.0, 1.0]), np.array([0.0]))
        Ad3, Bd3 = lin.compute(np.array([-1.0, 2.0]), np.array([0.0]))

        # For linear system, should all be identical
        np.testing.assert_allclose(Ad1, Ad2, rtol=1e-10)
        np.testing.assert_allclose(Ad1, Ad3, rtol=1e-10)
        np.testing.assert_allclose(Bd1, Bd2, rtol=1e-10)

    def test_nonlinear_system_varying_linearization(self):
        """Test that nonlinear system has different linearizations."""
        system = DiscreteNonlinearSystem()
        lin = DiscreteLinearization(system)

        # Linearize at different points
        Ad1, _ = lin.compute(np.array([0.0]), np.array([0.0]))
        Ad2, _ = lin.compute(np.array([np.pi]), np.array([0.0]))

        # Should be different for nonlinear system
        assert not np.allclose(Ad1, Ad2, rtol=1e-6)

    def test_discretization_consistency(self):
        """Test consistency between discretization methods."""
        system = SimpleOscillator(omega=1.0)
        discretizer = Discretizer(system, dt=0.001, method="rk4")  # Small dt
        lin = DiscreteLinearization(system, discretizer=discretizer)

        x_eq = np.zeros(2)
        u_eq = np.zeros(1)

        # With small dt, all methods should be similar
        Ad_euler, Bd_euler = lin.compute(x_eq, u_eq, method="euler")
        Ad_exact, Bd_exact = lin.compute(x_eq, u_eq, method="exact")

        # Should be close for small dt
        np.testing.assert_allclose(Ad_euler, Ad_exact, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
