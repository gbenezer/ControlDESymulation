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
Unit Tests for System Results Types Module

Tests cover:
- Direct imports from system_results module
- Base type field inheritance
- All 10 concrete result types
- All 6 union types for polymorphic code
- Type hierarchy and inheritance structure
- Required vs optional fields
- Polymorphic function patterns using SystemResult
- Type narrowing and discrimination
- Module exports (__all__)
- Edge cases and realistic usage patterns

This test suite is complementary to test_trajectories.py, which tests
result types via re-exports. This suite focuses on:
- Direct imports from system_results
- Type inheritance verification
- Union type polymorphism
- Module organization

Note: Result types use standardized keys per time-major-rollout refactoring:
- 't' for time (not 'time')
- 'x' for states (not 'states')
- 'u' for controls (not 'controls')
"""

import numpy as np
import pytest
from typing import get_type_hints

# Direct imports from system_results module
from cdesym.types.system_results import (
    # Base types
    IntegrationResultBase,
    SimulationResultBase,
    RolloutResultBase,
    DiscreteSimulationResultBase,
    # Continuous deterministic
    IntegrationResult,
    SimulationResult,
    RolloutResult,
    # Continuous stochastic
    SDEIntegrationResult,
    SDESimulationResult,
    SDERolloutResult,
    # Discrete deterministic
    DiscreteSimulationResult,
    DiscreteRolloutResult,
    # Discrete stochastic
    DiscreteStochasticSimulationResult,
    DiscreteStochasticRolloutResult,
    # Union types
    ContinuousIntegrationResultUnion,
    ContinuousSimulationResultUnion,
    ContinuousRolloutResultUnion,
    DiscreteSimulationResultUnion,
    DiscreteRolloutResultUnion,
    SystemResult,
)


# ============================================================================
# Test Module Exports
# ============================================================================


class TestModuleExports:
    """Test that all types are properly exported."""

    def test_all_base_types_exported(self):
        """Test all base types are importable."""
        assert IntegrationResultBase is not None
        assert SimulationResultBase is not None
        assert RolloutResultBase is not None
        assert DiscreteSimulationResultBase is not None

    def test_all_continuous_deterministic_exported(self):
        """Test all continuous deterministic types are importable."""
        assert IntegrationResult is not None
        assert SimulationResult is not None
        assert RolloutResult is not None

    def test_all_continuous_stochastic_exported(self):
        """Test all continuous stochastic types are importable."""
        assert SDEIntegrationResult is not None
        assert SDESimulationResult is not None
        assert SDERolloutResult is not None

    def test_all_discrete_deterministic_exported(self):
        """Test all discrete deterministic types are importable."""
        assert DiscreteSimulationResult is not None
        assert DiscreteRolloutResult is not None

    def test_all_discrete_stochastic_exported(self):
        """Test all discrete stochastic types are importable."""
        assert DiscreteStochasticSimulationResult is not None
        assert DiscreteStochasticRolloutResult is not None

    def test_all_union_types_exported(self):
        """Test all union types are importable."""
        assert ContinuousIntegrationResultUnion is not None
        assert ContinuousSimulationResultUnion is not None
        assert ContinuousRolloutResultUnion is not None
        assert DiscreteSimulationResultUnion is not None
        assert DiscreteRolloutResultUnion is not None
        assert SystemResult is not None

    def test_module_all_list(self):
        """Test __all__ contains expected types."""
        import cdesym.types.system_results as sr

        expected_exports = [
            # Base types
            "IntegrationResultBase",
            "SimulationResultBase",
            "RolloutResultBase",
            "DiscreteSimulationResultBase",
            # Continuous deterministic
            "IntegrationResult",
            "SimulationResult",
            "RolloutResult",
            # Continuous stochastic
            "SDEIntegrationResult",
            "SDESimulationResult",
            "SDERolloutResult",
            # Discrete deterministic
            "DiscreteSimulationResult",
            "DiscreteRolloutResult",
            # Discrete stochastic
            "DiscreteStochasticSimulationResult",
            "DiscreteStochasticRolloutResult",
            # Union types
            "ContinuousIntegrationResultUnion",
            "ContinuousSimulationResultUnion",
            "ContinuousRolloutResultUnion",
            "DiscreteSimulationResultUnion",
            "DiscreteRolloutResultUnion",
            "SystemResult",
        ]

        for name in expected_exports:
            assert name in sr.__all__, f"{name} not in __all__"


# ============================================================================
# Test Base Types
# ============================================================================


class TestIntegrationResultBase:
    """Test IntegrationResultBase type."""

    def test_base_fields(self):
        """Test base integration result has expected fields."""
        result: IntegrationResultBase = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
            "message": "Integration successful",
            "nfev": 500,
            "nsteps": 100,
            "solver": "RK45",
            "integration_time": 0.05,
        }

        assert "t" in result
        assert "x" in result
        assert "success" in result
        assert "nfev" in result
        assert "solver" in result

    def test_optional_solver_diagnostics(self):
        """Test optional solver diagnostic fields."""
        result: IntegrationResultBase = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
            "nfev": 200,
            "njev": 50,
            "nlu": 30,
            "status": 0,
            "dense_output": True,
        }

        assert result["njev"] == 50
        assert result["nlu"] == 30
        assert result["dense_output"] is True


class TestSimulationResultBase:
    """Test SimulationResultBase type."""

    def test_base_fields(self):
        """Test base simulation result has expected fields."""
        result: SimulationResultBase = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
            "message": "Simulation completed",
            "method": "rk4",
            "dt": 0.1,
        }

        assert "t" in result
        assert "x" in result
        assert "method" in result
        assert "dt" in result

    def test_optional_control_and_output(self):
        """Test optional control and output fields."""
        result: SimulationResultBase = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
            "u": np.random.randn(100, 1),
            "y": np.random.randn(101, 2),
        }

        assert result["u"].shape == (100, 1)
        assert result["y"].shape == (101, 2)

    def test_metadata_field(self):
        """Test metadata dictionary field."""
        result: SimulationResultBase = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
            "metadata": {
                "system_name": "pendulum",
                "backend": "numpy",
                "version": "1.0.0",
            },
        }

        assert result["metadata"]["system_name"] == "pendulum"
        assert result["metadata"]["backend"] == "numpy"


class TestRolloutResultBase:
    """Test RolloutResultBase type."""

    def test_base_fields_with_required_control(self):
        """Test rollout base requires control field."""
        result: RolloutResultBase = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "u": np.random.randn(100, 1),  # Always present in rollout
            "success": True,
            "method": "rk4",
            "dt": 0.1,
            "controller_type": "LQR",
            "closed_loop": True,
        }

        assert "u" in result
        assert result["closed_loop"] is True
        assert result["controller_type"] == "LQR"


class TestDiscreteSimulationResultBase:
    """Test DiscreteSimulationResultBase type."""

    def test_base_fields(self):
        """Test discrete base has integer time steps."""
        result: DiscreteSimulationResultBase = {
            "t": np.arange(101),  # Integer time steps
            "x": np.random.randn(101, 2),
            "dt": 0.01,
            "success": True,
            "method": "direct",
        }

        assert result["t"][0] == 0
        assert result["t"][-1] == 100
        assert result["dt"] == 0.01


# ============================================================================
# Test Type Inheritance
# ============================================================================


class TestTypeInheritance:
    """Test type inheritance relationships."""

    def test_integration_result_inherits_base(self):
        """Test IntegrationResult inherits from IntegrationResultBase."""
        # IntegrationResult should accept all base fields
        result: IntegrationResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
            "message": "Success",
            "nfev": 500,
            "nsteps": 100,
            "solver": "RK45",
            "integration_time": 0.05,
        }

        # Should work as base type too
        base_result: IntegrationResultBase = result
        assert base_result["solver"] == "RK45"

    def test_sde_integration_extends_base(self):
        """Test SDEIntegrationResult extends base with SDE fields."""
        result: SDEIntegrationResult = {
            # Base fields
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
            "nfev": 500,
            "nsteps": 100,
            "solver": "euler_maruyama",
            "integration_time": 0.1,
            # SDE-specific fields
            "diffusion_evals": 500,
            "noise_samples": np.random.randn(100, 2),
            "n_paths": 1,
            "noise_type": "diagonal",
            "sde_type": "ito",
            "convergence_type": "strong",
        }

        assert "diffusion_evals" in result
        assert "sde_type" in result
        assert result["convergence_type"] == "strong"

    def test_simulation_result_inherits_base(self):
        """Test SimulationResult inherits from SimulationResultBase."""
        result: SimulationResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
            "method": "rk4",
            "dt": 0.1,
        }

        base_result: SimulationResultBase = result
        assert base_result["dt"] == 0.1

    def test_rollout_result_inherits_base(self):
        """Test RolloutResult inherits from RolloutResultBase."""
        result: RolloutResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "u": np.random.randn(100, 1),
            "success": True,
            "method": "rk4",
            "dt": 0.1,
            "controller_type": "LQR",
            "closed_loop": True,
        }

        base_result: RolloutResultBase = result
        assert base_result["closed_loop"] is True

    def test_discrete_types_inherit_base(self):
        """Test discrete types inherit from DiscreteSimulationResultBase."""
        # DiscreteSimulationResult
        sim_result: DiscreteSimulationResult = {
            "t": np.arange(101),
            "x": np.random.randn(101, 2),
            "dt": 0.01,
            "success": True,
        }

        # DiscreteRolloutResult adds policy fields
        rollout_result: DiscreteRolloutResult = {
            "t": np.arange(101),
            "x": np.random.randn(101, 2),
            "u": np.random.randn(100, 1),
            "dt": 0.01,
            "success": True,
            "policy_type": "LQR",
            "closed_loop": True,
        }

        assert sim_result["dt"] == rollout_result["dt"]
        assert rollout_result["policy_type"] == "LQR"


# ============================================================================
# Test Continuous Deterministic Types
# ============================================================================


class TestIntegrationResult:
    """Test IntegrationResult concrete type."""

    def test_complete_integration_result(self):
        """Test complete integration result with all fields."""
        result: IntegrationResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 3),
            "success": True,
            "message": "Integration successful",
            "nfev": 523,
            "njev": 0,
            "nlu": 0,
            "nsteps": 100,
            "solver": "RK45",
            "integration_time": 0.05,
            "status": 0,
            "dense_output": False,
        }

        assert result["success"] is True
        assert result["t"].shape == (101,)
        assert result["x"].shape == (101, 3)

    def test_time_major_shape_convention(self):
        """Test time-major shape: (T, nx) not (nx, T)."""
        T = 101
        nx = 3

        result: IntegrationResult = {
            "t": np.linspace(0, 10, T),
            "x": np.random.randn(T, nx),  # Time-major!
            "success": True,
        }

        # First dimension is time
        assert result["x"].shape[0] == T
        # Second dimension is state
        assert result["x"].shape[1] == nx


class TestSimulationResult:
    """Test SimulationResult concrete type."""

    def test_regular_time_grid(self):
        """Test simulation uses regular time grid."""
        dt = 0.1
        t_span = (0.0, 10.0)
        T = int((t_span[1] - t_span[0]) / dt) + 1

        result: SimulationResult = {
            "t": np.linspace(t_span[0], t_span[1], T),
            "x": np.random.randn(T, 2),
            "success": True,
            "method": "rk4",
            "dt": dt,
        }

        # Verify regular spacing
        time_diffs = np.diff(result["t"])
        assert np.allclose(time_diffs, dt)

    def test_open_loop_control(self):
        """Test simulation with open-loop control sequence."""
        n_steps = 100
        nx = 2
        nu = 1

        result: SimulationResult = {
            "t": np.linspace(0, 10, n_steps + 1),
            "x": np.random.randn(n_steps + 1, nx),
            "u": np.random.randn(n_steps, nu),
            "success": True,
            "method": "euler",
            "dt": 0.1,
        }

        # Control sequence is one shorter than state
        assert result["x"].shape[0] == result["u"].shape[0] + 1


class TestRolloutResult:
    """Test RolloutResult concrete type."""

    def test_closed_loop_always_true(self):
        """Test rollout always has closed_loop=True."""
        result: RolloutResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "u": np.random.randn(100, 1),
            "success": True,
            "method": "rk4",
            "dt": 0.1,
            "controller_type": "LQR",
            "closed_loop": True,
        }

        assert result["closed_loop"] is True

    def test_control_always_present(self):
        """Test control is always present in rollout (from feedback)."""
        result: RolloutResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "u": np.random.randn(100, 1),  # Required, not optional
            "success": True,
            "controller_type": "PID",
            "closed_loop": True,
        }

        assert "u" in result
        assert result["u"] is not None


# ============================================================================
# Test Continuous Stochastic Types
# ============================================================================


class TestSDEIntegrationResult:
    """Test SDEIntegrationResult concrete type."""

    def test_single_path(self):
        """Test SDE integration with single path."""
        n_steps = 100
        nx = 2
        nw = 2

        result: SDEIntegrationResult = {
            "t": np.linspace(0, 10, n_steps + 1),
            "x": np.random.randn(n_steps + 1, nx),
            "success": True,
            "nfev": n_steps,
            "nsteps": n_steps,
            "solver": "euler_maruyama",
            "integration_time": 0.1,
            "diffusion_evals": n_steps,
            "noise_samples": np.random.randn(n_steps, nw),
            "n_paths": 1,
            "noise_type": "diagonal",
            "sde_type": "ito",
            "convergence_type": "strong",
        }

        assert result["n_paths"] == 1
        assert result["x"].ndim == 2  # (T, nx) for single path

    def test_monte_carlo_paths(self):
        """Test SDE integration with multiple Monte Carlo paths."""
        n_steps = 100
        n_paths = 500
        nx = 2
        nw = 2

        result: SDEIntegrationResult = {
            "t": np.linspace(0, 10, n_steps + 1),
            "x": np.random.randn(n_paths, n_steps + 1, nx),
            "success": True,
            "nfev": n_steps * n_paths,
            "nsteps": n_steps,
            "solver": "euler_maruyama",
            "integration_time": 1.5,
            "diffusion_evals": n_steps * n_paths,
            "noise_samples": np.random.randn(n_paths, n_steps, nw),
            "n_paths": n_paths,
            "noise_type": "diagonal",
            "sde_type": "ito",
            "convergence_type": "strong",
        }

        assert result["n_paths"] == 500
        assert result["x"].ndim == 3  # (n_paths, T, nx)
        assert result["x"].shape == (n_paths, n_steps + 1, nx)

    def test_sde_specific_fields(self):
        """Test SDE-specific field values."""
        result: SDEIntegrationResult = {
            "t": np.linspace(0, 1, 11),
            "x": np.random.randn(11, 2),
            "success": True,
            "nfev": 10,
            "nsteps": 10,
            "solver": "milstein",
            "integration_time": 0.01,
            "diffusion_evals": 10,
            "noise_samples": np.random.randn(10, 2),
            "n_paths": 1,
            "noise_type": "multiplicative",
            "sde_type": "stratonovich",
            "convergence_type": "weak",
        }

        assert result["noise_type"] in ["additive", "multiplicative", "diagonal"]
        assert result["sde_type"] in ["ito", "stratonovich"]
        assert result["convergence_type"] in ["strong", "weak"]


class TestSDESimulationResult:
    """Test SDESimulationResult concrete type."""

    def test_monte_carlo_shape(self):
        """Test Monte Carlo shape: (n_paths, T, nx)."""
        n_steps = 100
        n_paths = 500
        nx = 2
        nw = 2

        result: SDESimulationResult = {
            "t": np.linspace(0, 10, n_steps + 1),
            "x": np.random.randn(n_paths, n_steps + 1, nx),
            "success": True,
            "method": "euler_maruyama",
            "dt": 0.1,
            "n_paths": n_paths,
            "noise_type": "diagonal",
            "sde_type": "ito",
            "seed": 42,
            "noise_samples": np.random.randn(n_paths, n_steps, nw),
            "diffusion_evals": n_steps * n_paths,
        }

        assert result["x"].shape[0] == n_paths
        assert result["x"].shape[1] == n_steps + 1
        assert result["x"].shape[2] == nx

    def test_reproducibility_with_seed(self):
        """Test seed field for reproducibility."""
        result: SDESimulationResult = {
            "t": np.linspace(0, 1, 11),
            "x": np.random.randn(10, 11, 2),
            "success": True,
            "method": "euler_maruyama",
            "dt": 0.1,
            "n_paths": 10,
            "noise_type": "diagonal",
            "sde_type": "ito",
            "seed": 12345,
            "noise_samples": np.random.randn(10, 10, 2),
            "diffusion_evals": 100,
        }

        assert result["seed"] == 12345


class TestSDERolloutResult:
    """Test SDERolloutResult concrete type."""

    def test_closed_loop_stochastic(self):
        """Test SDE rollout with feedback control."""
        n_steps = 100
        n_paths = 50
        nx = 2
        nu = 1
        nw = 2

        result: SDERolloutResult = {
            "t": np.linspace(0, 10, n_steps + 1),
            "x": np.random.randn(n_paths, n_steps + 1, nx),
            "u": np.random.randn(n_paths, n_steps, nu),
            "success": True,
            "method": "euler_maruyama",
            "dt": 0.1,
            "controller_type": "LQG",
            "closed_loop": True,
            "n_paths": n_paths,
            "noise_type": "diagonal",
            "sde_type": "ito",
            "seed": 42,
            "noise_samples": np.random.randn(n_paths, n_steps, nw),
            "diffusion_evals": n_steps * n_paths,
        }

        assert result["closed_loop"] is True
        # Control varies per path
        assert result["u"].shape[0] == n_paths


# ============================================================================
# Test Discrete Deterministic Types
# ============================================================================


class TestDiscreteSimulationResult:
    """Test DiscreteSimulationResult concrete type."""

    def test_integer_time_steps(self):
        """Test discrete time uses integer steps."""
        n_steps = 100

        result: DiscreteSimulationResult = {
            "t": np.arange(n_steps + 1),
            "x": np.random.randn(n_steps + 1, 2),
            "dt": 0.01,
            "success": True,
            "method": "direct",
        }

        # Time steps are integers
        assert result["t"].dtype in [np.int32, np.int64, int]
        assert result["t"][0] == 0
        assert result["t"][-1] == n_steps

    def test_open_loop_discrete(self):
        """Test discrete open-loop simulation."""
        n_steps = 100
        nx = 3
        nu = 2

        result: DiscreteSimulationResult = {
            "t": np.arange(n_steps + 1),
            "x": np.random.randn(n_steps + 1, nx),
            "u": np.random.randn(n_steps, nu),
            "dt": 0.01,
            "success": True,
        }

        assert result["x"].shape == (n_steps + 1, nx)
        assert result["u"].shape == (n_steps, nu)


class TestDiscreteRolloutResult:
    """Test DiscreteRolloutResult concrete type."""

    def test_policy_type_field(self):
        """Test discrete rollout has policy_type (not controller_type)."""
        result: DiscreteRolloutResult = {
            "t": np.arange(101),
            "x": np.random.randn(101, 2),
            "u": np.random.randn(100, 1),
            "dt": 0.01,
            "success": True,
            "policy_type": "MPC",
            "closed_loop": True,
        }

        assert result["policy_type"] == "MPC"
        assert result["closed_loop"] is True


# ============================================================================
# Test Discrete Stochastic Types
# ============================================================================


class TestDiscreteStochasticSimulationResult:
    """Test DiscreteStochasticSimulationResult concrete type."""

    def test_monte_carlo_discrete(self):
        """Test discrete stochastic Monte Carlo."""
        n_steps = 100
        n_paths = 1000
        nx = 2
        nw = 2

        result: DiscreteStochasticSimulationResult = {
            "t": np.arange(n_steps + 1),
            "x": np.random.randn(n_paths, n_steps + 1, nx),
            "dt": 0.1,
            "success": True,
            "n_paths": n_paths,
            "noise_type": "additive",
            "seed": 42,
            "noise_samples": np.random.randn(n_paths, n_steps, nw),
        }

        assert result["x"].shape == (n_paths, n_steps + 1, nx)
        assert result["n_paths"] == 1000


class TestDiscreteStochasticRolloutResult:
    """Test DiscreteStochasticRolloutResult concrete type."""

    def test_stochastic_feedback(self):
        """Test discrete stochastic rollout with feedback."""
        n_steps = 100
        n_paths = 50
        nx = 2
        nu = 1
        nw = 2

        result: DiscreteStochasticRolloutResult = {
            "t": np.arange(n_steps + 1),
            "x": np.random.randn(n_paths, n_steps + 1, nx),
            "u": np.random.randn(n_paths, n_steps, nu),
            "dt": 0.1,
            "success": True,
            "policy_type": "robust_LQR",
            "closed_loop": True,
            "n_paths": n_paths,
            "noise_type": "additive",
            "seed": 42,
            "noise_samples": np.random.randn(n_paths, n_steps, nw),
        }

        assert result["policy_type"] == "robust_LQR"
        assert result["u"].shape == (n_paths, n_steps, nu)


# ============================================================================
# Test Union Types
# ============================================================================


class TestContinuousIntegrationResultUnion:
    """Test ContinuousIntegrationResultUnion type."""

    def test_accepts_integration_result(self):
        """Test union accepts IntegrationResult."""
        result: ContinuousIntegrationResultUnion = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
            "nfev": 500,
            "nsteps": 100,
            "solver": "RK45",
            "integration_time": 0.05,
        }

        assert result["solver"] == "RK45"

    def test_accepts_sde_integration_result(self):
        """Test union accepts SDEIntegrationResult."""
        result: ContinuousIntegrationResultUnion = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
            "nfev": 500,
            "nsteps": 100,
            "solver": "euler_maruyama",
            "integration_time": 0.1,
            "diffusion_evals": 500,
            "noise_samples": np.random.randn(100, 2),
            "n_paths": 1,
            "noise_type": "diagonal",
            "sde_type": "ito",
            "convergence_type": "strong",
        }

        assert result["sde_type"] == "ito"


class TestContinuousSimulationResultUnion:
    """Test ContinuousSimulationResultUnion type."""

    def test_accepts_simulation_result(self):
        """Test union accepts SimulationResult."""
        result: ContinuousSimulationResultUnion = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
            "method": "rk4",
            "dt": 0.1,
        }

        assert result["method"] == "rk4"

    def test_accepts_sde_simulation_result(self):
        """Test union accepts SDESimulationResult."""
        result: ContinuousSimulationResultUnion = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(50, 101, 2),
            "success": True,
            "method": "euler_maruyama",
            "dt": 0.1,
            "n_paths": 50,
            "noise_type": "diagonal",
            "sde_type": "ito",
            "seed": 42,
            "noise_samples": np.random.randn(50, 100, 2),
            "diffusion_evals": 5000,
        }

        assert result["n_paths"] == 50


class TestDiscreteSimulationResultUnion:
    """Test DiscreteSimulationResultUnion type."""

    def test_accepts_deterministic(self):
        """Test union accepts DiscreteSimulationResult."""
        result: DiscreteSimulationResultUnion = {
            "t": np.arange(101),
            "x": np.random.randn(101, 2),
            "dt": 0.01,
            "success": True,
        }

        assert result["dt"] == 0.01

    def test_accepts_stochastic(self):
        """Test union accepts DiscreteStochasticSimulationResult."""
        result: DiscreteSimulationResultUnion = {
            "t": np.arange(101),
            "x": np.random.randn(100, 101, 2),
            "dt": 0.01,
            "success": True,
            "n_paths": 100,
            "noise_type": "additive",
            "seed": 42,
            "noise_samples": np.random.randn(100, 100, 2),
        }

        assert result["n_paths"] == 100


class TestSystemResultUnion:
    """Test SystemResult union type (all types)."""

    def test_accepts_all_continuous_deterministic(self):
        """Test SystemResult accepts all continuous deterministic types."""
        # IntegrationResult
        int_result: SystemResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
            "nfev": 500,
            "nsteps": 100,
            "solver": "RK45",
            "integration_time": 0.05,
        }
        assert "solver" in int_result

        # SimulationResult
        sim_result: SystemResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
            "method": "rk4",
            "dt": 0.1,
        }
        assert "method" in sim_result

        # RolloutResult
        roll_result: SystemResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "u": np.random.randn(100, 1),
            "success": True,
            "controller_type": "LQR",
            "closed_loop": True,
        }
        assert "closed_loop" in roll_result

    def test_accepts_all_discrete_types(self):
        """Test SystemResult accepts all discrete types."""
        # DiscreteSimulationResult
        dsim: SystemResult = {
            "t": np.arange(101),
            "x": np.random.randn(101, 2),
            "dt": 0.01,
            "success": True,
        }
        assert dsim["t"][0] == 0

        # DiscreteRolloutResult
        droll: SystemResult = {
            "t": np.arange(101),
            "x": np.random.randn(101, 2),
            "u": np.random.randn(100, 1),
            "dt": 0.01,
            "success": True,
            "policy_type": "LQR",
            "closed_loop": True,
        }
        assert droll["closed_loop"] is True


# ============================================================================
# Test Polymorphic Functions
# ============================================================================


class TestPolymorphicFunctions:
    """Test functions that work with SystemResult union."""

    def test_extract_final_state_single_trajectory(self):
        """Test extracting final state from single trajectory."""

        def extract_final_state(result: SystemResult) -> np.ndarray:
            """Extract final state from any system result type."""
            x = result["x"]
            if x.ndim == 2:
                return x[-1]  # Single trajectory
            else:
                return x[:, -1]  # Multiple paths (n_paths, T, nx) -> (n_paths, nx)

        # Test with SimulationResult
        sim_result: SystemResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 3),
            "success": True,
        }
        final = extract_final_state(sim_result)
        assert final.shape == (3,)

    def test_extract_final_state_monte_carlo(self):
        """Test extracting final state from Monte Carlo result."""

        def extract_final_state(result: SystemResult) -> np.ndarray:
            x = result["x"]
            if x.ndim == 2:
                return x[-1]
            else:
                return x[:, -1]

        # Test with SDESimulationResult
        sde_result: SystemResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(500, 101, 2),
            "success": True,
            "n_paths": 500,
            "noise_type": "diagonal",
            "sde_type": "ito",
        }
        final = extract_final_state(sde_result)
        assert final.shape == (500, 2)

    def test_compute_trajectory_duration(self):
        """Test computing duration from any result type."""

        def compute_duration(result: SystemResult) -> float:
            t = result["t"]
            return float(t[-1] - t[0])

        # Continuous
        cont_result: SystemResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
        }
        assert compute_duration(cont_result) == pytest.approx(10.0)

        # Discrete (integer time)
        disc_result: SystemResult = {
            "t": np.arange(101),
            "x": np.random.randn(101, 2),
            "dt": 0.01,
            "success": True,
        }
        assert compute_duration(disc_result) == 100  # Steps, not time

    def test_check_success(self):
        """Test checking success across all result types."""

        def is_successful(result: SystemResult) -> bool:
            return result.get("success", False)

        successful: SystemResult = {
            "t": np.linspace(0, 1, 11),
            "x": np.random.randn(11, 2),
            "success": True,
        }
        assert is_successful(successful) is True

        failed: SystemResult = {
            "t": np.array([0.0]),
            "x": np.array([[1.0, 0.0]]),
            "success": False,
            "message": "Integration failed",
        }
        assert is_successful(failed) is False

    def test_generic_plotting_data(self):
        """Test extracting plotting data from any result."""

        def get_plot_data(result: SystemResult) -> tuple:
            """Extract time and state for plotting."""
            t = result["t"]
            x = result["x"]
            # Handle Monte Carlo by taking mean
            if x.ndim == 3:
                x = np.mean(x, axis=0)
            return t, x

        # Single trajectory
        sim_result: SystemResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
        }
        t, x = get_plot_data(sim_result)
        assert t.shape == (101,)
        assert x.shape == (101, 2)

        # Monte Carlo
        mc_result: SystemResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(500, 101, 2),
            "success": True,
            "n_paths": 500,
        }
        t, x = get_plot_data(mc_result)
        assert t.shape == (101,)
        assert x.shape == (101, 2)  # Mean trajectory


# ============================================================================
# Test Type Discrimination
# ============================================================================


class TestTypeDiscrimination:
    """Test discriminating between result types."""

    def test_discriminate_by_n_paths(self):
        """Test using n_paths to discriminate stochastic results."""

        def is_monte_carlo(result: SystemResult) -> bool:
            return result.get("n_paths", 1) > 1

        single: SystemResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
        }
        assert is_monte_carlo(single) is False

        monte_carlo: SystemResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(500, 101, 2),
            "success": True,
            "n_paths": 500,
        }
        assert is_monte_carlo(monte_carlo) is True

    def test_discriminate_by_closed_loop(self):
        """Test using closed_loop to discriminate rollout results."""

        def is_closed_loop(result: SystemResult) -> bool:
            return result.get("closed_loop", False)

        simulation: SystemResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
            "method": "rk4",
        }
        assert is_closed_loop(simulation) is False

        rollout: SystemResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "u": np.random.randn(100, 1),
            "success": True,
            "closed_loop": True,
        }
        assert is_closed_loop(rollout) is True

    def test_discriminate_by_sde_type(self):
        """Test using sde_type to discriminate SDE results."""

        def is_stochastic(result: SystemResult) -> bool:
            return "sde_type" in result

        deterministic: SystemResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
            "solver": "RK45",
        }
        assert is_stochastic(deterministic) is False

        stochastic: SystemResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
            "sde_type": "ito",
            "noise_type": "diagonal",
        }
        assert is_stochastic(stochastic) is True

    def test_discriminate_discrete_vs_continuous(self):
        """Test discriminating discrete from continuous results."""

        def is_discrete(result: SystemResult) -> bool:
            # Discrete results have integer time steps
            t = result["t"]
            return np.issubdtype(t.dtype, np.integer)

        continuous: SystemResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
        }
        assert is_discrete(continuous) is False

        discrete: SystemResult = {
            "t": np.arange(101),
            "x": np.random.randn(101, 2),
            "dt": 0.01,
            "success": True,
        }
        assert is_discrete(discrete) is True


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_result(self):
        """Test result with only essential fields."""
        result: SystemResult = {
            "t": np.array([0.0, 1.0]),
            "x": np.array([[1.0], [2.0]]),
            "success": True,
        }

        assert len(result["t"]) == 2
        assert result["success"] is True

    def test_failed_result_minimal(self):
        """Test failed result with minimal data."""
        result: IntegrationResult = {
            "t": np.array([0.0]),
            "x": np.array([[1.0, 0.0]]),
            "success": False,
            "message": "Step size too small",
            "nfev": 10000,
            "nsteps": 0,
            "solver": "RK45",
            "integration_time": 1.0,
        }

        assert result["success"] is False
        assert result["nsteps"] == 0

    def test_high_dimensional_state(self):
        """Test with high-dimensional state space."""
        nx = 100
        T = 101

        result: SimulationResult = {
            "t": np.linspace(0, 10, T),
            "x": np.random.randn(T, nx),
            "success": True,
            "method": "rk4",
            "dt": 0.1,
        }

        assert result["x"].shape == (T, nx)

    def test_large_monte_carlo(self):
        """Test with large number of Monte Carlo paths."""
        n_paths = 10000
        T = 101
        nx = 2
        nw = 2

        result: SDESimulationResult = {
            "t": np.linspace(0, 10, T),
            "x": np.random.randn(n_paths, T, nx),
            "success": True,
            "method": "euler_maruyama",
            "dt": 0.1,
            "n_paths": n_paths,
            "noise_type": "diagonal",
            "sde_type": "ito",
            "seed": 42,
            "noise_samples": np.random.randn(n_paths, T - 1, nw),
            "diffusion_evals": (T - 1) * n_paths,
        }

        assert result["x"].shape[0] == n_paths

    def test_single_step(self):
        """Test single time step result."""
        result: DiscreteSimulationResult = {
            "t": np.array([0, 1]),
            "x": np.random.randn(2, 3),
            "u": np.random.randn(1, 2),
            "dt": 1.0,
            "success": True,
        }

        assert len(result["t"]) == 2
        assert result["x"].shape[0] == 2
        assert result["u"].shape[0] == 1


# ============================================================================
# Test Common Field Access Patterns
# ============================================================================


class TestCommonPatterns:
    """Test common access patterns across all types."""

    def test_all_types_have_t_x_success(self):
        """Test all types share common t, x, success fields."""
        results = [
            # Continuous deterministic
            {"t": np.linspace(0, 1, 11), "x": np.random.randn(11, 2), "success": True},
            # With more fields
            {
                "t": np.linspace(0, 1, 11),
                "x": np.random.randn(11, 2),
                "success": True,
                "solver": "RK45",
            },
            # Discrete
            {"t": np.arange(11), "x": np.random.randn(11, 2), "success": True, "dt": 0.1},
        ]

        for result in results:
            typed_result: SystemResult = result
            assert "t" in typed_result
            assert "x" in typed_result
            assert "success" in typed_result

    def test_time_and_state_shape_consistency(self):
        """Test t and x shapes are consistent."""

        def check_consistency(result: SystemResult) -> bool:
            t = result["t"]
            x = result["x"]
            if x.ndim == 2:
                return len(t) == x.shape[0]
            else:  # Monte Carlo: (n_paths, T, nx)
                return len(t) == x.shape[1]

        # Single trajectory
        single: SystemResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(101, 2),
            "success": True,
        }
        assert check_consistency(single) is True

        # Monte Carlo
        mc: SystemResult = {
            "t": np.linspace(0, 10, 101),
            "x": np.random.randn(500, 101, 2),
            "success": True,
        }
        assert check_consistency(mc) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
