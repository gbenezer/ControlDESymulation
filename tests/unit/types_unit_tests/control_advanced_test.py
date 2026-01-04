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
Unit Tests for Advanced Control Theory Types

Tests TypedDict definitions and usage patterns for advanced control
design result types (MPC, H∞, adaptive, sliding mode, etc.).
"""

import numpy as np
import pytest

from cdesym.types.control_advanced import (
    AdaptiveControlResult,
    H2ControlResult,
    HInfControlResult,
    LMIResult,
    MHEResult,
    MPCResult,
    SlidingModeResult,
)


class TestMPCResult:
    """Test MPCResult TypedDict."""

    def test_mpc_result_creation(self):
        """Test creating MPCResult instance."""
        N = 10  # Horizon
        nx, nu = 2, 1

        result: MPCResult = {
            "control_sequence": np.zeros((N, nu)),
            "predicted_trajectory": np.zeros((N + 1, nx)),
            "cost": 5.2,
            "success": True,
            "iterations": 15,
            "solve_time": 0.01,
        }

        assert result["control_sequence"].shape == (10, 1)
        assert result["predicted_trajectory"].shape == (11, 2)
        assert result["success"] == True
        assert result["cost"] > 0

    def test_mpc_result_with_constraints(self):
        """Test MPCResult with constraint information."""
        result: MPCResult = {
            "control_sequence": np.random.randn(10, 1),
            "predicted_trajectory": np.random.randn(11, 2),
            "cost": 3.5,
            "success": True,
            "iterations": 20,
            "solve_time": 0.015,
            "constraint_violations": np.array([0.0, 0.0, 0.1]),
            "dual_variables": np.random.randn(10),
        }

        assert result["constraint_violations"] is not None
        assert result["dual_variables"] is not None

    def test_mpc_result_receding_horizon_usage(self):
        """Test using MPC result in receding horizon pattern."""
        # Simulate MPC solution
        result: MPCResult = {
            "control_sequence": np.array([[0.5], [0.3], [0.1]]),
            "predicted_trajectory": np.random.randn(4, 2),
            "cost": 2.1,
            "success": True,
            "iterations": 10,
            "solve_time": 0.008,
        }

        # Apply first control only (receding horizon principle)
        u_apply = result["control_sequence"][0]

        assert u_apply.shape == (1,)
        assert isinstance(float(u_apply[0]), float)

    def test_mpc_result_failed_optimization(self):
        """Test MPC result when optimization fails."""
        result: MPCResult = {
            "control_sequence": np.zeros((10, 1)),  # Fallback control
            "predicted_trajectory": np.zeros((11, 2)),
            "cost": float("inf"),
            "success": False,
            "iterations": 100,  # Max iterations reached
            "solve_time": 1.0,
        }

        assert result["success"] == False
        assert np.isinf(result["cost"])


class TestMHEResult:
    """Test MHEResult TypedDict."""

    def test_mhe_result_creation(self):
        """Test creating MHEResult instance."""
        result: MHEResult = {
            "state_estimate": np.array([1.0, 0.5]),
            "covariance_estimate": np.eye(2) * 0.1,
            "state_trajectory": np.random.randn(10, 2),
            "cost": 1.5,
            "success": True,
            "solve_time": 0.012,
        }

        assert result["state_estimate"].shape == (2,)
        assert result["covariance_estimate"].shape == (2, 2)
        assert result["state_trajectory"].shape == (10, 2)

    def test_mhe_result_with_innovations(self):
        """Test MHEResult with innovation sequence."""
        result: MHEResult = {
            "state_estimate": np.array([0.8, 0.3]),
            "covariance_estimate": np.diag([0.05, 0.03]),
            "state_trajectory": np.random.randn(10, 2),
            "cost": 2.0,
            "success": True,
            "solve_time": 0.015,
            "innovation_sequence": np.random.randn(10, 1),
        }

        assert result["innovation_sequence"] is not None
        assert result["innovation_sequence"].shape == (10, 1)

    def test_mhe_result_usage(self):
        """Test using MHE result for state estimation."""
        result: MHEResult = {
            "state_estimate": np.array([1.2, -0.3]),
            "covariance_estimate": np.diag([0.1, 0.08]),
            "state_trajectory": np.random.randn(5, 2),
            "cost": 0.8,
            "success": True,
            "solve_time": 0.01,
        }

        # Use estimate
        x_hat = result["state_estimate"]
        P_hat = result["covariance_estimate"]

        # Verify covariance properties
        assert np.allclose(P_hat, P_hat.T)  # Symmetric
        eigenvalues = np.linalg.eigvals(P_hat)
        assert np.all(eigenvalues > 0)  # Positive definite


class TestH2ControlResult:
    """Test H2ControlResult TypedDict."""

    def test_h2_control_result_creation(self):
        """Test creating H2ControlResult instance."""
        result: H2ControlResult = {
            "gain": np.array([[1.5, 0.8]]),
            "h2_norm": 2.3,
            "cost_to_go": np.diag([5.0, 3.0]),
            "closed_loop_stable": True,
            "closed_loop_poles": np.array([-2.0, -3.0]),
        }

        assert result["gain"].shape == (1, 2)
        assert result["h2_norm"] > 0
        assert result["closed_loop_stable"] == True

    def test_h2_control_result_stability(self):
        """Test verifying stability from H2 result."""
        result: H2ControlResult = {
            "gain": np.array([[2.0, 1.5]]),
            "h2_norm": 3.1,
            "cost_to_go": np.eye(2) * 10.0,
            "closed_loop_stable": True,
            "closed_loop_poles": np.array([-1.5, -2.5]),
        }

        # Verify continuous-time stability
        poles = result["closed_loop_poles"]
        is_stable = bool(np.all(np.real(poles) < 0))

        assert is_stable == True
        assert result["closed_loop_stable"] == True

    def test_h2_control_result_usage(self):
        """Test using H2 controller."""
        result: H2ControlResult = {
            "gain": np.array([[1.0, 0.5]]),
            "h2_norm": 1.8,
            "cost_to_go": np.eye(2),
            "closed_loop_stable": True,
            "closed_loop_poles": np.array([-3.0, -4.0]),
        }

        # Apply control
        K = result["gain"]
        x = np.array([1.0, 0.0])
        u = -K @ x

        assert u.shape == (1,)
        assert result["h2_norm"] < 2.0  # Performance bound


class TestHInfControlResult:
    """Test HInfControlResult TypedDict."""

    def test_hinf_control_result_creation(self):
        """Test creating HInfControlResult instance."""
        result: HInfControlResult = {
            "gain": np.array([[1.2, 0.9]]),
            "hinf_norm": 1.8,
            "gamma": 2.0,
            "central_solution": np.diag([8.0, 5.0]),
            "feasible": True,
            "robustness_margin": 0.5,
        }

        assert result["hinf_norm"] <= result["gamma"]
        assert result["feasible"] == True
        assert result["robustness_margin"] > 0

    def test_hinf_control_result_infeasible(self):
        """Test HInf result when gamma too small."""
        result: HInfControlResult = {
            "gain": np.zeros((1, 2)),
            "hinf_norm": float("inf"),
            "gamma": 0.5,
            "central_solution": np.zeros((2, 2)),
            "feasible": False,
            "robustness_margin": 0.0,
        }

        assert result["feasible"] == False
        assert result["gamma"] < 1.0

    def test_hinf_control_result_performance_bound(self):
        """Test HInf performance guarantee."""
        result: HInfControlResult = {
            "gain": np.array([[1.5, 1.0]]),
            "hinf_norm": 1.95,
            "gamma": 2.0,
            "central_solution": np.eye(2) * 5.0,
            "feasible": True,
            "robustness_margin": 0.3,
        }

        # Verify performance bound
        assert result["hinf_norm"] <= result["gamma"]

        # Margin indicates how much "room" we have
        # Higher margin = more robust
        assert result["robustness_margin"] > 0


class TestLMIResult:
    """Test LMIResult TypedDict."""

    def test_lmi_result_creation(self):
        """Test creating LMIResult instance."""
        P_solution = np.diag([5.0, 3.0])

        result: LMIResult = {
            "decision_variables": {"P": P_solution},
            "objective_value": 0.0,
            "feasible": True,
            "solver": "cvxpy",
            "solve_time": 0.05,
        }

        assert "P" in result["decision_variables"]
        assert result["feasible"] == True
        assert result["solver"] == "cvxpy"

    def test_lmi_result_multiple_variables(self):
        """Test LMI result with multiple decision variables."""
        result: LMIResult = {
            "decision_variables": {
                "P": np.eye(2) * 5.0,
                "Y": np.array([[1.0, 0.5]]),
                "gamma": np.array([2.0]),
            },
            "objective_value": 2.0,
            "feasible": True,
            "solver": "mosek",
            "solve_time": 0.08,
            "condition_number": 10.5,
        }

        assert len(result["decision_variables"]) == 3
        assert "condition_number" in result

    def test_lmi_result_infeasible(self):
        """Test LMI result when problem infeasible."""
        result: LMIResult = {
            "decision_variables": {},
            "objective_value": float("inf"),
            "feasible": False,
            "solver": "cvxpy",
            "solve_time": 0.01,
        }

        assert result["feasible"] == False
        assert len(result["decision_variables"]) == 0


class TestAdaptiveControlResult:
    """Test AdaptiveControlResult TypedDict."""

    def test_adaptive_control_result_creation(self):
        """Test creating AdaptiveControlResult instance."""
        result: AdaptiveControlResult = {
            "current_gain": np.array([[1.2, 0.8]]),
            "parameter_estimate": np.array([2.5, 1.3, 0.8]),
            "parameter_covariance": np.diag([0.1, 0.05, 0.08]),
            "adaptation_rate": 0.1,
            "tracking_error": 0.05,
        }

        assert result["current_gain"].shape == (1, 2)
        assert result["parameter_estimate"].shape == (3,)
        assert result["tracking_error"] >= 0

    def test_adaptive_control_result_with_error(self):
        """Test AdaptiveControlResult with parameter error (testing)."""
        theta_true = np.array([2.0, 1.0, 1.0])
        theta_hat = np.array([1.8, 1.1, 0.9])

        result: AdaptiveControlResult = {
            "current_gain": np.array([[1.0, 0.5]]),
            "parameter_estimate": theta_hat,
            "parameter_covariance": np.eye(3) * 0.1,
            "adaptation_rate": 0.05,
            "tracking_error": 0.1,
            "parameter_error": theta_hat - theta_true,
        }

        assert result["parameter_error"] is not None
        error_norm = np.linalg.norm(result["parameter_error"])
        assert error_norm > 0

    def test_adaptive_control_convergence(self):
        """Test adaptive control convergence pattern."""
        # Simulate decreasing tracking error over time
        errors = []
        for k in range(5):
            result: AdaptiveControlResult = {
                "current_gain": np.array([[1.0 + 0.1 * k, 0.5]]),
                "parameter_estimate": np.array([2.0 - 0.1 * k, 1.0]),
                "parameter_covariance": np.eye(2) * (0.5 / (k + 1)),
                "adaptation_rate": 0.1,
                "tracking_error": 1.0 / (k + 1),
            }
            errors.append(result["tracking_error"])

        # Tracking error should decrease
        assert errors[-1] < errors[0]


class TestSlidingModeResult:
    """Test SlidingModeResult TypedDict."""

    def test_sliding_mode_result_creation(self):
        """Test creating SlidingModeResult instance."""
        result: SlidingModeResult = {
            "control": np.array([5.0]),
            "sliding_variable": np.array([0.01]),
            "on_sliding_surface": True,
            "reaching_time_estimate": None,
            "chattering_magnitude": 0.5,
        }

        assert result["control"].shape == (1,)
        assert result["on_sliding_surface"] == True
        assert result["chattering_magnitude"] >= 0

    def test_sliding_mode_result_reaching_phase(self):
        """Test SMC result during reaching phase."""
        result: SlidingModeResult = {
            "control": np.array([10.0]),
            "sliding_variable": np.array([0.5]),
            "on_sliding_surface": False,
            "reaching_time_estimate": 0.3,
            "chattering_magnitude": 2.0,
        }

        assert result["on_sliding_surface"] == False
        assert result["reaching_time_estimate"] is not None
        assert result["reaching_time_estimate"] > 0

    def test_sliding_mode_result_chattering(self):
        """Test monitoring chattering in SMC."""
        # High chattering scenario
        result_high: SlidingModeResult = {
            "control": np.array([15.0]),
            "sliding_variable": np.array([0.001]),
            "on_sliding_surface": True,
            "reaching_time_estimate": None,
            "chattering_magnitude": 5.0,
        }

        # Low chattering scenario (boundary layer)
        result_low: SlidingModeResult = {
            "control": np.array([3.0]),
            "sliding_variable": np.array([0.05]),
            "on_sliding_surface": False,
            "reaching_time_estimate": 0.1,
            "chattering_magnitude": 0.2,
        }

        assert result_high["chattering_magnitude"] > result_low["chattering_magnitude"]

    def test_sliding_mode_multi_input(self):
        """Test SMC with multiple control inputs."""
        result: SlidingModeResult = {
            "control": np.array([5.0, -3.0]),
            "sliding_variable": np.array([0.01, -0.02]),
            "on_sliding_surface": True,
            "reaching_time_estimate": None,
            "chattering_magnitude": 1.2,
        }

        assert result["control"].shape == (2,)
        assert result["sliding_variable"].shape == (2,)


class TestPracticalUseCases:
    """Test realistic usage patterns."""

    def test_mpc_closed_loop(self):
        """Test MPC in closed-loop simulation."""
        # Simulate MPC control loop
        x = np.array([1.0, 0.0])
        trajectory = [x]

        for k in range(3):
            # Solve MPC
            result: MPCResult = {
                "control_sequence": np.random.randn(10, 1) * 0.1,
                "predicted_trajectory": np.random.randn(11, 2),
                "cost": 1.0 + k * 0.1,
                "success": True,
                "iterations": 10 + k,
                "solve_time": 0.01,
            }

            # Apply first control
            u = result["control_sequence"][0]

            # Simulate system (dummy)
            x = x * 0.9 + u.flatten() * 0.1
            trajectory.append(x.copy())

        trajectory = np.array(trajectory)
        assert trajectory.shape == (4, 2)

    def test_h2_hinf_comparison(self):
        """Test comparing H2 and HInf designs."""
        # H2 design (optimal for stochastic)
        h2_result: H2ControlResult = {
            "gain": np.array([[1.5, 1.0]]),
            "h2_norm": 2.5,
            "cost_to_go": np.eye(2) * 5.0,
            "closed_loop_stable": True,
            "closed_loop_poles": np.array([-2.0, -3.0]),
        }

        # HInf design (robust to worst-case)
        hinf_result: HInfControlResult = {
            "gain": np.array([[2.0, 1.5]]),
            "hinf_norm": 1.8,
            "gamma": 2.0,
            "central_solution": np.eye(2) * 8.0,
            "feasible": True,
            "robustness_margin": 0.4,
        }

        # HInf typically more conservative (higher gain)
        assert np.linalg.norm(hinf_result["gain"]) >= np.linalg.norm(h2_result["gain"])

    def test_adaptive_mrac_workflow(self):
        """Test Model Reference Adaptive Control workflow."""
        # Initialize
        theta_initial = np.array([1.0, 1.0])

        # Adaptation loop
        for k in range(3):
            result: AdaptiveControlResult = {
                "current_gain": np.array([[1.0 + 0.2 * k, 0.5]]),
                "parameter_estimate": theta_initial + k * np.array([0.1, -0.05]),
                "parameter_covariance": np.eye(2) * 0.5 / (k + 1),
                "adaptation_rate": 0.1,
                "tracking_error": 1.0 / (k + 1),
            }

            # Use adapted gain
            K = result["current_gain"]
            theta_hat = result["parameter_estimate"]

            assert K.shape == (1, 2)
            assert theta_hat.shape == (2,)


class TestDocumentationExamples:
    """Test that documentation examples work."""

    def test_mpc_result_example(self):
        """Test MPCResult example from docstring."""
        # Simulate MPC solution
        result: MPCResult = {
            "control_sequence": np.random.randn(20, 1),
            "predicted_trajectory": np.random.randn(21, 2),
            "cost": 5.3,
            "success": True,
            "iterations": 15,
            "solve_time": 0.012,
        }

        # Apply first control
        u_apply = result["control_sequence"][0]

        assert u_apply.shape == (1,)
        if result["success"]:
            assert result["cost"] > 0

    def test_h2_result_example(self):
        """Test H2ControlResult example structure."""
        result: H2ControlResult = {
            "gain": np.array([[1.5, 0.8]]),
            "h2_norm": 2.1,
            "cost_to_go": np.eye(2) * 5.0,
            "closed_loop_stable": True,
            "closed_loop_poles": np.array([-2.0, -3.0]),
        }

        K = result["gain"]
        assert K.shape == (1, 2)
        assert result["closed_loop_stable"] == True

    def test_sliding_mode_example(self):
        """Test SlidingModeResult example structure."""
        result: SlidingModeResult = {
            "control": np.array([5.0]),
            "sliding_variable": np.array([0.01]),
            "on_sliding_surface": True,
            "reaching_time_estimate": None,
            "chattering_magnitude": 0.8,
        }

        s = result["sliding_variable"]
        if result["on_sliding_surface"]:
            assert np.abs(s[0]) < 0.1


class TestFieldPresence:
    """Test that all required fields are accessible."""

    def test_mpc_result_has_required_fields(self):
        """Test MPCResult has core fields."""
        result: MPCResult = {
            "control_sequence": np.zeros((10, 1)),
            "predicted_trajectory": np.zeros((11, 2)),
            "cost": 1.0,
            "success": True,
            "iterations": 10,
            "solve_time": 0.01,
        }

        assert "control_sequence" in result
        assert "cost" in result
        assert "success" in result

    def test_hinf_result_has_required_fields(self):
        """Test HInfControlResult has all fields."""
        result: HInfControlResult = {
            "gain": np.array([[1.0]]),
            "hinf_norm": 1.5,
            "gamma": 2.0,
            "central_solution": np.eye(2),
            "feasible": True,
            "robustness_margin": 0.5,
        }

        assert "gain" in result
        assert "hinf_norm" in result
        assert "gamma" in result
        assert "feasible" in result

    def test_adaptive_result_has_required_fields(self):
        """Test AdaptiveControlResult has all fields."""
        result: AdaptiveControlResult = {
            "current_gain": np.array([[1.0]]),
            "parameter_estimate": np.array([1.0]),
            "parameter_covariance": np.eye(1),
            "adaptation_rate": 0.1,
            "tracking_error": 0.05,
        }

        assert "current_gain" in result
        assert "parameter_estimate" in result
        assert "tracking_error" in result

    def test_sliding_mode_result_has_required_fields(self):
        """Test SlidingModeResult has all fields."""
        result: SlidingModeResult = {
            "control": np.array([1.0]),
            "sliding_variable": np.array([0.01]),
            "on_sliding_surface": True,
            "reaching_time_estimate": None,
            "chattering_magnitude": 0.5,
        }

        assert "control" in result
        assert "sliding_variable" in result
        assert "on_sliding_surface" in result
        assert "chattering_magnitude" in result


class TestOptionalFields:
    """Test optional field handling."""

    def test_mpc_minimal_result(self):
        """Test MPCResult with only required fields."""
        result: MPCResult = {
            "control_sequence": np.zeros((10, 1)),
            "predicted_trajectory": np.zeros((11, 2)),
            "cost": 1.0,
            "success": True,
            "iterations": 10,
            "solve_time": 0.01,
        }

        # Optional fields should be absent or None
        assert "constraint_violations" not in result or result.get("constraint_violations") is None
        assert "dual_variables" not in result or result.get("dual_variables") is None

    def test_mhe_minimal_result(self):
        """Test MHEResult with minimal fields."""
        result: MHEResult = {
            "state_estimate": np.zeros(2),
            "covariance_estimate": np.eye(2),
            "state_trajectory": np.zeros((10, 2)),
            "cost": 1.0,
            "success": True,
            "solve_time": 0.01,
        }

        assert "innovation_sequence" not in result or result.get("innovation_sequence") is None

    def test_lmi_minimal_result(self):
        """Test LMIResult with minimal fields."""
        result: LMIResult = {
            "decision_variables": {"P": np.eye(2)},
            "objective_value": 0.0,
            "feasible": True,
        }

        # These fields are optional
        assert "solver" not in result or result.get("solver") is not None
        assert "solve_time" not in result or result.get("solve_time") is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_mpc_empty_horizon(self):
        """Test MPC with minimal horizon."""
        result: MPCResult = {
            "control_sequence": np.zeros((1, 1)),
            "predicted_trajectory": np.zeros((2, 2)),
            "cost": 0.0,
            "success": True,
            "iterations": 1,
            "solve_time": 0.001,
        }

        assert result["control_sequence"].shape[0] == 1
        assert result["predicted_trajectory"].shape[0] == 2

    def test_hinf_gamma_iteration(self):
        """Test HInf with different gamma values."""
        # Too small gamma - infeasible
        result_small: HInfControlResult = {
            "gain": np.zeros((1, 2)),
            "hinf_norm": float("inf"),
            "gamma": 0.1,
            "central_solution": np.zeros((2, 2)),
            "feasible": False,
            "robustness_margin": 0.0,
        }

        # Larger gamma - feasible
        result_large: HInfControlResult = {
            "gain": np.array([[1.0, 0.5]]),
            "hinf_norm": 1.8,
            "gamma": 5.0,
            "central_solution": np.eye(2) * 10.0,
            "feasible": True,
            "robustness_margin": 3.2,
        }

        assert result_small["feasible"] == False
        assert result_large["feasible"] == True
        assert result_large["hinf_norm"] < result_large["gamma"]

    def test_adaptive_zero_tracking_error(self):
        """Test adaptive control with perfect tracking."""
        result: AdaptiveControlResult = {
            "current_gain": np.array([[1.0, 0.5]]),
            "parameter_estimate": np.array([2.0, 1.0]),
            "parameter_covariance": np.zeros((2, 2)),  # Zero uncertainty
            "adaptation_rate": 0.0,  # No adaptation needed
            "tracking_error": 0.0,
        }

        assert result["tracking_error"] == 0.0
        assert result["adaptation_rate"] == 0.0

    def test_sliding_mode_on_surface(self):
        """Test sliding mode exactly on surface."""
        result: SlidingModeResult = {
            "control": np.array([0.0]),
            "sliding_variable": np.array([0.0]),
            "on_sliding_surface": True,
            "reaching_time_estimate": None,
            "chattering_magnitude": 0.0,
        }

        assert np.all(result["sliding_variable"] == 0.0)
        assert result["on_sliding_surface"] == True


class TestNumericalProperties:
    """Test numerical properties of results."""

    def test_mpc_cost_non_negative(self):
        """Test that MPC cost is non-negative."""
        result: MPCResult = {
            "control_sequence": np.random.randn(10, 1),
            "predicted_trajectory": np.random.randn(11, 2),
            "cost": 5.3,
            "success": True,
            "iterations": 15,
            "solve_time": 0.01,
        }

        assert result["cost"] >= 0

    def test_h2_norm_positive(self):
        """Test that H2 norm is positive."""
        result: H2ControlResult = {
            "gain": np.array([[1.0, 0.5]]),
            "h2_norm": 2.5,
            "cost_to_go": np.eye(2),
            "closed_loop_stable": True,
            "closed_loop_poles": np.array([-1.0, -2.0]),
        }

        assert result["h2_norm"] > 0

    def test_hinf_performance_bound(self):
        """Test that HInf norm satisfies bound."""
        result: HInfControlResult = {
            "gain": np.array([[1.0, 0.5]]),
            "hinf_norm": 1.95,
            "gamma": 2.0,
            "central_solution": np.eye(2),
            "feasible": True,
            "robustness_margin": 0.05,
        }

        assert result["hinf_norm"] <= result["gamma"]
        assert result["robustness_margin"] >= 0

    def test_mhe_covariance_positive_definite(self):
        """Test that MHE covariance is positive definite."""
        P = np.array([[1.0, 0.1], [0.1, 0.5]])

        result: MHEResult = {
            "state_estimate": np.zeros(2),
            "covariance_estimate": P,
            "state_trajectory": np.zeros((5, 2)),
            "cost": 1.0,
            "success": True,
            "solve_time": 0.01,
        }

        # Check positive definite
        eigenvalues = np.linalg.eigvals(result["covariance_estimate"])
        assert np.all(eigenvalues > 0)

        # Check symmetric
        P_result = result["covariance_estimate"]
        assert np.allclose(P_result, P_result.T)

    def test_adaptive_parameter_covariance_positive_semidefinite(self):
        """Test that adaptive parameter covariance is PSD."""
        result: AdaptiveControlResult = {
            "current_gain": np.array([[1.0]]),
            "parameter_estimate": np.array([1.0, 2.0]),
            "parameter_covariance": np.diag([0.1, 0.05]),
            "adaptation_rate": 0.1,
            "tracking_error": 0.02,
        }

        P_theta = result["parameter_covariance"]
        eigenvalues = np.linalg.eigvals(P_theta)
        assert np.all(eigenvalues >= 0)  # PSD (can be zero)


class TestIntegration:
    """Test integration between different result types."""

    def test_mpc_mhe_dual(self):
        """Test that MPC and MHE are duals (same structure)."""
        # MPC predicts forward
        mpc_result: MPCResult = {
            "control_sequence": np.random.randn(10, 1),
            "predicted_trajectory": np.random.randn(11, 2),
            "cost": 5.0,
            "success": True,
            "iterations": 20,
            "solve_time": 0.015,
        }

        # MHE estimates backward
        mhe_result: MHEResult = {
            "state_estimate": np.zeros(2),
            "covariance_estimate": np.eye(2),
            "state_trajectory": np.random.randn(10, 2),
            "cost": 3.0,
            "success": True,
            "solve_time": 0.012,
        }

        # Both solve optimization problems with similar structure
        assert mpc_result["success"] == True
        assert mhe_result["success"] == True
        assert "cost" in mpc_result
        assert "cost" in mhe_result

    def test_h2_lqr_relationship(self):
        """Test relationship between H2 and LQR (similar for certain problems)."""
        # H2 control
        h2_result: H2ControlResult = {
            "gain": np.array([[1.0, 0.5]]),
            "h2_norm": 2.0,
            "cost_to_go": np.eye(2) * 5.0,
            "closed_loop_stable": True,
            "closed_loop_poles": np.array([-2.0, -3.0]),
        }

        # For certain problem formulations, H2 = LQR
        # Both should have:
        # - Gain matrix
        # - Cost-to-go (Riccati solution)
        # - Stability guarantee
        assert h2_result["gain"].shape == (1, 2)
        assert h2_result["cost_to_go"].shape == (2, 2)
        assert h2_result["closed_loop_stable"] == True


class TestRobustness:
    """Test robustness and error handling."""

    def test_mpc_infeasible_problem(self):
        """Test MPC when constraints make problem infeasible."""
        result: MPCResult = {
            "control_sequence": np.zeros((10, 1)),  # Fallback
            "predicted_trajectory": np.zeros((11, 2)),
            "cost": float("inf"),
            "success": False,
            "iterations": 100,
            "solve_time": 1.0,
        }

        assert result["success"] == False
        assert np.isinf(result["cost"])

    def test_lmi_numerical_issues(self):
        """Test LMI with poor numerical conditioning."""
        result: LMIResult = {
            "decision_variables": {"P": np.eye(2) * 1e-10},
            "objective_value": 0.0,
            "feasible": True,
            "solver": "cvxpy",
            "solve_time": 0.1,
            "condition_number": 1e12,  # Very poorly conditioned
        }

        assert result["feasible"] == True
        assert result["condition_number"] > 1e6  # Warning threshold

    def test_adaptive_divergence(self):
        """Test adaptive control with diverging parameters."""
        result: AdaptiveControlResult = {
            "current_gain": np.array([[10.0, 5.0]]),  # Very large
            "parameter_estimate": np.array([100.0, 50.0]),  # Diverging
            "parameter_covariance": np.eye(2) * 10.0,  # Large uncertainty
            "adaptation_rate": 1.0,  # Too aggressive
            "tracking_error": 10.0,  # Poor tracking
        }

        # Large tracking error indicates problems
        assert result["tracking_error"] > 5.0
        # Could indicate need to reduce adaptation rate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
