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
Unit Tests for Robustness and Uncertainty Types

Tests TypedDict definitions and usage patterns for robust control
and uncertainty quantification types.
"""

import numpy as np
import pytest

from cdesym.types.robustness import (
    RiskSensitiveResult,
    RobustStabilityResult,
    StochasticMPCResult,
    StructuredSingularValueResult,
    TubeDefinition,
    TubeMPCResult,
    UncertaintySet,
)


class TestUncertaintySet:
    """Test UncertaintySet type alias."""

    def test_box_uncertainty(self):
        """Test box uncertainty representation."""
        # Box: θ ∈ [θ_min, θ_max]
        lower = np.array([0.8, 1.8])
        upper = np.array([1.2, 2.2])

        uncertainty: UncertaintySet = (lower, upper)

        theta_min, theta_max = uncertainty
        assert theta_min.shape == (2,)
        assert theta_max.shape == (2,)
        assert np.all(theta_min <= theta_max)

    def test_ellipsoid_uncertainty(self):
        """Test ellipsoid uncertainty representation."""
        # Ellipsoid: (θ - center)'P^{-1}(θ - center) ≤ 1
        center = np.array([1.0, 2.0])
        shape = np.diag([0.1, 0.2])

        uncertainty: UncertaintySet = (center, shape)

        c, P = uncertainty
        assert c.shape == (2,)
        assert P.shape == (2, 2)


class TestRobustStabilityResult:
    """Test RobustStabilityResult TypedDict."""

    def test_robust_stability_result_creation(self):
        """Test creating robust stability result."""
        result: RobustStabilityResult = {
            "robustly_stable": True,
            "worst_case_eigenvalue": -0.5 + 0.3j,
            "stability_margin": 0.5,
            "critical_parameter": np.array([0.9, 2.1]),
            "method": "polytope",
            "conservatism": 0.1,
        }

        assert result["robustly_stable"] == True
        assert result["stability_margin"] > 0
        assert result.get("conservatism", 0) >= 0

    def test_robust_stability_stable_case(self):
        """Test robustly stable system."""
        result: RobustStabilityResult = {
            "robustly_stable": True,
            "worst_case_eigenvalue": -0.8 + 0.1j,
            "stability_margin": 0.8,
            "method": "Lyapunov",
        }

        # Stable: real part < 0
        assert result["worst_case_eigenvalue"].real < 0
        assert result["robustly_stable"] == True

    def test_robust_stability_unstable_case(self):
        """Test robustly unstable system."""
        # Critical parameter causes instability
        theta_critical = np.array([1.5, 2.5])

        result: RobustStabilityResult = {
            "robustly_stable": False,
            "worst_case_eigenvalue": 0.2 + 0.5j,  # Unstable
            "stability_margin": 0.0,
            "critical_parameter": theta_critical,
            "method": "gridding",
        }

        # Unstable: real part > 0
        assert result["worst_case_eigenvalue"].real > 0
        assert result["robustly_stable"] == False
        assert result["critical_parameter"] is not None


class TestStructuredSingularValueResult:
    """Test StructuredSingularValueResult TypedDict."""

    def test_mu_analysis_result_creation(self):
        """Test creating μ-analysis result."""
        result: StructuredSingularValueResult = {
            "mu_value": 0.8,
            "robustness_margin": 1.25,  # 1/0.8
            "worst_case_uncertainty": np.random.randn(2, 2),
            "frequency_grid": np.logspace(-2, 2, 50),
            "upper_bound": 0.85,
            "lower_bound": 0.75,
        }

        assert result["mu_value"] > 0
        assert np.isclose(result["robustness_margin"], 1 / result["mu_value"])
        assert result["lower_bound"] <= result["mu_value"] <= result["upper_bound"]

    def test_mu_robust_stability(self):
        """Test μ < 1 implies robust stability."""
        # Robustly stable
        result_stable: StructuredSingularValueResult = {
            "mu_value": 0.6,
            "robustness_margin": 1 / 0.6,
            "worst_case_uncertainty": np.random.randn(2, 2),
            "upper_bound": 0.65,
            "lower_bound": 0.55,
        }

        # Unstable
        result_unstable: StructuredSingularValueResult = {
            "mu_value": 1.5,
            "robustness_margin": 1 / 1.5,
            "worst_case_uncertainty": np.random.randn(2, 2),
            "upper_bound": 1.6,
            "lower_bound": 1.4,
        }

        # μ < 1 → robustly stable
        assert result_stable["mu_value"] < 1
        assert result_unstable["mu_value"] > 1

    def test_mu_robustness_margin(self):
        """Test robustness margin calculation."""
        mu = 0.5
        margin = 1 / mu

        result: StructuredSingularValueResult = {
            "mu_value": mu,
            "robustness_margin": margin,
            "worst_case_uncertainty": np.zeros((2, 2)),
            "upper_bound": 0.55,
            "lower_bound": 0.45,
        }

        # Can tolerate ||Δ|| < margin
        assert result["robustness_margin"] == 2.0


class TestTubeDefinition:
    """Test TubeDefinition TypedDict."""

    def test_tube_definition_creation(self):
        """Test creating tube definition."""
        N = 20
        nx = 2

        tube: TubeDefinition = {
            "shape": "ellipsoid",
            "center_trajectory": np.random.randn(N + 1, nx),
            "tube_radii": np.linspace(0.1, 0.5, N + 1),
            "shape_matrices": [np.eye(nx) * 0.1 for _ in range(N + 1)],
        }

        assert tube["shape"] in ["ellipsoid", "polytope", "box"]
        assert tube["center_trajectory"].shape == (N + 1, nx)
        assert tube["tube_radii"].shape == (N + 1,)

    def test_tube_box_shape(self):
        """Test box-shaped tube."""
        tube: TubeDefinition = {
            "shape": "box",
            "center_trajectory": np.zeros((11, 2)),
            "tube_radii": np.ones(11) * 0.2,
        }

        assert tube["shape"] == "box"
        assert "shape_matrices" not in tube  # Not needed for box

    def test_tube_growth(self):
        """Test tube can grow over time."""
        # Tube grows due to disturbance accumulation
        radii = np.linspace(0.1, 1.0, 21)

        tube: TubeDefinition = {
            "shape": "ellipsoid",
            "center_trajectory": np.zeros((21, 3)),
            "tube_radii": radii,
            "shape_matrices": [np.eye(3) for _ in range(21)],
        }

        # Radii should grow
        assert tube["tube_radii"][0] < tube["tube_radii"][-1]


class TestTubeMPCResult:
    """Test TubeMPCResult TypedDict."""

    def test_tube_mpc_result_creation(self):
        """Test creating tube MPC result."""
        N = 20
        nx, nu = 2, 1

        tube_def: TubeDefinition = {
            "shape": "box",
            "center_trajectory": np.random.randn(N + 1, nx),
            "tube_radii": np.ones(N + 1) * 0.2,
        }

        result: TubeMPCResult = {
            "nominal_control": np.random.randn(N, nu),
            "feedback_control": np.random.randn(nu, nx),
            "actual_control": np.random.randn(N, nu),
            "nominal_trajectory": np.random.randn(N + 1, nx),
            "tube_definition": tube_def,
            "tightened_constraints": np.array([5.0, 5.0]) - 0.2,
        }

        assert result["nominal_control"].shape == (N, nu)
        assert result["feedback_control"].shape == (nu, nx)
        assert result["tube_definition"]["shape"] == "box"

    def test_tube_mpc_control_decomposition(self):
        """Test control decomposition u = v + K(x - x̄)."""
        # Nominal control
        v = np.array([[0.5], [0.6], [0.7]])

        # Feedback gain
        K = np.array([[1.0, 0.5]])

        # Nominal and actual states
        x_nom = np.array([[1.0, 0.5], [1.1, 0.4], [1.2, 0.3]])
        x_actual = x_nom + np.array([[0.1, 0.05], [0.05, 0.02], [0.0, 0.0]])

        # Actual control
        u = v + np.array([K @ (x_actual[i] - x_nom[i]) for i in range(3)]).reshape(-1, 1)

        result: TubeMPCResult = {
            "nominal_control": v,
            "feedback_control": K,
            "actual_control": u,
            "nominal_trajectory": x_nom,
            "tube_definition": {
                "shape": "box",
                "center_trajectory": x_nom,
                "tube_radii": np.ones(3) * 0.2,
            },
        }

        # Verify decomposition
        assert result["nominal_control"].shape == (3, 1)
        assert result["feedback_control"].shape == (1, 2)


class TestStochasticMPCResult:
    """Test StochasticMPCResult TypedDict."""

    def test_stochastic_mpc_result_creation(self):
        """Test creating stochastic MPC result."""
        N = 30
        nx, nu = 2, 1

        result: StochasticMPCResult = {
            "control_sequence": np.random.randn(N, nu),
            "predicted_mean": np.random.randn(N + 1, nx),
            "predicted_covariance": [0.1 * np.eye(nx) for _ in range(N + 1)],
            "constraint_violation_probability": 0.05,
            "expected_cost": 25.0,
            "cost_variance": 5.0,
            "robust_feasible": True,
            "chance_constraint_levels": np.ones(N) * 0.95,
        }

        assert result["control_sequence"].shape == (N, nu)
        assert result["predicted_mean"].shape == (N + 1, nx)
        assert len(result["predicted_covariance"]) == N + 1
        assert 0 <= result["constraint_violation_probability"] <= 1

    def test_stochastic_mpc_chance_constraints(self):
        """Test chance constraint satisfaction."""
        # 95% confidence level
        confidence = 0.95
        risk = 1 - confidence

        result: StochasticMPCResult = {
            "control_sequence": np.random.randn(10, 1),
            "predicted_mean": np.random.randn(11, 2),
            "predicted_covariance": [np.eye(2) * 0.1 for _ in range(11)],
            "constraint_violation_probability": risk,
            "expected_cost": 10.0,
            "cost_variance": 2.0,
            "robust_feasible": True,
            "chance_constraint_levels": np.ones(10) * confidence,
        }

        # Risk should match specified level
        assert np.isclose(result["constraint_violation_probability"], risk)
        assert np.all(result["chance_constraint_levels"] == confidence)

    def test_stochastic_mpc_uncertainty_propagation(self):
        """Test uncertainty propagation over time."""
        # Covariance should grow due to process noise
        cov_sequence = [
            0.01 * np.eye(2),
            0.02 * np.eye(2),
            0.04 * np.eye(2),
            0.08 * np.eye(2),
        ]

        result: StochasticMPCResult = {
            "control_sequence": np.random.randn(3, 1),
            "predicted_mean": np.random.randn(4, 2),
            "predicted_covariance": cov_sequence,
            "constraint_violation_probability": 0.05,
            "expected_cost": 5.0,
            "cost_variance": 1.0,
            "robust_feasible": True,
            "chance_constraint_levels": np.ones(3) * 0.95,
        }

        # Check covariance grows
        covs = result["predicted_covariance"]
        det0 = np.linalg.det(covs[0])
        det_final = np.linalg.det(covs[-1])
        assert det_final > det0


class TestRiskSensitiveResult:
    """Test RiskSensitiveResult TypedDict."""

    def test_risk_sensitive_result_creation(self):
        """Test creating risk-sensitive result."""
        result: RiskSensitiveResult = {
            "gain": np.array([[1.5, 0.8]]),
            "cost_to_go_matrix": np.array([[2.0, 0.5], [0.5, 1.5]]),
            "risk_parameter": 0.1,
            "expected_cost": 15.0,
            "cost_variance": 3.0,
            "certainty_equivalent": 16.5,
        }

        assert result["gain"].shape == (1, 2)
        assert result["cost_to_go_matrix"].shape == (2, 2)
        assert result["risk_parameter"] >= 0

    def test_risk_aversion_levels(self):
        """Test different risk aversion levels."""
        # Risk-neutral (θ = 0)
        result_neutral: RiskSensitiveResult = {
            "gain": np.array([[1.0, 0.5]]),
            "cost_to_go_matrix": np.eye(2),
            "risk_parameter": 0.0,  # Risk-neutral
            "expected_cost": 10.0,
            "cost_variance": 5.0,
            "certainty_equivalent": 10.0,
        }

        # Risk-averse (θ > 0)
        result_averse: RiskSensitiveResult = {
            "gain": np.array([[1.5, 0.8]]),  # More conservative
            "cost_to_go_matrix": 2 * np.eye(2),
            "risk_parameter": 0.5,  # Risk-averse
            "expected_cost": 12.0,
            "cost_variance": 2.0,  # Lower variance
            "certainty_equivalent": 13.0,
        }

        # Risk-averse has lower variance
        assert result_averse["cost_variance"] < result_neutral["cost_variance"]
        assert result_averse["risk_parameter"] > result_neutral["risk_parameter"]


class TestPracticalUseCases:
    """Test realistic usage patterns."""

    def test_robust_stability_check(self):
        """Test robust stability verification workflow."""
        # Uncertain system
        uncertainty = (np.array([0.8, 1.8]), np.array([1.2, 2.2]))

        result: RobustStabilityResult = {
            "robustly_stable": True,
            "worst_case_eigenvalue": -0.3 + 0.2j,
            "stability_margin": 0.3,
            "critical_parameter": np.array([1.2, 1.8]),
            "method": "polytope",
            "conservatism": 0.2,
        }

        if result["robustly_stable"]:
            margin = result["stability_margin"]
            # Can design controller with this margin
            assert margin > 0

    def test_tube_mpc_workflow(self):
        """Test tube MPC workflow."""
        # Solve tube MPC
        result: TubeMPCResult = {
            "nominal_control": np.array([[0.5], [0.4]]),
            "feedback_control": np.array([[1.0, 0.5]]),
            "actual_control": np.array([[0.55], [0.42]]),
            "nominal_trajectory": np.array([[1.0, 0.5], [0.9, 0.4], [0.8, 0.3]]),
            "tube_definition": {
                "shape": "box",
                "center_trajectory": np.zeros((3, 2)),
                "tube_radii": np.ones(3) * 0.2,
            },
        }

        # Apply first control
        u = result["actual_control"][0]
        assert u.shape == (1,)

    def test_stochastic_mpc_with_chance_constraints(self):
        """Test stochastic MPC with chance constraints."""
        result: StochasticMPCResult = {
            "control_sequence": np.random.randn(20, 1),
            "predicted_mean": np.random.randn(21, 2),
            "predicted_covariance": [0.1 * np.eye(2) for _ in range(21)],
            "constraint_violation_probability": 0.05,
            "expected_cost": 30.0,
            "cost_variance": 8.0,
            "robust_feasible": True,
            "chance_constraint_levels": np.ones(20) * 0.95,
        }

        # Check feasibility
        if result["robust_feasible"]:
            risk = result["constraint_violation_probability"]
            assert risk <= 0.05  # 95% confidence


class TestNumericalProperties:
    """Test numerical properties of results."""

    def test_stability_margin_non_negative(self):
        """Test stability margin is non-negative."""
        result: RobustStabilityResult = {
            "robustly_stable": True,
            "worst_case_eigenvalue": -0.5 + 0.1j,
            "stability_margin": 0.5,
            "method": "Lyapunov",
        }

        assert result["stability_margin"] >= 0

    def test_mu_value_positive(self):
        """Test μ is positive."""
        result: StructuredSingularValueResult = {
            "mu_value": 0.8,
            "robustness_margin": 1.25,
            "worst_case_uncertainty": np.zeros((2, 2)),
            "upper_bound": 0.85,
            "lower_bound": 0.75,
        }

        assert result["mu_value"] > 0
        assert result["robustness_margin"] > 0

    def test_risk_probability_bounds(self):
        """Test probability is in [0, 1]."""
        result: StochasticMPCResult = {
            "control_sequence": np.zeros((10, 1)),
            "predicted_mean": np.zeros((11, 2)),
            "predicted_covariance": [np.eye(2) for _ in range(11)],
            "constraint_violation_probability": 0.05,
            "expected_cost": 10.0,
            "cost_variance": 2.0,
            "robust_feasible": True,
            "chance_constraint_levels": np.ones(10) * 0.95,
        }

        assert 0 <= result["constraint_violation_probability"] <= 1
        assert np.all(result["chance_constraint_levels"] >= 0)
        assert np.all(result["chance_constraint_levels"] <= 1)


class TestDocumentationExamples:
    """Test that documentation examples work."""

    def test_robust_stability_example(self):
        """Test RobustStabilityResult example from docstring."""
        result: RobustStabilityResult = {
            "robustly_stable": True,
            "worst_case_eigenvalue": -0.5 + 0.3j,
            "stability_margin": 0.5,
            "method": "polytope",
        }

        if result["robustly_stable"]:
            assert result["stability_margin"] > 0

    def test_tube_mpc_example(self):
        """Test TubeMPCResult example structure."""
        result: TubeMPCResult = {
            "nominal_control": np.array([[0.5]]),
            "feedback_control": np.array([[1.0, 0.5]]),
            "nominal_trajectory": np.array([[1.0, 0.5], [0.9, 0.4]]),
            "tube_definition": {
                "shape": "box",
                "center_trajectory": np.zeros((2, 2)),
                "tube_radii": np.ones(2) * 0.2,
            },
        }

        assert result["feedback_control"].shape == (1, 2)


class TestFieldPresence:
    """Test that all fields are accessible."""

    def test_robust_stability_has_required_fields(self):
        """Test RobustStabilityResult has core fields."""
        result: RobustStabilityResult = {
            "robustly_stable": True,
            "worst_case_eigenvalue": -0.5,
            "stability_margin": 0.5,
            "method": "Lyapunov",
        }

        assert "robustly_stable" in result
        assert "stability_margin" in result

    def test_stochastic_mpc_has_required_fields(self):
        """Test StochasticMPCResult has core fields."""
        result: StochasticMPCResult = {
            "control_sequence": np.zeros((10, 1)),
            "predicted_mean": np.zeros((11, 2)),
            "predicted_covariance": [np.eye(2) for _ in range(11)],
            "expected_cost": 10.0,
        }

        assert "control_sequence" in result
        assert "predicted_mean" in result
        assert "predicted_covariance" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
