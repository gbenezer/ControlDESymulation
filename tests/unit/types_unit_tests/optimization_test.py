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
Unit Tests for Optimization Types

Tests TypedDict definitions and usage patterns for optimization
result types.
"""

import numpy as np
import pytest

from src.types.optimization import (
    ConstrainedOptimizationResult,
    ConvexOptimizationResult,
    OptimizationBounds,
    OptimizationResult,
    ParameterOptimizationResult,
    TrajectoryOptimizationResult,
)


class TestOptimizationBounds:
    """Test OptimizationBounds TypedDict."""

    def test_bounds_creation(self):
        """Test creating OptimizationBounds instance."""
        bounds: OptimizationBounds = {
            "lower": np.array([-10.0, -5.0]),
            "upper": np.array([10.0, 5.0]),
        }

        assert bounds["lower"].shape == (2,)
        assert bounds["upper"].shape == (2,)
        assert np.all(bounds["lower"] <= bounds["upper"])

    def test_bounds_validity(self):
        """Test that lower <= upper."""
        bounds: OptimizationBounds = {
            "lower": np.array([0.0, -1.0, -np.inf]),
            "upper": np.array([1.0, 1.0, 0.0]),
        }

        # All lower bounds should be <= upper bounds
        assert np.all(bounds["lower"] <= bounds["upper"])

    def test_unbounded_variables(self):
        """Test representing unbounded variables."""
        bounds: OptimizationBounds = {
            "lower": np.array([-np.inf, -np.inf]),
            "upper": np.array([np.inf, np.inf]),
        }

        assert np.all(np.isinf(bounds["lower"]))
        assert np.all(np.isinf(bounds["upper"]))


class TestOptimizationResult:
    """Test OptimizationResult TypedDict."""

    def test_optimization_result_creation(self):
        """Test creating OptimizationResult instance."""
        result: OptimizationResult = {
            "x": np.array([1.0, 2.0]),
            "fun": 5.0,
            "success": True,
            "message": "Optimization terminated successfully",
            "nit": 10,
            "nfev": 50,
            "njev": 12,
        }

        assert result["x"].shape == (2,)
        assert result["success"] == True
        assert result["nit"] > 0

    def test_optimization_result_minimal(self):
        """Test OptimizationResult with minimal fields."""
        result: OptimizationResult = {
            "x": np.array([3.0]),
            "fun": 9.0,
            "success": True,
        }

        assert "x" in result
        assert "fun" in result
        assert "success" in result

    def test_optimization_result_failed(self):
        """Test OptimizationResult for failed optimization."""
        result: OptimizationResult = {
            "x": np.array([0.0, 0.0]),  # Initial guess
            "fun": float("inf"),
            "success": False,
            "message": "Maximum iterations exceeded",
            "nit": 1000,
            "nfev": 5000,
            "njev": 1200,
        }

        assert result["success"] == False
        assert np.isinf(result["fun"])

    def test_optimization_result_scipy_format(self):
        """Test compatibility with scipy.optimize format."""
        # Simulates scipy.optimize.minimize return
        result: OptimizationResult = {
            "x": np.array([1.0, 1.0]),
            "fun": 0.0,
            "success": True,
            "message": "Optimization terminated successfully.",
            "nit": 25,
            "nfev": 100,
            "njev": 30,
        }

        # Extract solution
        x_opt = result["x"]
        cost = result["fun"]

        assert x_opt.shape == (2,)
        assert cost == 0.0


class TestConstrainedOptimizationResult:
    """Test ConstrainedOptimizationResult TypedDict."""

    def test_constrained_result_creation(self):
        """Test creating ConstrainedOptimizationResult."""
        result: ConstrainedOptimizationResult = {
            "x": np.array([0.5, 0.5]),
            "fun": 0.5,
            "success": True,
            "message": "Optimal solution found",
            "nit": 15,
            "nfev": 60,
            "njev": 18,
            "lagrange_multipliers": {
                "ineq": np.array([0.5]),
                "eq": np.array([0.0]),
            },
            "constraint_violations": np.array([0.0]),
            "kkt_residual": 1e-8,
        }

        assert "lagrange_multipliers" in result
        assert "kkt_residual" in result
        assert result["kkt_residual"] < 1e-6

    def test_lagrange_multipliers(self):
        """Test Lagrange multiplier structure."""
        result: ConstrainedOptimizationResult = {
            "x": np.array([1.0, 0.0]),
            "fun": 1.0,
            "success": True,
            "message": "Success",
            "nit": 10,
            "nfev": 40,
            "njev": 12,
            "lagrange_multipliers": {
                "ineq": np.array([1.5, 0.0, 0.0]),  # 3 inequality constraints
                "eq": np.array([0.3]),  # 1 equality constraint
            },
            "constraint_violations": np.array([0.0, 0.0, 0.0, 0.0]),
            "kkt_residual": 1e-9,
        }

        # Lagrange multipliers for inequality should be >= 0
        lambda_ineq = result["lagrange_multipliers"]["ineq"]
        assert np.all(lambda_ineq >= 0)

    def test_kkt_optimality(self):
        """Test KKT condition checking."""
        # Optimal solution
        result_optimal: ConstrainedOptimizationResult = {
            "x": np.array([0.5, 0.5]),
            "fun": 0.5,
            "success": True,
            "message": "Optimal",
            "nit": 20,
            "nfev": 80,
            "njev": 25,
            "lagrange_multipliers": {"ineq": np.array([1.0])},
            "constraint_violations": np.array([0.0]),
            "kkt_residual": 1e-10,
        }

        # Non-optimal solution
        result_suboptimal: ConstrainedOptimizationResult = {
            "x": np.array([0.3, 0.3]),
            "fun": 0.6,
            "success": False,
            "message": "Tolerance not met",
            "nit": 5,
            "nfev": 20,
            "njev": 8,
            "lagrange_multipliers": {"ineq": np.array([0.5])},
            "constraint_violations": np.array([0.1]),
            "kkt_residual": 1e-2,
        }

        assert result_optimal["kkt_residual"] < result_suboptimal["kkt_residual"]


class TestTrajectoryOptimizationResult:
    """Test TrajectoryOptimizationResult TypedDict."""

    def test_trajectory_result_creation(self):
        """Test creating TrajectoryOptimizationResult."""
        N = 100  # Horizon
        nx, nu = 2, 1

        result: TrajectoryOptimizationResult = {
            "state_trajectory": np.random.randn(N + 1, nx),
            "control_trajectory": np.random.randn(N, nu),
            "cost": 10.5,
            "success": True,
            "message": "Convergence achieved",
            "solve_time": 0.5,
            "iterations": 25,
        }

        assert result["state_trajectory"].shape == (101, 2)
        assert result["control_trajectory"].shape == (100, 1)
        assert result["cost"] > 0

    def test_trajectory_result_with_constraints(self):
        """Test trajectory result with constraint violations."""
        result: TrajectoryOptimizationResult = {
            "state_trajectory": np.random.randn(51, 4),
            "control_trajectory": np.random.randn(50, 2),
            "cost": 25.0,
            "success": True,
            "message": "Optimal",
            "solve_time": 1.2,
            "iterations": 40,
            "constraint_violations": np.random.rand(150) * 1e-6,  # Very small
        }

        assert "constraint_violations" in result
        assert np.max(result["constraint_violations"]) < 1e-5

    def test_trajectory_lengths_consistent(self):
        """Test that trajectory lengths are consistent."""
        N = 50

        result: TrajectoryOptimizationResult = {
            "state_trajectory": np.random.randn(N + 1, 3),
            "control_trajectory": np.random.randn(N, 2),
            "cost": 5.0,
            "success": True,
            "message": "Success",
            "solve_time": 0.3,
            "iterations": 15,
        }

        # State trajectory has N+1 points, control has N
        assert result["state_trajectory"].shape[0] == N + 1
        assert result["control_trajectory"].shape[0] == N


class TestConvexOptimizationResult:
    """Test ConvexOptimizationResult TypedDict."""

    def test_convex_result_creation(self):
        """Test creating ConvexOptimizationResult."""
        result: ConvexOptimizationResult = {
            "x": np.array([0.5, 0.5]),
            "objective_value": 0.5,
            "success": True,
            "solver": "ECOS",
            "solve_time": 0.01,
        }

        assert result["solver"] == "ECOS"
        assert result["success"] == True
        assert result["solve_time"] > 0

    def test_convex_result_with_duals(self):
        """Test convex result with dual variables."""
        result: ConvexOptimizationResult = {
            "x": np.array([1.0, 0.0]),
            "objective_value": 1.0,
            "success": True,
            "solver": "MOSEK",
            "solve_time": 0.05,
            "dual_variables": {
                "linear_ineq": np.array([0.5, 0.0]),
                "linear_eq": np.array([1.0]),
            },
        }

        assert "dual_variables" in result
        assert "linear_ineq" in result["dual_variables"]

    def test_convex_result_different_solvers(self):
        """Test results from different convex solvers."""
        # ECOS (for SOCP)
        result_ecos: ConvexOptimizationResult = {
            "x": np.array([1.0]),
            "objective_value": 1.0,
            "success": True,
            "solver": "ECOS",
            "solve_time": 0.01,
        }

        # SCS (for large-scale)
        result_scs: ConvexOptimizationResult = {
            "x": np.array([1.0]),
            "objective_value": 1.0,
            "success": True,
            "solver": "SCS",
            "solve_time": 0.05,
        }

        # MOSEK (commercial)
        result_mosek: ConvexOptimizationResult = {
            "x": np.array([1.0]),
            "objective_value": 1.0,
            "success": True,
            "solver": "MOSEK",
            "solve_time": 0.005,
        }

        # All should give same solution for convex problem
        assert np.allclose(result_ecos["x"], result_scs["x"])
        assert np.allclose(result_scs["x"], result_mosek["x"])


class TestParameterOptimizationResult:
    """Test ParameterOptimizationResult TypedDict."""

    def test_parameter_result_creation(self):
        """Test creating ParameterOptimizationResult."""
        result: ParameterOptimizationResult = {
            "parameters": np.array([2.5, 1.3, 0.8]),
            "cost": 0.05,
            "success": True,
            "iterations": 50,
            "gradient_norm": 1e-8,
        }

        assert result["parameters"].shape == (3,)
        assert result["gradient_norm"] < 1e-6

    def test_parameter_result_with_hessian(self):
        """Test parameter result with Hessian."""
        ntheta = 3

        result: ParameterOptimizationResult = {
            "parameters": np.array([1.0, 2.0, 0.5]),
            "cost": 0.01,
            "success": True,
            "iterations": 100,
            "gradient_norm": 1e-10,
            "hessian": np.eye(ntheta) * 10.0,  # Well-conditioned
        }

        assert "hessian" in result
        assert result["hessian"].shape == (3, 3)

        # Hessian should be positive definite at minimum
        eigenvalues = np.linalg.eigvals(result["hessian"])
        assert np.all(eigenvalues > 0)

    def test_parameter_uncertainty_from_hessian(self):
        """Test computing parameter uncertainty from Hessian."""
        H = np.diag([100.0, 50.0, 25.0])  # Hessian

        result: ParameterOptimizationResult = {
            "parameters": np.array([1.0, 2.0, 3.0]),
            "cost": 0.001,
            "success": True,
            "iterations": 80,
            "gradient_norm": 1e-9,
            "hessian": H,
        }

        # Parameter covariance = H^{-1}
        param_cov = np.linalg.inv(result["hessian"])
        param_std = np.sqrt(np.diag(param_cov))

        # Higher Hessian eigenvalue → lower uncertainty
        assert param_std[0] < param_std[2]


class TestOptimizationWorkflows:
    """Test realistic optimization workflows."""

    def test_unconstrained_optimization_workflow(self):
        """Test unconstrained optimization workflow."""
        # Simulate optimization
        result: OptimizationResult = {
            "x": np.array([1.0, 1.0]),  # Rosenbrock minimum
            "fun": 0.0,
            "success": True,
            "message": "Optimization terminated successfully",
            "nit": 45,
            "nfev": 180,
            "njev": 50,
        }

        if result["success"]:
            x_opt = result["x"]
            cost = result["fun"]

            assert np.allclose(x_opt, [1.0, 1.0])
            assert cost < 1e-6

    def test_trajectory_optimization_workflow(self):
        """Test trajectory optimization workflow."""
        # Simulate OCP solution
        N = 100
        result: TrajectoryOptimizationResult = {
            "state_trajectory": np.random.randn(N + 1, 2),
            "control_trajectory": np.random.randn(N, 1),
            "cost": 15.0,
            "success": True,
            "message": "Optimal trajectory found",
            "solve_time": 0.8,
            "iterations": 30,
            "constraint_violations": np.random.rand(N) * 1e-7,
        }

        # Extract trajectories
        x_traj = result["state_trajectory"]
        u_traj = result["control_trajectory"]

        # Check dimensions
        assert x_traj.shape == (101, 2)
        assert u_traj.shape == (100, 1)

        # Verify constraints satisfied
        max_viol = np.max(result["constraint_violations"])
        assert max_viol < 1e-5

    def test_parameter_fitting_workflow(self):
        """Test parameter fitting workflow."""
        # Simulate parameter estimation
        result: ParameterOptimizationResult = {
            "parameters": np.array([2.0, 1.0, 0.5]),
            "cost": 0.002,
            "success": True,
            "iterations": 150,
            "gradient_norm": 5e-9,
            "hessian": np.diag([80.0, 60.0, 40.0]),
        }

        # Extract parameters
        theta_opt = result["parameters"]

        # Check convergence
        assert result["gradient_norm"] < 1e-6

        # Estimate uncertainty
        H = result["hessian"]
        param_cov = np.linalg.inv(H)
        param_std = np.sqrt(np.diag(param_cov))

        # Report with uncertainty
        for i, (theta, std) in enumerate(zip(theta_opt, param_std)):
            assert std > 0  # Uncertainty should be positive


class TestNumericalProperties:
    """Test numerical properties of results."""

    def test_bounds_consistency(self):
        """Test that bounds are consistent."""
        bounds: OptimizationBounds = {
            "lower": np.array([0.0, -1.0]),
            "upper": np.array([1.0, 1.0]),
        }

        # Check element-wise
        for i in range(len(bounds["lower"])):
            assert bounds["lower"][i] <= bounds["upper"][i]

    def test_optimization_cost_non_negative(self):
        """Test that optimization cost is reasonable."""
        result: OptimizationResult = {
            "x": np.array([1.0]),
            "fun": 5.0,
            "success": True,
            "message": "Success",
            "nit": 10,
            "nfev": 40,
            "njev": 12,
        }

        # Cost can be negative (e.g., maximize profit)
        # But should be finite
        assert np.isfinite(result["fun"])

    def test_gradient_norm_at_optimum(self):
        """Test gradient norm at optimum."""
        result: ParameterOptimizationResult = {
            "parameters": np.array([1.0, 2.0]),
            "cost": 0.0,
            "success": True,
            "iterations": 50,
            "gradient_norm": 1e-12,
        }

        # At local minimum, gradient should be ≈0
        assert result["gradient_norm"] < 1e-6


class TestDocumentationExamples:
    """Test that documentation examples work."""

    def test_optimization_result_example(self):
        """Test OptimizationResult example from docstring."""
        result: OptimizationResult = {
            "x": np.array([1.0, 1.0]),
            "fun": 0.0,
            "success": True,
            "message": "Optimization terminated successfully",
            "nit": 25,
            "nfev": 100,
            "njev": 30,
        }

        if result["success"]:
            x_opt = result["x"]
            assert x_opt.shape == (2,)

    def test_trajectory_result_example(self):
        """Test TrajectoryOptimizationResult example structure."""
        result: TrajectoryOptimizationResult = {
            "state_trajectory": np.random.randn(101, 2),
            "control_trajectory": np.random.randn(100, 1),
            "cost": 5.0,
            "success": True,
            "message": "Success",
            "solve_time": 0.5,
            "iterations": 20,
        }

        x_traj = result["state_trajectory"]
        u_traj = result["control_trajectory"]

        assert x_traj.shape == (101, 2)
        assert u_traj.shape == (100, 1)


class TestFieldPresence:
    """Test that all fields are accessible."""

    def test_optimization_result_has_required_fields(self):
        """Test OptimizationResult has core fields."""
        result: OptimizationResult = {
            "x": np.zeros(2),
            "fun": 0.0,
            "success": True,
        }

        assert "x" in result
        assert "fun" in result
        assert "success" in result

    def test_trajectory_result_has_required_fields(self):
        """Test TrajectoryOptimizationResult has core fields."""
        result: TrajectoryOptimizationResult = {
            "state_trajectory": np.zeros((10, 2)),
            "control_trajectory": np.zeros((9, 1)),
            "cost": 1.0,
            "success": True,
        }

        assert "state_trajectory" in result
        assert "control_trajectory" in result
        assert "cost" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
