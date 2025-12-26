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
Unit Tests for Model Reduction Types

Tests TypedDict definitions and usage patterns for model order reduction
and balanced realization types.
"""

import pytest
import numpy as np
from src.types.model_reduction import (
    BalancedRealizationResult,
    ReducedOrderModelResult,
)


class TestBalancedRealizationResult:
    """Test BalancedRealizationResult TypedDict."""
    
    def test_balanced_realization_result_creation(self):
        """Test creating balanced realization result."""
        nx, nu, ny = 5, 2, 3
        
        result: BalancedRealizationResult = {
            'A_balanced': np.random.randn(nx, nx),
            'B_balanced': np.random.randn(nx, nu),
            'C_balanced': np.random.randn(ny, nx),
            'transformation': np.eye(nx),
            'hankel_singular_values': np.array([10, 5, 2, 0.5, 0.1]),
            'controllability_gramian': np.diag([10, 5, 2, 0.5, 0.1]),
            'observability_gramian': np.diag([10, 5, 2, 0.5, 0.1]),
        }
        
        assert result['A_balanced'].shape == (nx, nx)
        assert result['B_balanced'].shape == (nx, nu)
        assert result['C_balanced'].shape == (ny, nx)
        assert len(result['hankel_singular_values']) == nx
    
    def test_balanced_gramians_equal_diagonal(self):
        """Test balanced gramians are equal and diagonal."""
        nx = 4
        hsv = np.array([8.0, 4.0, 1.0, 0.2])
        
        result: BalancedRealizationResult = {
            'A_balanced': np.random.randn(nx, nx),
            'B_balanced': np.random.randn(nx, 2),
            'C_balanced': np.random.randn(2, nx),
            'transformation': np.eye(nx),
            'hankel_singular_values': hsv,
            'controllability_gramian': np.diag(hsv),
            'observability_gramian': np.diag(hsv),
        }
        
        Wc = result['controllability_gramian']
        Wo = result['observability_gramian']
        
        # Gramians should be equal
        assert np.allclose(Wc, Wo)
        
        # Should be diagonal
        assert np.allclose(Wc, np.diag(np.diag(Wc)))
        
        # Diagonal elements are HSVs
        assert np.allclose(np.diag(Wc), hsv)
    
    def test_hankel_singular_values_decreasing(self):
        """Test HSVs are in decreasing order."""
        hsv = np.array([10.0, 5.0, 2.0, 0.8, 0.1])
        
        result: BalancedRealizationResult = {
            'A_balanced': np.random.randn(5, 5),
            'B_balanced': np.random.randn(5, 1),
            'C_balanced': np.random.randn(1, 5),
            'transformation': np.eye(5),
            'hankel_singular_values': hsv,
            'controllability_gramian': np.diag(hsv),
            'observability_gramian': np.diag(hsv),
        }
        
        # HSVs should be in decreasing order
        assert np.all(np.diff(result['hankel_singular_values']) <= 0)
    
    def test_hankel_singular_values_positive(self):
        """Test HSVs are non-negative."""
        hsv = np.array([5.0, 3.0, 1.0, 0.5, 0.0])
        
        result: BalancedRealizationResult = {
            'A_balanced': np.random.randn(5, 5),
            'B_balanced': np.random.randn(5, 2),
            'C_balanced': np.random.randn(2, 5),
            'transformation': np.eye(5),
            'hankel_singular_values': hsv,
            'controllability_gramian': np.diag(hsv),
            'observability_gramian': np.diag(hsv),
        }
        
        assert np.all(result['hankel_singular_values'] >= 0)


class TestReducedOrderModelResult:
    """Test ReducedOrderModelResult TypedDict."""
    
    def test_reduced_order_model_result_creation(self):
        """Test creating reduced order model result."""
        nx_orig = 20
        nx_red = 5
        nu, ny = 2, 3
        
        result: ReducedOrderModelResult = {
            'A_reduced': np.random.randn(nx_red, nx_red),
            'B_reduced': np.random.randn(nx_red, nu),
            'C_reduced': np.random.randn(ny, nx_red),
            'D_reduced': np.zeros((ny, nu)),
            'original_order': nx_orig,
            'reduced_order': nx_red,
            'approximation_error': 0.05,
            'method': 'balanced',
            'preserved_modes': np.random.randn(nx_red) + 1j*np.random.randn(nx_red),
        }
        
        assert result['A_reduced'].shape == (nx_red, nx_red)
        assert result['original_order'] == nx_orig
        assert result['reduced_order'] == nx_red
        assert result['approximation_error'] >= 0
    
    def test_reduced_order_smaller_than_original(self):
        """Test reduced order is smaller than original."""
        result: ReducedOrderModelResult = {
            'A_reduced': np.random.randn(5, 5),
            'B_reduced': np.random.randn(5, 1),
            'C_reduced': np.random.randn(1, 5),
            'D_reduced': np.zeros((1, 1)),
            'original_order': 20,
            'reduced_order': 5,
            'approximation_error': 0.02,
            'method': 'balanced',
            'preserved_modes': np.array([-1, -2, -3, -4, -5]),
        }
        
        assert result['reduced_order'] < result['original_order']
    
    def test_reduction_methods(self):
        """Test different reduction methods."""
        methods = ['balanced', 'modal', 'hankel', 'POD']
        
        for method in methods:
            result: ReducedOrderModelResult = {
                'A_reduced': np.random.randn(3, 3),
                'B_reduced': np.random.randn(3, 1),
                'C_reduced': np.random.randn(1, 3),
                'D_reduced': np.zeros((1, 1)),
                'original_order': 10,
                'reduced_order': 3,
                'approximation_error': 0.01,
                'method': method,
                'preserved_modes': np.array([-1, -2, -3]),
            }
            assert result['method'] == method
    
    def test_approximation_error_bound(self):
        """Test approximation error is non-negative."""
        result: ReducedOrderModelResult = {
            'A_reduced': np.random.randn(4, 4),
            'B_reduced': np.random.randn(4, 2),
            'C_reduced': np.random.randn(2, 4),
            'D_reduced': np.zeros((2, 2)),
            'original_order': 15,
            'reduced_order': 4,
            'approximation_error': 0.08,
            'method': 'balanced',
            'preserved_modes': np.array([-1, -2, -3, -4]),
        }
        
        assert result['approximation_error'] >= 0


class TestPracticalUseCases:
    """Test realistic usage patterns."""
    
    def test_balanced_truncation_workflow(self):
        """Test balanced truncation workflow."""
        # Original system
        nx_orig = 10
        nu, ny = 2, 2
        
        # Balanced realization
        hsv = np.array([10, 8, 5, 3, 1, 0.5, 0.2, 0.1, 0.05, 0.01])
        
        bal_result: BalancedRealizationResult = {
            'A_balanced': np.random.randn(nx_orig, nx_orig),
            'B_balanced': np.random.randn(nx_orig, nu),
            'C_balanced': np.random.randn(ny, nx_orig),
            'transformation': np.eye(nx_orig),
            'hankel_singular_values': hsv,
            'controllability_gramian': np.diag(hsv),
            'observability_gramian': np.diag(hsv),
        }
        
        # Choose reduced order (keep HSVs > 0.5)
        threshold = 0.5
        r = np.sum(bal_result['hankel_singular_values'] > threshold)
        assert r == 5
        
        # Truncate
        A_bal = bal_result['A_balanced']
        B_bal = bal_result['B_balanced']
        C_bal = bal_result['C_balanced']
        
        A_r = A_bal[:r, :r]
        B_r = B_bal[:r, :]
        C_r = C_bal[:, :r]
        
        # Error bound: 2 * sum of discarded HSVs
        discarded_hsv = bal_result['hankel_singular_values'][r:]
        error_bound = 2 * np.sum(discarded_hsv)
        
        rom_result: ReducedOrderModelResult = {
            'A_reduced': A_r,
            'B_reduced': B_r,
            'C_reduced': C_r,
            'D_reduced': np.zeros((ny, nu)),
            'original_order': nx_orig,
            'reduced_order': r,
            'approximation_error': error_bound,
            'method': 'balanced',
            'preserved_modes': np.linalg.eigvals(A_r),
        }
        
        assert rom_result['reduced_order'] == 5
        assert rom_result['approximation_error'] > 0
    
    def test_modal_reduction(self):
        """Test modal reduction workflow."""
        # Original system with known eigenvalues
        eigenvalues_orig = np.array([-10, -8, -5, -3, -1, -0.5, -0.2, -0.1])
        nx_orig = len(eigenvalues_orig)
        
        # Keep fastest 4 modes
        r = 4
        preserved_modes = eigenvalues_orig[:r]
        
        result: ReducedOrderModelResult = {
            'A_reduced': np.diag(preserved_modes),
            'B_reduced': np.random.randn(r, 1),
            'C_reduced': np.random.randn(1, r),
            'D_reduced': np.zeros((1, 1)),
            'original_order': nx_orig,
            'reduced_order': r,
            'approximation_error': 0.15,
            'method': 'modal',
            'preserved_modes': preserved_modes,
        }
        
        # Verify preserved modes are fastest
        assert np.all(np.real(result['preserved_modes']) <= -1.0)
    
    def test_reduction_order_selection(self):
        """Test choosing reduced order based on HSVs."""
        hsv = np.array([100, 50, 20, 10, 5, 2, 0.8, 0.3, 0.1, 0.05])
        
        # Different thresholds
        # threshold=1.0: keep [100, 50, 20, 10, 5, 2] = 6 states
        # threshold=5.0: keep [100, 50, 20, 10] = 4 states (5 is NOT > 5.0)
        # threshold=10.0: keep [100, 50, 20] = 3 states (10 is NOT > 10.0)
        thresholds = [1.0, 5.0, 10.0]
        expected_orders = [6, 4, 3]
        
        for threshold, expected_r in zip(thresholds, expected_orders):
            r = np.sum(hsv > threshold)
            assert r == expected_r


class TestNumericalProperties:
    """Test numerical properties of results."""
    
    def test_gramians_positive_semidefinite(self):
        """Test gramians are positive semidefinite."""
        hsv = np.array([5.0, 3.0, 1.0, 0.5])
        Wc = np.diag(hsv)
        Wo = np.diag(hsv)
        
        result: BalancedRealizationResult = {
            'A_balanced': np.random.randn(4, 4),
            'B_balanced': np.random.randn(4, 2),
            'C_balanced': np.random.randn(2, 4),
            'transformation': np.eye(4),
            'hankel_singular_values': hsv,
            'controllability_gramian': Wc,
            'observability_gramian': Wo,
        }
        
        # Eigenvalues should be non-negative
        eig_Wc = np.linalg.eigvals(result['controllability_gramian'])
        eig_Wo = np.linalg.eigvals(result['observability_gramian'])
        
        assert np.all(eig_Wc >= -1e-10)  # Numerical tolerance
        assert np.all(eig_Wo >= -1e-10)
    
    def test_error_bound_positive(self):
        """Test approximation error bound is positive."""
        result: ReducedOrderModelResult = {
            'A_reduced': np.random.randn(3, 3),
            'B_reduced': np.random.randn(3, 1),
            'C_reduced': np.random.randn(1, 3),
            'D_reduced': np.zeros((1, 1)),
            'original_order': 10,
            'reduced_order': 3,
            'approximation_error': 0.05,
            'method': 'balanced',
            'preserved_modes': np.array([-1, -2, -3]),
        }
        
        assert result['approximation_error'] >= 0
    
    def test_transformation_invertible(self):
        """Test balancing transformation is invertible."""
        T = np.random.randn(5, 5)
        # Make it invertible
        T = T + 5 * np.eye(5)
        
        result: BalancedRealizationResult = {
            'A_balanced': np.random.randn(5, 5),
            'B_balanced': np.random.randn(5, 2),
            'C_balanced': np.random.randn(2, 5),
            'transformation': T,
            'hankel_singular_values': np.array([5, 3, 1, 0.5, 0.1]),
            'controllability_gramian': np.diag([5, 3, 1, 0.5, 0.1]),
            'observability_gramian': np.diag([5, 3, 1, 0.5, 0.1]),
        }
        
        # Should be invertible (det != 0)
        det = np.linalg.det(result['transformation'])
        assert abs(det) > 1e-10


class TestErrorBounds:
    """Test error bound calculations."""
    
    def test_balanced_truncation_error_bound(self):
        """Test balanced truncation error bound formula."""
        # HSVs
        hsv = np.array([10, 8, 5, 2, 1, 0.5, 0.2, 0.1])
        
        # Keep r=5 states
        r = 5
        discarded_hsv = hsv[r:]
        
        # Error bound: ||G - G_r||_∞ ≤ 2 * sum(discarded HSVs)
        error_bound = 2 * np.sum(discarded_hsv)
        
        result: ReducedOrderModelResult = {
            'A_reduced': np.random.randn(r, r),
            'B_reduced': np.random.randn(r, 1),
            'C_reduced': np.random.randn(1, r),
            'D_reduced': np.zeros((1, 1)),
            'original_order': len(hsv),
            'reduced_order': r,
            'approximation_error': error_bound,
            'method': 'balanced',
            'preserved_modes': np.random.randn(r),
        }
        
        # Verify error bound
        expected_error = 2 * (0.5 + 0.2 + 0.1)
        assert np.isclose(result['approximation_error'], expected_error)
    
    def test_smaller_reduction_smaller_error(self):
        """Test that smaller reduction gives smaller error."""
        hsv = np.array([10, 5, 2, 1, 0.5, 0.2, 0.1])
        
        # Keep r=5 states
        error_r5 = 2 * np.sum(hsv[5:])
        
        # Keep r=4 states (more reduction)
        error_r4 = 2 * np.sum(hsv[4:])
        
        # More reduction → larger error
        assert error_r4 > error_r5


class TestDocumentationExamples:
    """Test that documentation examples work."""
    
    def test_balanced_realization_example(self):
        """Test BalancedRealizationResult example from docstring."""
        nx = 5
        hsv = np.array([10, 5, 2, 0.8, 0.2])
        
        result: BalancedRealizationResult = {
            'A_balanced': np.random.randn(nx, nx),
            'B_balanced': np.random.randn(nx, 2),
            'C_balanced': np.random.randn(2, nx),
            'transformation': np.eye(nx),
            'hankel_singular_values': hsv,
            'controllability_gramian': np.diag(hsv),
            'observability_gramian': np.diag(hsv),
        }
        
        # Choose reduced order (all HSVs > 1e-3)
        r = np.sum(result['hankel_singular_values'] > 1e-3)
        assert r == 5  # All: 10, 5, 2, 0.8, 0.2 are all > 0.001
    
    def test_reduced_order_model_example(self):
        """Test ReducedOrderModelResult example structure."""
        result: ReducedOrderModelResult = {
            'A_reduced': np.random.randn(10, 10),
            'B_reduced': np.random.randn(10, 2),
            'C_reduced': np.random.randn(2, 10),
            'D_reduced': np.zeros((2, 2)),
            'original_order': 100,
            'reduced_order': 10,
            'approximation_error': 0.05,
            'method': 'balanced',
            'preserved_modes': np.random.randn(10),
        }
        
        assert result['original_order'] == 100
        assert result['reduced_order'] == 10


class TestFieldPresence:
    """Test that all fields are accessible."""
    
    def test_balanced_realization_has_required_fields(self):
        """Test BalancedRealizationResult has core fields."""
        result: BalancedRealizationResult = {
            'A_balanced': np.random.randn(3, 3),
            'B_balanced': np.random.randn(3, 1),
            'C_balanced': np.random.randn(1, 3),
            'transformation': np.eye(3),
            'hankel_singular_values': np.array([5, 2, 0.5]),
            'controllability_gramian': np.diag([5, 2, 0.5]),
            'observability_gramian': np.diag([5, 2, 0.5]),
        }
        
        assert 'A_balanced' in result
        assert 'hankel_singular_values' in result
        assert 'transformation' in result
    
    def test_reduced_order_model_has_required_fields(self):
        """Test ReducedOrderModelResult has core fields."""
        result: ReducedOrderModelResult = {
            'A_reduced': np.random.randn(5, 5),
            'B_reduced': np.random.randn(5, 2),
            'C_reduced': np.random.randn(2, 5),
            'D_reduced': np.zeros((2, 2)),
            'original_order': 20,
            'reduced_order': 5,
            'approximation_error': 0.03,
            'method': 'balanced',
            'preserved_modes': np.random.randn(5),
        }
        
        assert 'A_reduced' in result
        assert 'original_order' in result
        assert 'approximation_error' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])