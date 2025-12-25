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
Model Order Reduction Types

Result types for model reduction and balanced realization:
- Balanced realization
- Model order reduction
- Hankel singular values
- Error bounds

Model reduction creates lower-order approximations of high-order systems
while preserving essential dynamics.

Mathematical Background
----------------------
Balanced Realization:
    Transform system to coordinates where controllability and
    observability gramians are equal and diagonal:
        
        Wc = Wo = Σ = diag(σ₁, σ₂, ..., σₙ)
    
    Where σᵢ are Hankel singular values (HSVs)
    
    Transformation: x_bal = T x
        A_bal = T A T⁻¹
        B_bal = T B
        C_bal = C T⁻¹

Gramians:
    Controllability gramian Wc:
        A Wc + Wc A' + B B' = 0
    
    Observability gramian Wo:
        A' Wo + Wo A + C' C = 0
    
    Hankel singular values:
        σᵢ = √λᵢ(Wc Wo)

Model Order Reduction:
    Truncate balanced realization by keeping r largest HSVs:
    
    1. Partition: x_bal = [x₁; x₂] where dim(x₁) = r
    2. Keep only x₁ states
    3. Reduced system:
        A_r = A₁₁
        B_r = B₁
        C_r = C₁
    
    Error bound (H∞):
        ||G - G_r||_∞ ≤ 2(σᵣ₊₁ + σᵣ₊₂ + ... + σₙ)

Methods:
    1. Balanced truncation: Truncate balanced realization
    2. Modal reduction: Keep dominant eigenvalues
    3. Hankel norm: Optimal Hankel norm approximation
    4. Proper Orthogonal Decomposition (POD): Data-driven
    5. Moment matching (Krylov): Match transfer function moments

Usage
-----
>>> from src.types.model_reduction import (
...     BalancedRealizationResult,
...     ReducedOrderModelResult,
... )
>>> 
>>> # Balanced realization
>>> bal_result: BalancedRealizationResult = balanced_realization(
...     A, B, C
... )
>>> hsv = bal_result['hankel_singular_values']
>>> 
>>> # Choose reduced order
>>> r = np.sum(hsv > 1e-3)  # Keep HSVs > threshold
>>> 
>>> # Reduce model
>>> rom_result: ReducedOrderModelResult = reduce_model(
...     system, target_order=r, method='balanced'
... )
>>> A_r = rom_result['A_reduced']
>>> error_bound = rom_result['approximation_error']
"""

from typing_extensions import TypedDict
import numpy as np

from .core import (
    StateMatrix,
    InputMatrix,
    OutputMatrix,
    FeedthroughMatrix,
    CovarianceMatrix,
    ArrayLike,
)


# ============================================================================
# Balanced Realization
# ============================================================================

class BalancedRealizationResult(TypedDict):
    """
    Balanced realization result.
    
    Transforms system to balanced coordinates where controllability
    and observability gramians are equal and diagonal.
    
    Fields
    ------
    A_balanced : StateMatrix
        Balanced state matrix A_bal (nx, nx)
    B_balanced : InputMatrix
        Balanced input matrix B_bal (nx, nu)
    C_balanced : OutputMatrix
        Balanced output matrix C_bal (ny, nx)
    transformation : ArrayLike
        Balancing transformation T (nx, nx)
    hankel_singular_values : ArrayLike
        HSVs σ₁ ≥ σ₂ ≥ ... ≥ σₙ ≥ 0
    controllability_gramian : CovarianceMatrix
        Controllability gramian Wc (nx, nx)
    observability_gramian : CovarianceMatrix
        Observability gramian Wo (nx, nx)
    
    Examples
    --------
    >>> # Balance system
    >>> result: BalancedRealizationResult = balanced_realization(
    ...     A=A, B=B, C=C
    ... )
    >>> 
    >>> # Extract balanced system
    >>> A_bal = result['A_balanced']
    >>> B_bal = result['B_balanced']
    >>> C_bal = result['C_balanced']
    >>> 
    >>> # Hankel singular values
    >>> hsv = result['hankel_singular_values']
    >>> print(f"HSVs: {hsv}")
    >>> 
    >>> # Plot HSVs to choose reduced order
    >>> import matplotlib.pyplot as plt
    >>> plt.semilogy(hsv, 'o-')
    >>> plt.axhline(1e-3, color='r', linestyle='--', label='Threshold')
    >>> plt.xlabel('State index')
    >>> plt.ylabel('Hankel singular value')
    >>> plt.grid(True)
    >>> 
    >>> # Choose reduced order (e.g., keep HSVs > 1e-3)
    >>> r = np.sum(hsv > 1e-3)
    >>> print(f"Reduced order: {r}/{len(hsv)}")
    >>> 
    >>> # Verify gramians are balanced
    >>> Wc = result['controllability_gramian']
    >>> Wo = result['observability_gramian']
    >>> assert np.allclose(Wc, Wo)
    >>> assert np.allclose(Wc, np.diag(hsv))
    """
    A_balanced: StateMatrix
    B_balanced: InputMatrix
    C_balanced: OutputMatrix
    transformation: ArrayLike
    hankel_singular_values: ArrayLike
    controllability_gramian: CovarianceMatrix
    observability_gramian: CovarianceMatrix


# ============================================================================
# Model Order Reduction
# ============================================================================

class ReducedOrderModelResult(TypedDict):
    """
    Model order reduction result.
    
    Lower-order approximation of original high-order system with
    guaranteed error bounds.
    
    Fields
    ------
    A_reduced : StateMatrix
        Reduced state matrix (n_r, n_r)
    B_reduced : InputMatrix
        Reduced input matrix (n_r, nu)
    C_reduced : OutputMatrix
        Reduced output matrix (ny, n_r)
    D_reduced : FeedthroughMatrix
        Reduced feedthrough (ny, nu)
    original_order : int
        Original state dimension nx
    reduced_order : int
        Reduced state dimension n_r
    approximation_error : float
        Error bound ||G - G_r||
    method : str
        Reduction method ('balanced', 'modal', 'hankel', 'POD')
    preserved_modes : np.ndarray
        Eigenvalues preserved in reduction
    
    Examples
    --------
    >>> # Reduce 100-state model to 10 states
    >>> result: ReducedOrderModelResult = reduce_model(
    ...     system=(A, B, C, D),
    ...     target_order=10,
    ...     method='balanced'
    ... )
    >>> 
    >>> # Extract reduced system
    >>> A_r = result['A_reduced']
    >>> B_r = result['B_reduced']
    >>> C_r = result['C_reduced']
    >>> D_r = result['D_reduced']
    >>> 
    >>> print(f"Original order: {result['original_order']}")
    >>> print(f"Reduced order: {result['reduced_order']}")
    >>> print(f"Approximation error: {result['approximation_error']:.2e}")
    >>> 
    >>> # Compare transfer functions
    >>> from scipy import signal
    >>> sys_orig = signal.StateSpace(A, B, C, D)
    >>> sys_red = signal.StateSpace(A_r, B_r, C_r, D_r)
    >>> 
    >>> omega = np.logspace(-2, 2, 100)
    >>> _, G_orig = signal.freqresp((A, B, C, D), omega)
    >>> _, G_red = signal.freqresp((A_r, B_r, C_r, D_r), omega)
    >>> 
    >>> import matplotlib.pyplot as plt
    >>> plt.loglog(omega, np.abs(G_orig[0, 0]), label='Original')
    >>> plt.loglog(omega, np.abs(G_red[0, 0]), '--', label='Reduced')
    >>> plt.xlabel('Frequency (rad/s)')
    >>> plt.ylabel('Magnitude')
    >>> plt.legend()
    >>> 
    >>> # Error
    >>> error = np.abs(G_orig - G_red)
    >>> print(f"Max frequency error: {np.max(error):.2e}")
    >>> print(f"Error bound: {result['approximation_error']:.2e}")
    """
    A_reduced: StateMatrix
    B_reduced: InputMatrix
    C_reduced: OutputMatrix
    D_reduced: FeedthroughMatrix
    original_order: int
    reduced_order: int
    approximation_error: float
    method: str
    preserved_modes: np.ndarray


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    'BalancedRealizationResult',
    'ReducedOrderModelResult',
]