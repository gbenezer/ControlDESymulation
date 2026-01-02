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
Classical Control Theory Functions

Pure stateless functions for classical control design and analysis:

**Control Design:**
- Linear Quadratic Regulator (LQR) - continuous and discrete
- Kalman Filter - optimal state estimation
- Linear Quadratic Gaussian (LQG) - combined LQR + Kalman

**System Analysis:**
- Stability analysis - eigenvalue-based
- Controllability - rank test
- Observability - rank test

All functions are pure (no side effects, no state) and work like scipy.
Backend conversion is handled internally.

Mathematical Background
-----------------------
LQR minimizes:
    J = ∫₀^∞ (x'Qx + u'Ru) dt  (continuous)
    J = Σₖ₌₀^∞ (x'Qx + u'Ru)     (discrete)

Solution via algebraic Riccati equation (ARE):
    Continuous: A'P + PA - PBR⁻¹B'P + Q = 0
    Discrete:   P = A'PA - A'PB(R + B'PB)⁻¹B'PA + Q

Optimal gain: K = R⁻¹B'P (continuous), K = (R + B'PB)⁻¹B'PA (discrete)

Kalman Filter for:
    x[k+1] = Ax[k] + Bu[k] + w[k],  w ~ N(0,Q)
    y[k] = Cx[k] + v[k],            v ~ N(0,R)

Estimator: x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])

Stability:
    Continuous: All Re(λ) < 0 (left half-plane)
    Discrete:   All |λ| < 1 (inside unit circle)

Controllability: rank([B AB A²B ... Aⁿ⁻¹B]) = n
Observability:   rank([C; CA; CA²; ...; CAⁿ⁻¹]) = n

Usage
-----
>>> from src.control.classical import design_lqr_continuous, analyze_stability
>>> import numpy as np
>>>
>>> # LQR design
>>> A = np.array([[0, 1], [-2, -3]])
>>> B = np.array([[0], [1]])
>>> Q = np.diag([10, 1])
>>> R = np.array([[0.1]])
>>>
>>> result = design_lqr_continuous(A, B, Q, R)
>>> K = result['gain']
>>> print(f"Optimal gain: {K}")
>>> print(f"Stability margin: {result['stability_margin']:.3f}")
>>>
>>> # Stability analysis
>>> stability = analyze_stability(A, system_type='continuous')
>>> print(f"Stable: {stability['is_stable']}")
"""

from typing import Optional

import numpy as np
from scipy import linalg

from src.types.backends import Backend
from src.types.control_classical import (
    ControllabilityInfo,
    KalmanFilterResult,
    LQGResult,
    LQRResult,
    ObservabilityInfo,
    StabilityInfo,
)
from src.types.core import (
    InputMatrix,
    OutputMatrix,
    StateMatrix,
)


# ============================================================================
# Backend Conversion Utilities (Internal)
# ============================================================================


def _to_numpy(arr, backend: Backend):
    """
    Convert array to NumPy for scipy operations.

    Args:
        arr: Array in any backend
        backend: Source backend identifier

    Returns:
        NumPy array
    """
    if isinstance(arr, np.ndarray):
        return arr

    if backend == "torch" or hasattr(arr, "cpu"):
        # PyTorch tensor
        return arr.detach().cpu().numpy()
    elif backend == "jax" or hasattr(arr, "__array__"):
        # JAX array
        return np.array(arr)
    else:
        # Try generic conversion
        return np.asarray(arr)


def _from_numpy(arr: np.ndarray, backend: Backend):
    """
    Convert NumPy array back to target backend.

    Args:
        arr: NumPy array
        backend: Target backend

    Returns:
        Array in target backend
    """
    if backend == "numpy":
        return arr
    elif backend == "torch":
        import torch

        return torch.from_numpy(arr)
    elif backend == "jax":
        import jax.numpy as jnp

        return jnp.array(arr)
    else:
        return arr


# ============================================================================
# LQR - Linear Quadratic Regulator
# ============================================================================


def design_lqr_continuous(
    A: StateMatrix,
    B: InputMatrix,
    Q: StateMatrix,
    R: InputMatrix,
    N: Optional[InputMatrix] = None,
    backend: Backend = "numpy",
) -> LQRResult:
    """
    Design continuous-time Linear Quadratic Regulator (LQR).

    Minimizes cost functional:
        J = ∫₀^∞ (x'Qx + u'Ru + 2x'Nu) dt

    Solves continuous-time algebraic Riccati equation (CARE):
        A'P + PA - (PB + N)R⁻¹(B'P + N') + Q = 0

    Optimal control law: u = -Kx where K = R⁻¹(B'P + N')

    Args:
        A: State matrix (nx, nx)
        B: Input matrix (nx, nu)
        Q: State cost matrix (nx, nx), must be positive semi-definite (Q ≥ 0)
        R: Control cost matrix (nu, nu), must be positive definite (R > 0)
        N: Cross-coupling matrix (nx, nu), optional. Default is zero.
        backend: Computational backend ('numpy', 'torch', 'jax')

    Returns:
        LQRResult containing:
            - gain: Optimal feedback gain K (nu, nx)
            - cost_to_go: Riccati solution P (nx, nx)
            - closed_loop_eigenvalues: Eigenvalues of (A - BK)
            - stability_margin: -max(Re(λ)) for continuous systems (positive = stable)

    Raises:
        ValueError: If matrices have incompatible shapes
        LinAlgError: If Riccati equation has no solution (system may be unstable or uncontrollable)

    Examples
    --------
    >>> # Simple double integrator
    >>> A = np.array([[0, 1], [0, 0]])
    >>> B = np.array([[0], [1]])
    >>> Q = np.diag([10, 1])  # Penalize position more
    >>> R = np.array([[0.1]])  # Small control cost
    >>>
    >>> result = design_lqr_continuous(A, B, Q, R)
    >>> K = result['gain']
    >>> print(f"Gain: {K}")  # Approximately [[10, 4.47]]
    >>> print(f"Stable: {result['stability_margin'] > 0}")  # True
    >>>
    >>> # Apply control
    >>> x = np.array([1.0, 0.0])
    >>> u = -K @ x
    >>>
    >>> # Pendulum linearized at upright
    >>> A = np.array([[0, 1], [g/l, 0]])
    >>> B = np.array([[0], [1/(m*l**2)]])
    >>> Q = np.diag([100, 10])  # Heavy penalty on angle
    >>> R = np.array([[1]])
    >>> result = design_lqr_continuous(A, B, Q, R)

    Notes
    -----
    - For controllability, (A,B) must be controllable
    - For stabilizability, unstable modes must be controllable
    - Q must be positive semi-definite (all eigenvalues ≥ 0)
    - R must be positive definite (all eigenvalues > 0)
    - Cross-term N allows for non-quadratic objectives
    """
    # Convert to NumPy for scipy
    A_np = _to_numpy(A, backend)
    B_np = _to_numpy(B, backend)
    Q_np = _to_numpy(Q, backend)
    R_np = _to_numpy(R, backend)

    # Validate shapes
    nx = A_np.shape[0]
    nu = B_np.shape[1]

    if A_np.shape != (nx, nx):
        raise ValueError(f"A must be square, got shape {A_np.shape}")
    if B_np.shape[0] != nx:
        raise ValueError(f"B must have {nx} rows, got {B_np.shape[0]}")
    if Q_np.shape != (nx, nx):
        raise ValueError(f"Q must be ({nx}, {nx}), got {Q_np.shape}")
    if R_np.shape != (nu, nu):
        raise ValueError(f"R must be ({nu}, {nu}), got {R_np.shape}")

    # Solve continuous-time ARE
    if N is not None:
        N_np = _to_numpy(N, backend)
        if N_np.shape != (nx, nu):
            raise ValueError(f"N must be ({nx}, {nu}), got {N_np.shape}")
        P = linalg.solve_continuous_are(A_np, B_np, Q_np, R_np, s=N_np)
        # K = R^{-1}(B'P + N')
        K = linalg.solve(R_np, B_np.T @ P + N_np.T)
    else:
        P = linalg.solve_continuous_are(A_np, B_np, Q_np, R_np)
        # K = R^{-1}B'P
        K = linalg.solve(R_np, B_np.T @ P)

    # Closed-loop system: A_cl = A - BK
    A_cl = A_np - B_np @ K
    eigenvalues = np.linalg.eigvals(A_cl)

    # Stability margin for continuous: -max(Re(λ))
    # Positive margin = stable (all Re(λ) < 0)
    stability_margin = -np.max(np.real(eigenvalues))

    # Convert back to target backend
    result: LQRResult = {
        "gain": _from_numpy(K, backend),
        "cost_to_go": _from_numpy(P, backend),
        "closed_loop_eigenvalues": _from_numpy(eigenvalues, backend),
        "stability_margin": float(stability_margin),
    }

    return result


def design_lqr_discrete(
    A: StateMatrix,
    B: InputMatrix,
    Q: StateMatrix,
    R: InputMatrix,
    N: Optional[InputMatrix] = None,
    backend: Backend = "numpy",
) -> LQRResult:
    """
    Design discrete-time Linear Quadratic Regulator (LQR).

    Minimizes cost functional:
        J = Σₖ₌₀^∞ (x[k]'Qx[k] + u[k]'Ru[k] + 2x[k]'Nu[k])

    Solves discrete-time algebraic Riccati equation (DARE):
        P = A'PA - (A'PB + N)(R + B'PB)⁻¹(B'PA + N') + Q

    Optimal control law: u[k] = -Kx[k] where K = (R + B'PB)⁻¹(B'PA + N')

    Args:
        A: State matrix (nx, nx)
        B: Input matrix (nx, nu)
        Q: State cost matrix (nx, nx), must be positive semi-definite (Q ≥ 0)
        R: Control cost matrix (nu, nu), must be positive definite (R > 0)
        N: Cross-coupling matrix (nx, nu), optional. Default is zero.
        backend: Computational backend ('numpy', 'torch', 'jax')

    Returns:
        LQRResult containing:
            - gain: Optimal feedback gain K (nu, nx)
            - cost_to_go: Riccati solution P (nx, nx)
            - closed_loop_eigenvalues: Eigenvalues of (A - BK)
            - stability_margin: 1 - max(|λ|) for discrete systems (positive = stable)

    Raises:
        ValueError: If matrices have incompatible shapes
        LinAlgError: If Riccati equation has no solution

    Examples
    --------
    >>> # Discretized double integrator (dt = 0.1)
    >>> A = np.array([[1, 0.1], [0, 1]])
    >>> B = np.array([[0.005], [0.1]])
    >>> Q = np.diag([10, 1])
    >>> R = np.array([[0.1]])
    >>>
    >>> result = design_lqr_discrete(A, B, Q, R)
    >>> K = result['gain']
    >>> print(f"Discrete gain: {K}")
    >>> print(f"Stable: {result['stability_margin'] > 0}")  # True
    >>>
    >>> # Apply control in simulation
    >>> x = np.array([1.0, 0.0])
    >>> for k in range(100):
    ...     u = -K @ x
    ...     x = A @ x + B @ u  # Closed-loop dynamics

    Notes
    -----
    - For discrete systems, stability requires |λ| < 1 (inside unit circle)
    - Q must be positive semi-definite
    - R must be positive definite
    - (A, B) should be stabilizable for solution to exist
    """
    # Convert to NumPy
    A_np = _to_numpy(A, backend)
    B_np = _to_numpy(B, backend)
    Q_np = _to_numpy(Q, backend)
    R_np = _to_numpy(R, backend)

    # Validate shapes
    nx = A_np.shape[0]
    nu = B_np.shape[1]

    if A_np.shape != (nx, nx):
        raise ValueError(f"A must be square, got shape {A_np.shape}")
    if B_np.shape[0] != nx:
        raise ValueError(f"B must have {nx} rows, got {B_np.shape[0]}")
    if Q_np.shape != (nx, nx):
        raise ValueError(f"Q must be ({nx}, {nx}), got {Q_np.shape}")
    if R_np.shape != (nu, nu):
        raise ValueError(f"R must be ({nu}, {nu}), got {R_np.shape}")

    # Solve discrete-time ARE
    if N is not None:
        N_np = _to_numpy(N, backend)
        if N_np.shape != (nx, nu):
            raise ValueError(f"N must be ({nx}, {nu}), got {N_np.shape}")
        P = linalg.solve_discrete_are(A_np, B_np, Q_np, R_np, s=N_np)
        # K = (R + B'PB)^{-1}(B'PA + N')
        K = linalg.solve(R_np + B_np.T @ P @ B_np, B_np.T @ P @ A_np + N_np.T)
    else:
        P = linalg.solve_discrete_are(A_np, B_np, Q_np, R_np)
        # K = (R + B'PB)^{-1}B'PA
        K = linalg.solve(R_np + B_np.T @ P @ B_np, B_np.T @ P @ A_np)

    # Closed-loop system
    A_cl = A_np - B_np @ K
    eigenvalues = np.linalg.eigvals(A_cl)

    # Stability margin for discrete: 1 - max(|λ|)
    # Positive margin = stable (all |λ| < 1)
    max_magnitude = np.max(np.abs(eigenvalues))
    stability_margin = 1.0 - max_magnitude

    # Convert back to target backend
    result: LQRResult = {
        "gain": _from_numpy(K, backend),
        "cost_to_go": _from_numpy(P, backend),
        "closed_loop_eigenvalues": _from_numpy(eigenvalues, backend),
        "stability_margin": float(stability_margin),
    }

    return result


# ============================================================================
# Kalman Filter - Optimal State Estimation
# ============================================================================


def design_kalman_filter(
    A: StateMatrix,
    C: OutputMatrix,
    Q: StateMatrix,
    R: OutputMatrix,
    system_type: str = "discrete",
    backend: Backend = "numpy",
) -> KalmanFilterResult:
    """
    Design Kalman filter for optimal state estimation.

    For linear system with Gaussian noise:
        x[k+1] = Ax[k] + Bu[k] + w[k],  w ~ N(0, Q)  (process noise)
        y[k] = Cx[k] + v[k],            v ~ N(0, R)  (measurement noise)

    Kalman filter provides optimal state estimate:
        Discrete: x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])
        Continuous: ˙x̂ = Ax̂ + Bu + L(y - Cx̂)

    Minimizes steady-state estimation error covariance.

    Args:
        A: State matrix (nx, nx)
        C: Output matrix (ny, nx)
        Q: Process noise covariance (nx, nx), Q ≥ 0
        R: Measurement noise covariance (ny, ny), R > 0
        system_type: 'continuous' or 'discrete'
        backend: Computational backend

    Returns:
        KalmanFilterResult containing:
            - gain: Kalman gain L (nx, ny)
            - error_covariance: Steady-state error covariance P (nx, nx)
            - innovation_covariance: Innovation covariance S = CPC' + R (ny, ny)
            - observer_eigenvalues: Eigenvalues of (A - LC)

    Raises:
        ValueError: If matrices have incompatible shapes or invalid system_type
        LinAlgError: If Riccati equation has no solution

    Examples
    --------
    >>> # Discrete Kalman filter
    >>> A = np.array([[1, 0.1], [0, 0.95]])  # Slightly unstable
    >>> C = np.array([[1, 0]])  # Measure position only
    >>> Q = 0.01 * np.eye(2)    # Small process noise
    >>> R = np.array([[0.1]])   # Measurement noise
    >>>
    >>> result = design_kalman_filter(A, C, Q, R, system_type='discrete')
    >>> L = result['gain']
    >>> print(f"Kalman gain: {L}")
    >>>
    >>> # Use in estimation loop
    >>> x_hat = np.zeros(2)
    >>> for k in range(N):
    ...     # Prediction
    ...     x_hat_pred = A @ x_hat + B @ u[k]
    ...
    ...     # Correction
    ...     innovation = y[k] - C @ x_hat_pred
    ...     x_hat = x_hat_pred + L @ innovation
    >>>
    >>> # Check observer stability
    >>> print(f"Observer stable: {np.all(np.abs(result['observer_eigenvalues']) < 1)}")
    >>>
    >>> # Continuous Kalman filter
    >>> result_c = design_kalman_filter(A, C, Q, R, system_type='continuous')
    >>> L_c = result_c['gain']

    Notes
    -----
    - For observability, (A, C) must be observable
    - For detectability, unstable modes must be observable
    - Q must be positive semi-definite (process noise)
    - R must be positive definite (measurement noise)
    - Kalman filter is optimal for linear Gaussian systems
    - For nonlinear systems, use Extended Kalman Filter (EKF) or Unscented Kalman Filter (UKF)
    """
    # Convert to NumPy
    A_np = _to_numpy(A, backend)
    C_np = _to_numpy(C, backend)
    Q_np = _to_numpy(Q, backend)
    R_np = _to_numpy(R, backend)

    # Validate shapes
    nx = A_np.shape[0]
    ny = C_np.shape[0]

    if A_np.shape != (nx, nx):
        raise ValueError(f"A must be square, got shape {A_np.shape}")
    if C_np.shape[1] != nx:
        raise ValueError(f"C must have {nx} columns, got {C_np.shape[1]}")
    if Q_np.shape != (nx, nx):
        raise ValueError(f"Q must be ({nx}, {nx}), got {Q_np.shape}")
    if R_np.shape != (ny, ny):
        raise ValueError(f"R must be ({ny}, {ny}), got {R_np.shape}")

    if system_type not in ["continuous", "discrete"]:
        raise ValueError(f"system_type must be 'continuous' or 'discrete', got '{system_type}'")

    if system_type == "continuous":
        # Continuous-time Kalman filter
        # Solve: PA' + AP - PC'R^{-1}CP + Q = 0 (dual of LQR)
        P = linalg.solve_continuous_are(A_np.T, C_np.T, Q_np, R_np)
        # Kalman gain: L = PC'R^{-1}
        L = P @ C_np.T @ linalg.inv(R_np)
        # Innovation covariance
        S = C_np @ P @ C_np.T + R_np
        # Observer dynamics: A - LC
        A_observer = A_np - L @ C_np
    else:
        # Discrete-time Kalman filter
        # Solve: P = APA' - APC'(CPC' + R)^{-1}CPA' + Q (dual of LQR)
        P = linalg.solve_discrete_are(A_np.T, C_np.T, Q_np, R_np)
        # Innovation covariance: S = CPC' + R
        S = C_np @ P @ C_np.T + R_np
        # Kalman gain: L = APC'S^{-1}
        L = A_np @ P @ C_np.T @ linalg.inv(S)
        # Observer dynamics: A - LC
        A_observer = A_np - L @ C_np

    # Observer eigenvalues (for convergence rate)
    observer_eigenvalues = np.linalg.eigvals(A_observer)

    # Convert back to target backend
    result: KalmanFilterResult = {
        "gain": _from_numpy(L, backend),
        "error_covariance": _from_numpy(P, backend),
        "innovation_covariance": _from_numpy(S, backend),
        "observer_eigenvalues": _from_numpy(observer_eigenvalues, backend),
    }

    return result


# ============================================================================
# LQG - Linear Quadratic Gaussian (LQR + Kalman)
# ============================================================================


def design_lqg(
    A: StateMatrix,
    B: InputMatrix,
    C: OutputMatrix,
    Q_state: StateMatrix,
    R_control: InputMatrix,
    Q_process: StateMatrix,
    R_measurement: OutputMatrix,
    system_type: str = "discrete",
    backend: Backend = "numpy",
) -> LQGResult:
    """
    Design Linear Quadratic Gaussian (LQG) controller.

    Combines LQR controller with Kalman filter estimator.
    Separation principle allows independent design of controller and estimator.

    System:
        x[k+1] = Ax[k] + Bu[k] + w[k],  w ~ N(0, Q_process)
        y[k] = Cx[k] + v[k],            v ~ N(0, R_measurement)

    Controller: u[k] = -Kx̂[k] (feedback on estimate)
    Estimator: x̂[k+1] = Ax̂[k] + Bu[k] + L(y[k] - Cx̂[k])

    Args:
        A: State matrix (nx, nx)
        B: Input matrix (nx, nu)
        C: Output matrix (ny, nx)
        Q_state: LQR state cost matrix (nx, nx)
        R_control: LQR control cost matrix (nu, nu)
        Q_process: Process noise covariance (nx, nx)
        R_measurement: Measurement noise covariance (ny, ny)
        system_type: 'continuous' or 'discrete'
        backend: Computational backend

    Returns:
        LQGResult containing:
            - controller_gain: LQR gain K (nu, nx)
            - estimator_gain: Kalman gain L (nx, ny)
            - controller_riccati: LQR Riccati solution P_c (nx, nx)
            - estimator_covariance: Kalman covariance P_e (nx, nx)
            - closed_loop_eigenvalues: Controller eigenvalues of (A - BK)
            - observer_eigenvalues: Estimator eigenvalues of (A - LC)

    Examples
    --------
    >>> # Design LQG controller
    >>> A = np.array([[1, 0.1], [0, 0.9]])
    >>> B = np.array([[0], [0.1]])
    >>> C = np.array([[1, 0]])  # Measure position
    >>>
    >>> # LQR weights
    >>> Q_state = np.diag([10, 1])
    >>> R_control = np.array([[0.1]])
    >>>
    >>> # Noise covariances
    >>> Q_process = 0.01 * np.eye(2)
    >>> R_measurement = np.array([[0.1]])
    >>>
    >>> result = design_lqg(
    ...     A, B, C,
    ...     Q_state, R_control,
    ...     Q_process, R_measurement,
    ...     system_type='discrete'
    ... )
    >>>
    >>> K = result['controller_gain']
    >>> L = result['estimator_gain']
    >>>
    >>> # Implementation
    >>> x_hat = np.zeros(2)
    >>> for k in range(N):
    ...     # Control
    ...     u = -K @ x_hat
    ...
    ...     # Estimation
    ...     x_hat_pred = A @ x_hat + B @ u
    ...     innovation = y[k] - C @ x_hat_pred
    ...     x_hat = x_hat_pred + L @ innovation

    Notes
    -----
    - Separation principle: LQR and Kalman can be designed independently
    - LQG is optimal for linear systems with Gaussian noise
    - Closed-loop has eigenvalues of both controller and estimator
    - Controller must stabilize system
    - Estimator must converge faster than controller for good performance
    - Trade-off: Lower Q_process/R_measurement → more aggressive estimator
    """
    # Design LQR controller
    if system_type == "continuous":
        lqr_result = design_lqr_continuous(A, B, Q_state, R_control, backend=backend)
    elif system_type == "discrete":
        lqr_result = design_lqr_discrete(A, B, Q_state, R_control, backend=backend)
    else:
        raise ValueError(f"system_type must be 'continuous' or 'discrete', got '{system_type}'")

    # Design Kalman filter estimator
    kalman_result = design_kalman_filter(A, C, Q_process, R_measurement, system_type, backend)

    # Construct LQG result
    result: LQGResult = {
        "controller_gain": lqr_result["gain"],
        "estimator_gain": kalman_result["gain"],
        "controller_riccati": lqr_result["cost_to_go"],
        "estimator_covariance": kalman_result["error_covariance"],
        "closed_loop_eigenvalues": lqr_result["closed_loop_eigenvalues"],
        "observer_eigenvalues": kalman_result["observer_eigenvalues"],
    }

    return result


# ============================================================================
# Stability Analysis
# ============================================================================


def analyze_stability(
    A: StateMatrix,
    system_type: str = "continuous",
    tolerance: float = 1e-10,
) -> StabilityInfo:
    """
    Analyze system stability via eigenvalue analysis.

    Stability criteria:
        Continuous (dx/dt = Ax): All Re(λ) < 0 (left half-plane)
        Discrete (x[k+1] = Ax): All |λ| < 1 (inside unit circle)

    Args:
        A: State matrix (nx, nx)
        system_type: 'continuous' or 'discrete'
        tolerance: Tolerance for marginal stability detection

    Returns:
        StabilityInfo containing:
            - eigenvalues: Eigenvalues of A (complex array)
            - magnitudes: |λ| for all eigenvalues
            - max_magnitude: max(|λ|) = spectral radius
            - spectral_radius: Same as max_magnitude
            - is_stable: True if asymptotically stable
            - is_marginally_stable: True if critically stable
            - is_unstable: True if unstable

    Examples
    --------
    >>> # Stable continuous system
    >>> A = np.array([[0, 1], [-2, -3]])
    >>> stability = analyze_stability(A, system_type='continuous')
    >>> print(stability['is_stable'])  # True
    >>> print(stability['eigenvalues'])  # [-1, -2]
    >>>
    >>> # Unstable continuous system
    >>> A_unstable = np.array([[1, 1], [0, 1]])
    >>> stability = analyze_stability(A_unstable, system_type='continuous')
    >>> print(stability['is_unstable'])  # True
    >>>
    >>> # Stable discrete system
    >>> Ad = np.array([[0.9, 0.1], [0, 0.8]])
    >>> stability = analyze_stability(Ad, system_type='discrete')
    >>> print(stability['is_stable'])  # True
    >>> print(stability['spectral_radius'])  # 0.9
    >>>
    >>> # Marginally stable (on boundary)
    >>> A_marginal = np.array([[0, 1], [-1, 0]])  # Pure oscillation
    >>> stability = analyze_stability(A_marginal, system_type='continuous')
    >>> print(stability['is_marginally_stable'])  # True

    Notes
    -----
    - Marginal stability: Eigenvalues on stability boundary
        - Continuous: Re(λ) = 0 (imaginary axis)
        - Discrete: |λ| = 1 (unit circle)
    - Asymptotic stability: All trajectories converge to zero
    - Lyapunov stability: Bounded trajectories (includes marginal)
    """
    # Convert to NumPy
    A_np = np.asarray(A)

    if A_np.ndim != 2 or A_np.shape[0] != A_np.shape[1]:
        raise ValueError(f"A must be square matrix, got shape {A_np.shape}")

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(A_np)
    magnitudes = np.abs(eigenvalues)
    max_magnitude = np.max(magnitudes)

    if system_type == "continuous":
        # Continuous: check real parts
        max_real = np.max(np.real(eigenvalues))
        is_stable = max_real < -tolerance
        is_marginally_stable = np.abs(max_real) <= tolerance
        is_unstable = max_real > tolerance
    elif system_type == "discrete":
        # Discrete: check magnitudes
        is_stable = max_magnitude < 1.0 - tolerance
        is_marginally_stable = np.abs(max_magnitude - 1.0) <= tolerance
        is_unstable = max_magnitude > 1.0 + tolerance
    else:
        raise ValueError(f"system_type must be 'continuous' or 'discrete', got '{system_type}'")

    result: StabilityInfo = {
        "eigenvalues": eigenvalues,
        "magnitudes": magnitudes,
        "max_magnitude": float(max_magnitude),
        "spectral_radius": float(max_magnitude),
        "is_stable": bool(is_stable),
        "is_marginally_stable": bool(is_marginally_stable),
        "is_unstable": bool(is_unstable),
    }

    return result


# ============================================================================
# Controllability Analysis
# ============================================================================


def analyze_controllability(
    A: StateMatrix,
    B: InputMatrix,
    tolerance: float = 1e-10,
) -> ControllabilityInfo:
    """
    Test controllability of linear system (A, B).

    A system is controllable if all states can be driven to any desired
    value in finite time using appropriate control inputs.

    Controllability test:
        rank(C) = n, where C = [B, AB, A²B, ..., Aⁿ⁻¹B]

    Args:
        A: State matrix (nx, nx)
        B: Input matrix (nx, nu)
        tolerance: Tolerance for rank computation

    Returns:
        ControllabilityInfo containing:
            - controllability_matrix: C = [B, AB, ...] (nx, nx*nu)
            - rank: Rank of controllability matrix
            - is_controllable: True if rank = nx (full rank)
            - uncontrollable_modes: Eigenvalues of uncontrollable subspace (if any)

    Examples
    --------
    >>> # Fully controllable
    >>> A = np.array([[0, 1], [-2, -3]])
    >>> B = np.array([[0], [1]])
    >>> info = analyze_controllability(A, B)
    >>> print(info['is_controllable'])  # True
    >>> print(info['rank'])  # 2
    >>>
    >>> # Uncontrollable system (diagonal with identical input)
    >>> A = np.array([[1, 0], [0, 2]])
    >>> B = np.array([[1], [1]])  # Can't control modes independently
    >>> info = analyze_controllability(A, B)
    >>> print(info['is_controllable'])  # False
    >>> print(info['rank'])  # 1
    >>>
    >>> # Single-input controllable
    >>> A = np.array([[0, 1, 0], [0, 0, 1], [-1, -2, -3]])
    >>> B = np.array([[0], [0], [1]])
    >>> info = analyze_controllability(A, B)
    >>> print(info['is_controllable'])  # True

    Notes
    -----
    - Controllability is necessary for pole placement
    - Stabilizability: Unstable modes must be controllable (weaker condition)
    - Numerical issues: Use SVD for better numerical stability
    - For large systems, consider PBH test or Gram matrix
    """
    # Convert to NumPy
    A_np = np.asarray(A)
    B_np = np.asarray(B)

    nx = A_np.shape[0]
    nu = B_np.shape[1]

    if A_np.shape != (nx, nx):
        raise ValueError(f"A must be square, got shape {A_np.shape}")
    if B_np.shape[0] != nx:
        raise ValueError(f"B must have {nx} rows, got {B_np.shape[0]}")

    # Build controllability matrix: C = [B, AB, A²B, ..., Aⁿ⁻¹B]
    C = np.zeros((nx, nx * nu))
    C[:, :nu] = B_np

    AB = B_np.copy()
    for i in range(1, nx):
        AB = A_np @ AB
        C[:, i * nu : (i + 1) * nu] = AB

    # Compute rank
    rank = np.linalg.matrix_rank(C, tol=tolerance)
    is_controllable = rank == nx

    # Find uncontrollable modes (if any)
    # TODO: Implement PBH test or controllability decomposition
    uncontrollable_modes = None
    if not is_controllable:
        # For now, just note that some modes are uncontrollable
        # Full implementation would use Kalman decomposition
        pass

    result: ControllabilityInfo = {
        "controllability_matrix": C,
        "rank": int(rank),
        "is_controllable": bool(is_controllable),
        "uncontrollable_modes": uncontrollable_modes,
    }

    return result


# ============================================================================
# Observability Analysis
# ============================================================================


def analyze_observability(
    A: StateMatrix,
    C: OutputMatrix,
    tolerance: float = 1e-10,
) -> ObservabilityInfo:
    """
    Test observability of linear system (A, C).

    A system is observable if the initial state can be determined from
    output measurements over a finite time interval.

    Observability test:
        rank(O) = n, where O = [C; CA; CA²; ...; CAⁿ⁻¹]

    Args:
        A: State matrix (nx, nx)
        C: Output matrix (ny, nx)
        tolerance: Tolerance for rank computation

    Returns:
        ObservabilityInfo containing:
            - observability_matrix: O = [C; CA; ...] (nx*ny, nx)
            - rank: Rank of observability matrix
            - is_observable: True if rank = nx (full rank)
            - unobservable_modes: Eigenvalues of unobservable subspace (if any)

    Examples
    --------
    >>> # Fully observable
    >>> A = np.array([[0, 1], [-2, -3]])
    >>> C = np.array([[1, 0]])  # Measure position only
    >>> info = analyze_observability(A, C)
    >>> print(info['is_observable'])  # True
    >>> print(info['rank'])  # 2
    >>>
    >>> # Unobservable system
    >>> A = np.array([[1, 0], [0, 2]])
    >>> C = np.array([[1, 1]])  # Can't distinguish states
    >>> info = analyze_observability(A, C)
    >>> print(info['is_observable'])  # False
    >>>
    >>> # Full state measurement
    >>> A = np.array([[0, 1, 0], [0, 0, 1], [-1, -2, -3]])
    >>> C = np.eye(3)  # Measure all states
    >>> info = analyze_observability(A, C)
    >>> print(info['is_observable'])  # True

    Notes
    -----
    - Observability is necessary for state estimation (Kalman filter)
    - Detectability: Unstable modes must be observable (weaker condition)
    - Dual to controllability: (A, C) observable ⟺ (A', C') controllable
    - For large systems, use dual controllability test
    """
    # Convert to NumPy
    A_np = np.asarray(A)
    C_np = np.asarray(C)

    nx = A_np.shape[0]
    ny = C_np.shape[0]

    if A_np.shape != (nx, nx):
        raise ValueError(f"A must be square, got shape {A_np.shape}")
    if C_np.shape[1] != nx:
        raise ValueError(f"C must have {nx} columns, got {C_np.shape[1]}")

    # Build observability matrix: O = [C; CA; CA²; ...; CAⁿ⁻¹]
    O = np.zeros((nx * ny, nx))
    O[:ny, :] = C_np

    CA = C_np.copy()
    for i in range(1, nx):
        CA = CA @ A_np
        O[i * ny : (i + 1) * ny, :] = CA

    # Compute rank
    rank = np.linalg.matrix_rank(O, tol=tolerance)
    is_observable = rank == nx

    # Find unobservable modes (if any)
    # TODO: Implement PBH test or observability decomposition
    unobservable_modes = None
    if not is_observable:
        # For now, just note that some modes are unobservable
        # Full implementation would use Kalman decomposition
        pass

    result: ObservabilityInfo = {
        "observability_matrix": O,
        "rank": int(rank),
        "is_observable": bool(is_observable),
        "unobservable_modes": unobservable_modes,
    }

    return result


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # LQR
    "design_lqr_continuous",
    "design_lqr_discrete",
    # Kalman Filter
    "design_kalman_filter",
    # LQG
    "design_lqg",
    # Analysis
    "analyze_stability",
    "analyze_controllability",
    "analyze_observability",
]
