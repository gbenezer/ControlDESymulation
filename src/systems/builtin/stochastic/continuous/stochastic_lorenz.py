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
Stochastic Lorenz System - Chaos Meets Randomness
==================================================

This module provides the stochastic Lorenz system, combining deterministic chaos
with random perturbations to create one of the most complex dynamical behaviors
in mathematical physics. The stochastic Lorenz system serves as:

- The canonical example of chaos plus randomness interaction
- A fundamental model for understanding predictability limits in turbulent systems
- A benchmark for distinguishing deterministic chaos from stochastic noise
- An illustration of how noise affects strange attractors and fractal structure
- A test case for data assimilation and filtering in chaotic systems

The stochastic Lorenz system represents the ultimate challenge in nonlinear
dynamics: separating deterministic unpredictability (chaos) from random
unpredictability (stochasticity) in a system exhibiting both simultaneously.

Physical Context
----------------

**Lorenz's Discovery (1963):**

Edward Lorenz discovered deterministic chaos while studying atmospheric
convection. Simplified Rayleigh-Bénard convection equations led to:

    dx/dt = σ·(y - x)
    dy/dt = x·(ρ - z) - y  
    dz/dt = x·y - β·z

**Key Discovery:**
Small changes in initial conditions → vastly different trajectories
(sensitive dependence on initial conditions).

This demolished predictability assumptions in weather forecasting!

**Real Atmosphere:**

But the atmosphere has BOTH chaos AND noise:
- Deterministic chaos: Nonlinear fluid dynamics
- Stochastic noise: Turbulent fluctuations, measurement errors
- Combined: Even less predictable

**Stochastic Lorenz System:**

Add noise to Lorenz equations:
    dx = σ·(y - x)·dt + σ_x·dW_x
    dy = (x·(ρ - z) - y)·dt + σ_y·dW_y
    dz = (x·y - β·z)·dt + σ_z·dW_z

**Questions:**
- How does noise affect strange attractor?
- Can we distinguish chaos from noise in data?
- Does noise destroy or enhance chaos?
- What are predictability limits?

Mathematical Background
-----------------------

**Deterministic Lorenz System:**

Three coupled nonlinear ODEs:
    ẋ = σ·(y - x)
    ẏ = x·(ρ - z) - y
    ż = x·y - β·z

where:
- σ: Prandtl number (ratio of momentum/thermal diffusivity)
- ρ: Rayleigh number (driving force for convection)
- β: Geometric factor (aspect ratio)

**Classic Parameters (Chaotic):**
    σ = 10, ρ = 28, β = 8/3

Creates famous "butterfly" strange attractor.

**Stochastic Extension:**

Add Brownian noise to each equation:
    dX = f(X)·dt + Σ·dW

where:
- f(X): Deterministic Lorenz drift
- Σ: Diffusion matrix (noise intensities)
- W: Vector Wiener process (3D)

**Noise Structure Options:**

1. **Additive (Implemented):**
   Σ = diag(σ_x, σ_y, σ_z) - constant noise

2. **Multiplicative:**
   Σ(X) = diag(σ_x·x, σ_y·y, σ_z·z) - state-dependent

3. **Correlated:**
   Full 3×3 matrix Σ - common shocks

Chaos vs Randomness
-------------------

**Deterministic Chaos:**

Characteristics:
- Sensitive dependence on ICs: |δX(t)| ~ |δX(0)|·exp(λ·t)
- Positive Lyapunov exponent: λ > 0
- Strange attractor: Fractal dimension
- Long-term unpredictable but deterministic

**Stochastic Noise:**

Characteristics:
- Random forcing from external source
- Brownian motion (continuous random walk)
- Gaussian increments (typically)
- Unpredictable by definition

**Combined System:**

Chaos + noise creates:
- Enhanced unpredictability (both sources compound)
- Destroyed fractal structure (noise smooths attractor)
- Modified Lyapunov exponents (noise affects divergence rate)
- Fundamental limits on predictability

**Distinguishing Chaos from Noise:**

From data, how to tell if unpredictability is chaos or noise?

**Methods:**
1. **Lyapunov Exponents:** 
   - Chaos: Positive λ from deterministic dynamics
   - Noise: Apparent λ from random fluctuations
   - Combined: Must separate contributions

2. **Attractor Reconstruction:**
   - Chaos: Clean fractal structure (Takens embedding)
   - Noise: Blurred structure
   - Combined: Need denoising

3. **Recurrence Analysis:**
   - Chaos: Recurrence patterns
   - Noise: Random recurrences
   - Recurrence plots distinguish

4. **Power Spectrum:**
   - Chaos: Broadband but structured
   - Noise: White (flat spectrum)
   - Combined: Structured background + white noise

Stochastic Lorenz Phenomena
----------------------------

**1. Noise-Perturbed Strange Attractor:**

Deterministic: Trajectories on 2D manifold (strange attractor).
With noise: Trajectories diffuse around manifold.
- Small noise: Attractor recognizable (slightly blurred)
- Large noise: Attractor destroyed (fills volume)

**2. Noise-Induced Transitions:**

Lorenz has two "wings" (lobes around equilibria C± = (±√(β(ρ-1)), ±√(β(ρ-1)), ρ-1)).
Deterministic: Chaotic switching between wings.
With noise: Transition statistics change.
- Can increase or decrease switching rate
- Depends on noise intensity and correlation

**3. Synchronization Under Noise:**

Two coupled stochastic Lorenz systems:
- Common noise → can synchronize
- Independent noise → desynchronizes
- Applications: Secure communications, neuroscience

**4. Modified Lyapunov Exponents:**

Largest Lyapunov exponent λ_max changes with noise:
- Small noise: λ ≈ λ_det (chaos dominates)
- Large noise: λ increases (noise adds randomness)
- Crossover: Where chaos and noise equally important

**5. Intermittency:**

Alternation between quasi-regular and chaotic bursts.
Noise can trigger or suppress intermittent phases.

**6. Stochastic Bifurcations:**

As noise σ increases:
- P-bifurcation: Stationary distribution shape changes
- D-bifurcation: Lyapunov exponent changes sign
- Different from deterministic bifurcation diagram

Predictability Limits
---------------------

**Atmospheric Analogy:**

Lorenz created this for weather. Stochastic Lorenz models:
- Deterministic chaos: Butterfly effect (small errors grow)
- Stochastic forcing: Unresolved turbulence, measurement errors

**Predictability Horizon:**

Time beyond which forecast no better than climatology:
    T_pred = (1/λ)·ln(L_attractor/δ_0)

where:
- λ: Lyapunov exponent (chaos rate)
- L_attractor: Attractor size
- δ_0: Initial error (IC + noise)

**With Noise:**
    δ(t)² = δ_0²·exp(2λ·t) + (σ²/λ)·(exp(2λ·t) - 1)

Error grows from both IC uncertainty AND noise accumulation.

**Implications:**
- Even perfect IC (δ_0=0): Noise limits prediction
- T_pred finite (weather: ~2 weeks, Lorenz model: ~10 Lyapunov times)
- Cannot predict arbitrarily far ahead

Data Assimilation
-----------------

**Problem:** Estimate current state from noisy observations.

**Challenge:** 
- Chaotic divergence amplifies estimation errors
- Noise obscures true state
- Must balance model prediction vs observations

**Approaches:**

**1. Extended Kalman Filter:**
- Linearize around current estimate
- Problem: Linearization poor for chaotic system
- Can diverge

**2. Ensemble Kalman Filter (EnKF):**
- Monte Carlo: Propagate ensemble of states
- Covariance from ensemble spread
- Standard in weather forecasting (NWP)

**3. Particle Filter:**
- Fully nonlinear
- Expensive (need many particles for 3D)
- Robust for chaotic systems

**4. 4D-Var (Variational):**
- Optimize trajectory over time window
- Used operationally in weather prediction
- Handles nonlinearity better

Applications
------------

**1. Atmospheric Science:**

**Weather Prediction:**
Simplified model of atmospheric dynamics:
- x, y, z: Convection modes
- Chaos: Inherent unpredictability
- Noise: Unresolved processes
- Data assimilation: Forecast initialization

**Climate Modeling:**
- Long-time statistics (attractors)
- Effect of noise on climate
- Tipping points under noise

**2. Fluid Dynamics:**

**Turbulence:**
- Rayleigh-Bénard convection
- Transition to turbulence
- Statistical properties

**3. Nonlinear Optics:**

**Laser Dynamics:**
- Multimode lasers
- Chaos in intensity
- Noise from spontaneous emission

**4. Chemical Reactions:**

**Belousov-Zhabotinsky:**
- Oscillating chemical reaction
- Chaotic regimes
- Molecular noise

**5. Neuroscience:**

**Neural Mass Models:**
- Population activity
- Chaotic firing patterns
- Synaptic noise

**6. Control Theory:**

**Chaos Control:**
- Stabilize unstable periodic orbits (UPOs)
- Targeting: Reach specific UPO
- Under noise: Probabilistic control

**7. Machine Learning:**

**Chaotic Time Series:**
- Prediction with neural networks
- Chaos vs noise identification
- Uncertainty quantification

Numerical Integration
---------------------

**Challenges:**

1. **Chaos:**
   - Sensitive dependence on numerics
   - Need small dt for accuracy
   - Long-time simulations expensive

2. **Stiffness:**
   - For large σ: Moderately stiff
   - Fast noise, slower deterministic drift

3. **Attractor Capture:**
   - Must accurately represent strange attractor
   - Numerical errors can distort geometry

**Recommended Methods:**

1. **Euler-Maruyama:**
   - dt = 0.001-0.01 (smaller than deterministic)
   - Simple, robust
   - Check convergence carefully

2. **Runge-Kutta SDE:**
   - Higher order
   - Better for clean attractor
   - More computational cost

3. **Implicit Methods:**
   - For very noisy (large σ)
   - Stabilizes stiffness

**Validation:**

Compare with deterministic:
- Small σ: Should resemble Lorenz attractor
- Large σ: Fills space around attractor
- Check Lyapunov exponents converge

Common Pitfalls
---------------

1. **Confusing Chaos with Noise:**
   - Both create unpredictability
   - Chaos: Deterministic but sensitive
   - Noise: Random by nature
   - Need analysis to distinguish

2. **Too Large dt:**
   - Chaos requires small dt (deterministic)
   - Noise requires smaller dt (stochastic)
   - Use dt < 0.01 for both

3. **Single Trajectory:**
   - One path not representative
   - Need ensemble for statistics
   - Attractor from many trajectories

4. **Ignoring Initial Transient:**
   - Takes time ~50 to reach attractor
   - Discard initial transient for statistics

5. **Linearization:**
   - EKF hopeless (highly nonlinear, chaotic)
   - Use ensemble or particle methods

6. **Over-Interpreting Structure:**
   - Noise obscures deterministic structure
   - Statistical tests needed to validate findings

**Impact:**
Lorenz system demonstrated:
- Long-term prediction impossible (chaos)
- Stochastic forcing further limits prediction
- Statistical properties still predictable (climate vs weather)

"""

import numpy as np
import sympy as sp
from typing import Optional

from src.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem


class StochasticLorenz(ContinuousStochasticSystem):
    """
    Stochastic Lorenz system - deterministic chaos with random perturbations.

    Combines the iconic chaotic Lorenz attractor with continuous Brownian
    noise, creating a fundamental model for understanding the interplay
    between deterministic chaos and stochastic fluctuations.

    Stochastic Differential Equations
    ----------------------------------
    The stochastic Lorenz equations:

        dx = σ·(y - x)·dt + σ_x·dW_x
        dy = (x·(ρ - z) - y)·dt + σ_y·dW_y
        dz = (x·y - β·z)·dt + σ_z·dW_z

    where:
        - x: Convection intensity (horizontal circulation) [dimensionless]
        - y: Horizontal temperature variation [dimensionless]
        - z: Vertical temperature variation [dimensionless]
        - σ: Prandtl number (fluid property, typically 10)
        - ρ: Rayleigh number (driving force, typically 28 for chaos)
        - β: Geometric factor (aspect ratio, typically 8/3)
        - σ_x, σ_y, σ_z: Noise intensities [1/√s]
        - W_x, W_y, W_z: Independent Wiener processes

    **Deterministic Part:**
    Classic Lorenz dynamics - creates strange attractor and chaos.

    **Stochastic Part:**
    Additive noise representing:
    - Unresolved turbulent fluctuations
    - Measurement errors
    - Model uncertainty
    - External forcing variability

    Physical Interpretation
    -----------------------
    **Original Context: Rayleigh-Bénard Convection**

    Fluid layer heated from below:
    - x: Convection roll intensity
    - y: Temperature difference (horizontal)
    - z: Temperature difference (vertical)

    **Parameters:**
    - σ = 10: Prandtl number (momentum/thermal diffusivity)
    - ρ = 28: Rayleigh number (buoyancy/viscosity)
    - β = 8/3: Geometry (cell aspect ratio)

    **Atmospheric Interpretation:**
    - x: Zonal wind component
    - y: Meridional wind component  
    - z: Temperature deviation

    **Noise Sources:**
    - Turbulent eddies below grid scale
    - Latent heat release variations
    - Radiation fluctuations
    - Measurement uncertainty

    Deterministic Chaos Primer
    ---------------------------
    **Strange Attractor:**

    For σ=10, ρ=28, β=8/3:
    - Butterfly-shaped attractor in 3D
    - Fractal dimension: D ≈ 2.06
    - Trajectories never repeat
    - Sensitive to initial conditions

    **Lyapunov Exponents:**
    - λ₁ ≈ +0.9: Positive (chaos!)
    - λ₂ ≈ 0: Zero (volume-preserving direction)
    - λ₃ ≈ -14.6: Negative (dissipation)

    **Predictability Time:**
    - Doubling time: τ ≈ 1/λ₁ ≈ 1.1 Lorenz time units
    - After ~10 doubling times: Completely unpredictable

    **Equilibria:**
    - Origin (0,0,0): Unstable for ρ > 1
    - C± = (±√(β(ρ-1)), ±√(β(ρ-1)), ρ-1): Unstable for ρ > ρ_crit

    All unstable → trajectories on attractor.

    Effect of Noise on Chaos
    -------------------------

    **Small Noise (σ_noise << 1):**
    - Attractor recognizable (slightly blurred)
    - Lyapunov exponent nearly unchanged
    - Chaos dominates, noise is perturbation

    **Medium Noise (σ_noise ~ 1):**
    - Attractor significantly blurred
    - Lyapunov exponent increases
    - Chaos and noise comparable

    **Large Noise (σ_noise >> 1):**
    - Attractor destroyed (fills volume)
    - Noise dominates dynamics
    - Chaos irrelevant (randomness wins)

    **Crossover Scale:**
    When σ_noise ~ √λ_max, chaos and noise equally important.

    **Fractal Dimension:**
    Noise increases effective dimension:
    - Deterministic: D ≈ 2.06 (fractal)
    - With noise: D → 3 (fills space)

    **Predictability:**
    Combined error growth:
        δ²(t) = δ_0²·exp(2λt) + (σ²/λ)·(exp(2λt) - 1)

    Both exponential growth (chaos) and accumulation (noise).

    Key Properties
    --------------

    **1. Nonlinearity:**
    Quadratic terms: x·y, x·z create complex dynamics.

    **2. Chaos (Deterministic Part):**
    Positive Lyapunov exponent → sensitive dependence.

    **3. Strange Attractor:**
    Fractal structure (deterministic) blurred by noise.

    **4. Non-Stationary Locally:**
    On attractor, system never settles.

    **5. Stationary Globally:**
    Probability distribution on attractor converges (Fokker-Planck).

    **6. Three-Dimensional:**
    Minimum dimension for chaos (Poincaré-Bendixson theorem).

    **7. Additive Noise (This Implementation):**
    Constant diffusion coefficients (simplest model).

    Mathematical Properties
    -----------------------
    **No Closed-Form Solution:**
    Even deterministic Lorenz has no analytical solution.
    Must integrate numerically.

    **Fokker-Planck Equation:**

    Probability density p(x,y,z,t) satisfies:
        ∂p/∂t = -∇·(f·p) + (1/2)·Σᵢ ∂²((σᵢ²)·p)/∂xᵢ²

    For stationary distribution: ∂p/∂t = 0

    **Stationary Distribution:**
    Concentrated on noise-perturbed attractor:
    - No simple analytical form
    - Compute via long-time simulation (ergodic hypothesis)

    **Moments:**
    No simple analytical expressions.
    Compute empirically from ensemble or long trajectory.

    State Space
    -----------
    State: X = [x, y, z] ∈ ℝ³
        - Unbounded (but attracted to finite region)
        - Chaotic wandering on strange attractor
        - Noise perturbs trajectories

    Control: u (optional, not standard)
        - Can add control to stabilize UPOs
        - Chaos control applications

    Noise: W = [W_x, W_y, W_z] ∈ ℝ³
        - Three independent Wiener processes
        - Models unresolved fluctuations

    Parameters
    ----------
    sigma : float, default=10.0
        Prandtl number (deterministic parameter)
        - Controls x-y coupling strength
        - Standard: 10

    rho : float, default=28.0
        Rayleigh number (deterministic parameter)
        - Controls bifurcations
        - ρ < 1: Stable origin
        - 1 < ρ < 24.74: Stable C±
        - ρ > 24.74: Chaos
        - Standard: 28 (chaotic regime)

    beta : float, default=8/3
        Geometric factor (deterministic parameter)
        - Standard: 8/3 ≈ 2.667

    sigma_x : float, default=0.1
        Noise intensity for x [1/√s]

    sigma_y : float, default=0.1
        Noise intensity for y [1/√s]

    sigma_z : float, default=0.1
        Noise intensity for z [1/√s]

    Stochastic Properties
    ---------------------
    - System Type: NONLINEAR (quadratic)
    - Noise Type: ADDITIVE (constant)
    - SDE Type: Itô
    - Noise Dimension: nw = 3
    - Chaotic: Yes (deterministic part)
    - Stationary: Yes (global, on attractor)
    - Ergodic: Yes (time = ensemble averages)

    Applications
    ------------
    **1. Meteorology:**
    - Weather prediction limits
    - Ensemble forecasting
    - Data assimilation (EnKF, 4D-Var)

    **2. Climate Science:**
    - Chaos vs variability
    - Tipping points under noise
    - Extreme events

    **3. Chaos Theory:**
    - Noise effects on strange attractors
    - Modified Lyapunov exponents
    - Stochastic bifurcations

    **4. Signal Processing:**
    - Chaotic time series analysis
    - Noise reduction
    - Chaos vs noise discrimination

    **5. Nonlinear Filtering:**
    - Benchmark for EnKF, particle filters
    - Chaotic dynamics challenge

    **6. Control:**
    - Chaos control under noise
    - Targeting unstable periodic orbits
    - Synchronization

    Numerical Integration
    ---------------------
    **Recommended:**
    - Euler-Maruyama: dt = 0.001-0.01
    - Smaller than deterministic (chaos + noise)
    - Check attractor structure

    **Validation:**
    - Small noise: Should resemble Lorenz attractor
    - Statistics: Mean, variance converge
    - Lyapunov: Positive λ₁ (chaos indicator)

    Monte Carlo Analysis
    ---------------------
    **Ensemble Methods:**

    Run N = 100-1,000 trajectories:
    - Mean trajectory (climbs to attractor)
    - Spread on attractor
    - Long-time statistics

    **Attractor Visualization:**
    - 3D scatter plot of all trajectories
    - Should show butterfly shape (if noise small)

    Comparison with Deterministic
    ------------------------------
    **Deterministic Lorenz:**
    - Pure chaos (no external randomness)
    - Clean strange attractor
    - Exponential divergence only

    **Stochastic Lorenz:**
    - Chaos + noise combined
    - Blurred attractor
    - Exponential + diffusive error growth

    **When Stochastic Needed:**
    - Real data has noise (always)
    - Predictability limits more realistic
    - Data assimilation applications

    Limitations
    -----------
    - Additive noise only
    - Constant noise (not state-dependent)
    - Independent noise (no correlation)
    - No control (in standard form)

    Extensions
    ----------
    - Multiplicative noise: σᵢ(X)
    - Colored noise: OU driving processes
    - Coupled Lorenz systems
    - Controlled Lorenz
    - Stochastic Rössler system

    See Also
    --------
    Lorenz : Deterministic version
    """

    def define_system(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0/3.0,
        sigma_x: float = 0.1,
        sigma_y: float = 0.1,
        sigma_z: float = 0.1,
    ):
        """
        Define stochastic Lorenz system dynamics.

        Parameters
        ----------
        sigma : float, default=10.0
            Prandtl number (deterministic parameter)
            - Controls x-y coupling
            - Standard: 10
            - Ratio of momentum/thermal diffusivity

        rho : float, default=28.0
            Rayleigh number (deterministic parameter)
            - Controls convection strength
            - ρ < 1: Stable origin (no convection)
            - 1 < ρ < 24.74: Stable C± (steady convection)
            - ρ > 24.74: Chaos (turbulent convection)
            - Standard: 28 (chaotic regime)

        beta : float, default=8/3
            Geometric factor (deterministic parameter)
            - Standard: 8/3 ≈ 2.667
            - Related to cell aspect ratio

        sigma_x : float, default=0.1
            Noise intensity for x [1/√s]
            - Unresolved turbulence
            - Typical: 0.01-1.0

        sigma_y : float, default=0.1
            Noise intensity for y [1/√s]

        sigma_z : float, default=0.1
            Noise intensity for z [1/√s]

        Notes
        -----
        **Parameter Regimes:**

        **ρ (Rayleigh Number):**
        - ρ < 1: No convection (stable origin)
        - 1 < ρ < 13.93: Steady convection (stable C±)
        - 13.93 < ρ < 24.06: Transient chaos
        - 24.74 < ρ < 30: Strange attractor (standard)
        - ρ > 30: Different attractor structures

        **Classic Chaotic:**
        σ = 10, ρ = 28, β = 8/3

        **Noise Intensity Guidelines:**

        Relative to chaos strength:
        - σ_noise < 0.1: Noise negligible (chaos dominates)
        - σ_noise ~ 0.1-1.0: Noise perturbs chaos (typical)
        - σ_noise > 1.0: Noise dominates (chaos obscured)

        **Physical Interpretation:**

        For atmospheric application:
        - σ_noise represents unresolved processes
        - Should be calibrated from data (residuals)
        - Typical: 0.1-0.5 for simplified models

        **Noise Correlation:**

        Currently: Independent noise on each variable.

        In reality: May be correlated (common weather systems).
        Extension: Use full 3×3 covariance matrix.

        **Chaos Indicators:**

        Check if deterministic part is chaotic:
        - Compute Lyapunov exponents from noise-free simulation
        - λ₁ > 0 indicates chaos
        - For ρ = 28: λ₁ ≈ 0.9 (strongly chaotic)

        **Noise Impact on Lyapunov:**

        Noise increases apparent Lyapunov exponent:
            λ_apparent ≈ λ_det + σ²/(2·scale²)

        where scale is attractor size.
        """
        if sigma <= 0:
            raise ValueError(f"sigma (Prandtl) must be positive, got {sigma}")
        if rho <= 0:
            raise ValueError(f"rho (Rayleigh) must be positive, got {rho}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if any(s < 0 for s in [sigma_x, sigma_y, sigma_z]):
            raise ValueError("Noise intensities must be non-negative")

        # State variables
        x, y, z = sp.symbols("x y z", real=True)

        # Deterministic parameters
        sigma_sym = sp.symbols("sigma", positive=True)
        rho_sym = sp.symbols("rho", positive=True)
        beta_sym = sp.symbols("beta", positive=True)

        # Noise parameters
        sigma_x_sym = sp.symbols("sigma_x", real=True, nonnegative=True)
        sigma_y_sym = sp.symbols("sigma_y", real=True, nonnegative=True)
        sigma_z_sym = sp.symbols("sigma_z", real=True, nonnegative=True)

        self.state_vars = [x, y, z]
        self.control_vars = []  # Standard Lorenz is autonomous

        # DRIFT (Deterministic Lorenz equations)
        self._f_sym = sp.Matrix([
            sigma_sym * (y - x),
            x * (rho_sym - z) - y,
            x * y - beta_sym * z
        ])

        self.parameters = {
            sigma_sym: sigma,
            rho_sym: rho,
            beta_sym: beta,
            sigma_x_sym: sigma_x,
            sigma_y_sym: sigma_y,
            sigma_z_sym: sigma_z,
        }
        self.order = 1

        # DIFFUSION (Additive noise on each equation)
        self.diffusion_expr = sp.Matrix([
            [sigma_x_sym, 0, 0],
            [0, sigma_y_sym, 0],
            [0, 0, sigma_z_sym]
        ])

        # Itô SDE
        self.sde_type = "ito"

        # Output: Full state
        self._h_sym = sp.Matrix([x, y, z])

    def setup_equilibria(self):
        """
        Set up equilibrium points (deterministic part).

        Note: All equilibria are unstable in chaotic regime (ρ=28).
        Trajectories never settle but wander on attractor.
        """
        rho = self.parameters[sp.symbols('rho')]
        beta = self.parameters[sp.symbols('beta')]

        # Origin (unstable for ρ > 1)
        stability_origin = "stable" if rho < 1 else "unstable"
        self.add_equilibrium(
            "origin",
            x_eq=np.array([0.0, 0.0, 0.0]),
            u_eq=np.array([]),
            verify=True,
            stability=stability_origin,
            notes=f"Origin (no convection). Stable if ρ<1, unstable if ρ>1. Current ρ={rho}."
        )

        # C± equilibria (exist for ρ > 1)
        if rho > 1:
            C_val = np.sqrt(beta * (rho - 1))
            
            # C+ 
            self.add_equilibrium(
                "C_plus",
                x_eq=np.array([C_val, C_val, rho - 1]),
                u_eq=np.array([]),
                verify=True,
                stability="unstable" if rho > 24.74 else "stable",
                notes=f"Steady convection C+. Stable if ρ<24.74, unstable (chaotic) if ρ>24.74."
            )
            
            # C-
            self.add_equilibrium(
                "C_minus",
                x_eq=np.array([-C_val, -C_val, rho - 1]),
                u_eq=np.array([]),
                verify=True,
                stability="unstable" if rho > 24.74 else "stable",
                notes=f"Steady convection C-. Mirror image of C+."
            )

        self.set_default_equilibrium("origin")

    def get_noise_intensities(self) -> dict:
        """
        Get noise intensity parameters.

        Returns
        -------
        dict
            {'sigma_x': ..., 'sigma_y': ..., 'sigma_z': ...}
        """
        return {
            'sigma_x': self.parameters[sp.symbols('sigma_x')],
            'sigma_y': self.parameters[sp.symbols('sigma_y')],
            'sigma_z': self.parameters[sp.symbols('sigma_z')],
        }

    def compute_attractor_size(self) -> float:
        """
        Estimate typical attractor size (order of magnitude).

        Returns
        -------
        float
            Approximate attractor diameter

        Notes
        -----
        Rough estimate based on C± equilibria distance from origin.

        Examples
        --------
        >>> lorenz = StochasticLorenz(rho=28.0, beta=8/3)
        >>> size = lorenz.compute_attractor_size()
        >>> print(f"Attractor size: ~{size:.1f}")
        """
        rho = self.parameters[sp.symbols('rho')]
        beta = self.parameters[sp.symbols('beta')]
        
        if rho <= 1:
            return 0.0  # No attractor (stable origin)
        
        # Distance to C± is rough measure
        C_val = np.sqrt(beta * (rho - 1))
        return 2 * C_val  # Diameter


# Convenience functions
def create_classical_lorenz_with_noise(
    noise_level: str = 'small',
) -> StochasticLorenz:
    """
    Create classic chaotic Lorenz (σ=10, ρ=28, β=8/3) with noise.

    Parameters
    ----------
    noise_level : str, default='small'
        'small', 'medium', or 'large'

    Returns
    -------
    StochasticLorenz

    Notes
    -----
    Uses standard chaotic parameters with varying noise.

    Examples
    --------
    >>> # Classic attractor with small perturbations
    >>> lorenz = create_classical_lorenz_with_noise('small')
    """
    noise_presets = {
        'small': {'sigma_x': 0.01, 'sigma_y': 0.01, 'sigma_z': 0.01},
        'medium': {'sigma_x': 0.1, 'sigma_y': 0.1, 'sigma_z': 0.1},
        'large': {'sigma_x': 1.0, 'sigma_y': 1.0, 'sigma_z': 1.0},
    }
    
    preset = noise_presets.get(noise_level, noise_presets['small'])
    
    return StochasticLorenz(
        sigma=10.0,
        rho=28.0,
        beta=8.0/3.0,
        **preset
    )


def create_atmospheric_model(
    turbulence_intensity: float = 0.5,
) -> StochasticLorenz:
    """
    Create Lorenz model for atmospheric dynamics with turbulence.

    Parameters
    ----------
    turbulence_intensity : float, default=0.5
        Unresolved turbulence noise level

    Returns
    -------
    StochasticLorenz

    Notes
    -----
    Models atmospheric convection with subgrid turbulence.

    Examples
    --------
    >>> # Moderate turbulence
    >>> atmos = create_atmospheric_model(turbulence_intensity=0.5)
    """
    return StochasticLorenz(
        sigma=10.0,
        rho=28.0,
        beta=8.0/3.0,
        sigma_x=turbulence_intensity,
        sigma_y=turbulence_intensity,
        sigma_z=turbulence_intensity,
    )