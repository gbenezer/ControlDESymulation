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
Ornstein-Uhlenbeck Process - Mean-Reverting Stochastic System
==============================================================

This module provides implementations of the Ornstein-Uhlenbeck (OU) process,
the fundamental mean-reverting stochastic differential equation. The OU process
is distinguished by:

- The canonical example of mean-reverting stochastic processes
- Foundation for many models in physics, finance, and biology
- The stochastic analog of linear damped dynamics
- Unique stationary Gaussian process with exponential correlation
- Explicit analytical solution in closed form

The OU process represents the continuous-time limit of discrete AR(1)
autoregressive processes and serves as the building block for more
complex mean-reverting models.

Mathematical Background
-----------------------
The Ornstein-Uhlenbeck process was introduced in 1930 by Leonard Ornstein
and George Uhlenbeck as a model for the velocity of a massive Brownian
particle undergoing friction. It solved a fundamental problem with
Einstein's original Brownian motion model: velocities must be bounded
due to finite energy.

**Physical Motivation:**
Consider a particle of mass m in a viscous fluid:
- Friction force: -γ·v (opposes motion)
- Random thermal force: η(t) (white noise)
- Newton's law: m·dv = -γ·v·dt + η(t)·dt

Dividing by m and setting α = γ/m, σ = √(2kᵦT/m):
    dv = -α·v·dt + σ·dW

This is the OU process. The particle's velocity fluctuates around zero
(thermal equilibrium) with characteristic time scale 1/α.

Mathematical Formulation
------------------------

**Standard Form (Centered):**
    dX = -α·X·dt + σ·dW

where:
    - X(t): State variable (velocity, rate, deviation from mean)
    - α > 0: Mean reversion rate (dimension: 1/time)
    - σ > 0: Volatility (noise intensity, dimension: state/√time)
    - W(t): Standard Wiener process
    - dW ~ N(0, dt)

**General Form (Non-Centered):**
    dX = α·(μ - X)·dt + σ·dW
    
Equivalent to: dX = -α·X·dt + α·μ·dt + σ·dW

This reverts to long-term mean μ rather than zero.

**With Control Input:**
    dX = (-α·X + u)·dt + σ·dW

The control u provides external forcing (can be time-varying or
state-dependent).

**Key Distinction from Brownian Motion:**
- Brownian Motion: dX = σ·dW (no drift, no reversion)
- OU Process: dX = -α·X·dt + σ·dW (drift opposes displacement)

The drift term -α·X creates a restoring force toward the origin,
fundamentally changing the long-term behavior.

Analytical Solution
-------------------
The OU process is one of the few SDEs with an explicit closed-form solution.

**Exact Solution:**
For X(0) = X₀ and constant control u:

    X(t) = X₀·e^(-α·t) + (u/α)·(1 - e^(-α·t)) + ∫₀ᵗ σ·e^(-α·(t-s))·dW(s)

The stochastic integral is Gaussian with zero mean.

**Interpretation:**
1. X₀·e^(-α·t): Initial condition exponentially decays
2. (u/α)·(1 - e^(-α·t)): Deterministic approach to u/α
3. Stochastic integral: Accumulated noise, filtered by reversion

**Moments (u = 0):**

Mean:
    E[X(t)] = X₀·e^(-α·t)

Exponentially decays to zero with time constant τ = 1/α.

Variance:
    Var[X(t)] = (σ²/2α)·(1 - e^(-2α·t))

Increases from 0 to stationary value σ²/(2α).

Covariance (s < t):
    Cov[X(s), X(t)] = (σ²/2α)·e^(-α·|t-s|)·(1 - e^(-2α·s))

For stationary process (s,t → ∞):
    Cov[X(s), X(t)] = (σ²/2α)·e^(-α·|t-s|)

**Asymptotic Behavior (t → ∞):**

Mean:
    E[X(∞)] = u/α

Variance (Stationary):
    Var[X(∞)] = σ²/(2α)

Standard Deviation:
    Std[X(∞)] = σ/√(2α)

Distribution:
    X(∞) ~ N(u/α, σ²/(2α))

The process reaches a **stationary distribution** - unique among
common SDEs

Key Properties
--------------

**1. Mean Reversion:**
The defining feature. Drift -α·X pulls process toward mean:
- Above mean (X > 0): Negative drift pushes down
- Below mean (X < 0): Positive drift pushes up
- At mean (X = 0): No drift (but still diffuses)

**2. Stationarity:**
Unique stationary Gaussian distribution:
    π(x) = N(0, σ²/(2α))

This is the **equilibrium distribution** - probability density
doesn't change with time for t → ∞.

**3. Ergodicity:**
Time averages equal ensemble averages:
    lim_{T→∞} (1/T)∫₀ᵀ f(X(t))dt = E[f(X)]

Can estimate statistics from single long trajectory.

**4. Markov Property:**
Future independent of past given present:
    P(X(t)|X(s), s ≤ u) = P(X(t)|X(u)) for t > u

**5. Gaussian Process:**
For any finite set of times {t₁, ..., tₙ}:
    (X(t₁), ..., X(tₙ)) ~ Multivariate Normal

Completely characterized by mean and covariance functions.

**6. Exponential Correlation:**
Autocorrelation function:
    ρ(τ) = Cov[X(t), X(t+τ)]/Var[X] = e^(-α·τ)

Correlation decays exponentially with lag τ.
- τ = 0: ρ = 1 (perfect correlation)
- τ = 1/α: ρ = e^(-1) ≈ 0.37
- τ = 5/α: ρ ≈ 0.007 (essentially uncorrelated)

**7. Additive Noise:**
Diffusion σ is constant (state-independent).
- Simplifies analysis and simulation
- Noise intensity same everywhere in state space
- Contrasts with multiplicative noise (GBM)

**8. Continuity:**
Sample paths are continuous but nowhere differentiable (like
Brownian motion).

**9. Gaussianity:**
X(t) is Gaussian for all t (inherited from Brownian motion).

Physical and Mathematical Interpretation
-----------------------------------------

**Mean Reversion Rate α:**
- Dimension: [1/time]
- Physical: Friction coefficient / mass
- Financial: Speed of adjustment to equilibrium
- Controls time scale of dynamics

**Time Constant τ = 1/α:**
- Average time to revert to mean
- After time τ: Deviation reduced by factor e ≈ 0.37
- After time 5τ: ~99% reverted

**Half-Life t₁/₂ = ln(2)/α ≈ 0.693/α:**
- Time to reduce deviation by 50%
- More intuitive than time constant
- Commonly used in finance

**Relaxation Time:**
Time to reach stationarity: t_relax ≈ 3-5τ

**Examples:**
- α = 0.1: Slow reversion, τ = 10s
- α = 1.0: Moderate reversion, τ = 1s
- α = 10.0: Fast reversion, τ = 0.1s

**Volatility σ:**
- Dimension: [state]/√[time]
- Physical: Noise intensity from thermal fluctuations
- Financial: Instantaneous standard deviation of changes
- Not the stationary standard deviation

**Stationary Standard Deviation:**
    σ_stat = σ/√(2α)

Balance between noise injection (σ) and dissipation (α):
- Large σ: More noise → larger fluctuations
- Large α: Faster reversion → smaller fluctuations

**Ratio α/σ:**
Signal-to-noise ratio:
- Large α/σ: Strong mean reversion, tight around mean
- Small α/σ: Weak mean reversion, large excursions

**Energy Interpretation:**
In physical systems:
    σ² = 2α·kᵦ·T/m

This is the **fluctuation-dissipation theorem**: noise and friction
are related through temperature.

Connection to Discrete AR(1)
-----------------------------
The OU process is the continuous-time limit of discrete AR(1):
    X[k+1] = φ·X[k] + ε[k]

where ε ~ N(0, σ_ε²).

**Correspondence:**
For small Δt:
    φ = e^(-α·Δt) ≈ 1 - α·Δt
    σ_ε² = σ²·Δt

Exact discretization:
    X[k+1] = e^(-α·Δt)·X[k] + √((σ²/2α)·(1 - e^(-2α·Δt)))·Z[k]

where Z ~ N(0,1).

Applications
------------

**1. Physics:**

**Langevin Equation (Particle Velocity):**
Velocity of massive particle in viscous fluid:
    dv = -γ·v·dt + √(2D)·dW

Where γ is friction, D is diffusion coefficient.

**Thermal Equilibrium:**
At equilibrium: E[v²] = kᵦT/m (equipartition theorem)
Implies: σ² = 2γ·kᵦT/m (fluctuation-dissipation)

**Applications:**
- Brownian motion (velocity, not position)
- Colloidal particle dynamics
- Molecular diffusion in traps
- Optical tweezers experiments
- Single-molecule biophysics

**2. Mathematical Finance:**

**Vasicek Interest Rate Model (1977):**
    dr = κ·(θ - r)·dt + σ·dW

Where:
- r(t): Short-term interest rate
- θ: Long-term mean rate
- κ: Mean reversion speed
- σ: Interest rate volatility

**Advantages:**
- Analytical bond prices
- Tractable option formulas
- Mean reversion captures rate dynamics

**Limitations:**
- Can become negative (unrealistic)
- Constant volatility (unrealistic)
- Led to extensions (CIR, Hull-White)

**Commodity Prices:**
Many commodities (oil, gas, metals) exhibit mean reversion:
- High prices → increased production → price drop
- Low prices → decreased supply → price rise

Model: dS = κ·(μ - ln(S))·S·dt + σ·S·dW
(Geometric OU in log-space)

**Pairs Trading:**
Spread between correlated assets often mean-reverts:
    Spread = Stock_A - β·Stock_B

**Credit Spreads:**
Corporate bond spreads over treasuries mean-revert.

**3. Neuroscience:**

**Neural Membrane Potential:**
Voltage across neuron membrane between spikes:
    dV = -(V - V_rest)/τ_m·dt + σ·dW

Where:
- V: Membrane potential
- V_rest: Resting potential
- τ_m: Membrane time constant (~10-20 ms)
- σ: Synaptic noise

**Leaky Integrate-and-Fire Model:**
OU process until threshold, then spike and reset.

**Synaptic Conductances:**
Time-varying conductances often modeled as OU.

**4. Biology:**

**Population Fluctuations:**
Population around carrying capacity:
    dN = -α·(N - K)·dt + σ·dW

Where K is carrying capacity.

**Gene Expression:**
Protein levels with stochastic production/degradation.

**Ecological Dynamics:**
Predator-prey systems with environmental noise.

**5. Climate Science:**

**Temperature Anomalies:**
Deviations from long-term average exhibit mean reversion.

**El Niño/La Niña:**
Ocean temperature fluctuations modeled as OU.

**6. Engineering:**

**Control Systems:**
Benchmark for stochastic optimal control:
- LQG (Linear-Quadratic-Gaussian) problem
- Kalman filtering applications
- Stochastic stability analysis

**Signal Processing:**
- Colored noise generation
- Time series modeling
- Autoregressive processes

**Communication Channels:**
Channel state fluctuations.

Numerical Simulation
--------------------

**Euler-Maruyama Discretization:**
    X[k+1] = X[k] + (-α·X[k] + u)·Δt + σ·√Δt·Z[k]
            = (1 - α·Δt)·X[k] + u·Δt + σ·√Δt·Z[k]

where Z ~ N(0,1).

**Convergence:**
- Weak order: O(Δt)
- Strong order: O(√Δt)

**Stability:**
Requires α·Δt < 1 for numerical stability.
Otherwise can overshoot and oscillate.

**Exact Discretization (Preferred):**
    X[k+1] = e^(-α·Δt)·X[k] + (u/α)·(1 - e^(-α·Δt)) 
             + √((σ²/2α)·(1 - e^(-2α·Δt)))·Z[k]

**Advantages:**
- Exact (no discretization error)
- Unconditionally stable
- Preserves stationary distribution
- Matches autocorrelation exactly

**Implementation:**
For additive noise, framework can use specialized solvers:
- No state dependence in diffusion
- Can precompute diffusion matrix
- More efficient integration

**Recommended Methods:**
- 'euler-maruyama': Simple, fast for small Δt
- 'milstein': Higher order (but same as Euler for additive noise)
- Exact scheme: Best choice when available

Statistical Analysis
--------------------

**Parameter Estimation:**
Given observations X₀, X₁, ..., X_n at times t₀, t₁, ..., t_n:

**Maximum Likelihood (Discrete Sampling):**
For equally spaced observations with Δt:

α̂ = -ln(∑X_i·X_{i-1} / ∑X_i²) / Δt

σ̂² = (2α̂/n)·∑(X_{i+1} - e^(-α̂·Δt)·X_i)² / (1 - e^(-2α̂·Δt))

**Method of Moments:**
Sample mean: m̄ → E[X] = 0 (check consistency)
Sample variance: s² → σ²/(2α)
Sample autocorrelation: ρ̂(Δt) → e^(-α·Δt)

From autocorrelation: α̂ = -ln(ρ̂(Δt)) / Δt

**Hypothesis Testing:**

1. **Mean Reversion Test:**
   - H₀: α = 0 (Brownian motion)
   - H₁: α > 0 (mean reversion)
   - Use unit root tests (ADF, PP)

2. **Stationarity Test:**
   - H₀: Process is stationary
   - Use KPSS test

3. **Gaussianity:**
   - Test residuals for normality
   - Jarque-Bera, Shapiro-Wilk

4. **Autocorrelation:**
   - Should decay exponentially
   - Plot log(ρ̂(τ)) vs τ (should be linear)

**Model Validation:**
- Compare theoretical and sample ACF
- Check residual independence
- Verify stationary variance: s² ≈ σ²/(2α)
- Test for parameter constancy over time

Comparison with Other Processes
--------------------------------

**vs. Brownian Motion:**
- BM: No drift, grows indefinitely (non-stationary)
- OU: Mean-reverting, bounded variance (stationary)
- BM is OU with α = 0

**vs. Geometric Brownian Motion:**
- GBM: Multiplicative noise, log-normal, for prices
- OU: Additive noise, Gaussian, for rates/deviations
- GBM non-stationary, OU stationary

**vs. Cox-Ingersoll-Ross (CIR):**
- CIR: dX = κ·(θ-X)·dt + σ·√X·dW (multiplicative noise)
- OU: dX = α·(μ-X)·dt + σ·dW (additive noise)
- CIR ensures X > 0, OU can be negative
- CIR for interest rates (positive), OU for spreads (can be negative)

**vs. Vasicek (Extended OU):**
- Vasicek: OU with non-zero mean μ
- Both are Gaussian, mean-reverting
- Vasicek = OU shifted and scaled

**vs. AR(1) Process:**
- AR(1): Discrete-time analog
- OU: Continuous-time limit of AR(1)
- Similar properties (stationarity, autocorrelation)

Extensions and Generalizations
-------------------------------

**1. Multivariate OU:**
    dX = A·X·dt + Σ·dW

Where A is stability matrix, Σ is diffusion matrix.
Applications: Multiple correlated rates, portfolio dynamics.

**2. Geometric OU:**
    dX = α·(μ - ln(X))·X·dt + σ·X·dW

Mean reversion in log-space, ensures X > 0.

**3. Non-Linear Mean Reversion:**
    dX = f(X)·dt + σ·dW

Where f(X) is non-linear restoring force.

**4. Time-Varying Parameters:**
    dX = -α(t)·X·dt + σ(t)·dW

Captures changing market conditions.

**5. Jump-OU:**
    dX = -α·X·dt + σ·dW + dJ

Adds discontinuous jumps to continuous process.

**6. Fractional OU:**
Replace Brownian motion with fractional Brownian motion (long memory).

Limitations
-----------

**1. Can Be Negative:**
OU process ranges over all real numbers. Problematic for:
- Interest rates (should be ≥ 0)
- Prices (should be > 0)
- Populations (should be ≥ 0)

**Solution:** Use CIR or geometric OU for positive quantities.

**2. Constant Volatility:**
σ independent of state. Reality:
- Interest rate volatility increases with rate level
- Spread volatility changes with spread

**Solution:** Use state-dependent diffusion.

**3. Linear Mean Reversion:**
Drift -α·X assumes linear restoring force. Reality:
- May have non-linear reversion
- Asymmetric reversion (faster from extremes)

**4. Gaussian Distribution:**
Real data often shows:
- Fat tails (jumps, regime changes)
- Skewness
- Time-varying moments

**5. Stationarity Assumption:**
Markets change:
- Mean μ drifts over time
- α, σ change with market regime

Common Pitfalls
---------------

1. **Confusing σ with Stationary Std:**
   - σ: Noise intensity parameter
   - σ/√(2α): Stationary standard deviation
   - σ_stat < σ (reduced by mean reversion)

2. **Ignoring α·Δt Constraint:**
   - Euler-Maruyama unstable if α·Δt > 1
   - Use smaller Δt or exact discretization

3. **Wrong Time Scale:**
   - Must match α units with time units
   - α in years⁻¹ requires time in years

4. **Forgetting Initial Transient:**
   - Takes time 3-5τ to reach stationarity
   - Don't use initial data for estimating stationary stats

5. **Assuming Independence:**
   - OU has exponential autocorrelation
   - Not white noise!
   - Observations correlated over time scale 1/α

6. **Negative Rate Problem:**
   - OU can go negative
   - Vasicek model predicts negative interest rates
   - Use CIR or other models for strictly positive quantities

**Physical Significance:**
The OU process resolved a fundamental problem: Einstein's Brownian
motion for position implies infinite kinetic energy. The OU process
for velocity respects finite energy (equipartition theorem) while
maintaining stochastic behavior.

**Mathematical Significance:**
First example showing that:
- Stationary processes need not be trivial
- Gaussian processes can have non-trivial correlation
- Mean reversion creates equilibrium distribution

Testing and Validation
-----------------------

**Unit Tests for OU Process:**

1. **Mean Reversion:**
   - Start at x₀ ≠ 0
   - Verify E[X(t)] → 0 as t → ∞
   - Check exponential decay rate

2. **Stationary Variance:**
   - Long simulation (t >> 1/α)
   - Sample variance should equal σ²/(2α)

3. **Autocorrelation:**
   - Compute ACF from simulation
   - Should match e^(-α·τ)

4. **Gaussianity:**
   - Histogram should be normal
   - Q-Q plot should be linear

5. **Exact Solution:**
   - Compare numerical to analytical moments

"""

import numpy as np
import sympy as sp

from src.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem


class OrnsteinUhlenbeck(ContinuousStochasticSystem):
    """
    Ornstein-Uhlenbeck process with mean reversion and additive noise.

    The fundamental mean-reverting stochastic process, combining deterministic
    relaxation toward an equilibrium with random fluctuations. This is the
    stochastic analog of a damped harmonic oscillator and the continuous-time
    limit of AR(1) processes.

    Stochastic Differential Equation
    ---------------------------------
    Centered form (mean = 0):
        dX = -α·X·dt + σ·dW

    With control:
        dX = (-α·X + u)·dt + σ·dW

    where:
        - X(t) ∈ ℝ: State (velocity, rate, deviation)
        - α > 0: Mean reversion rate (1/time)
        - σ > 0: Volatility (noise intensity, state/√time)
        - u ∈ ℝ: External forcing/control
        - W(t): Standard Wiener process

    Key Features
    ------------
    **Mean Reversion:**
    Drift term -α·X creates restoring force toward zero:
    - Above zero: Negative drift pushes down
    - Below zero: Positive drift pushes up
    - Strength proportional to displacement

    **Stationarity:**
    Unique stationary distribution:
        X(∞) ~ N(u/α, σ²/(2α))

    Unlike Brownian motion or GBM, variance bounded.

    **Additive Noise:**
    Diffusion σ constant (state-independent).
    Simplifies analysis and simulation.

    **Ergodicity:**
    Time averages equal ensemble averages.
    Can estimate statistics from single long trajectory.

    Mathematical Properties
    -----------------------
    **Exact Solution (u = 0, X(0) = X₀):**
        X(t) = X₀·e^(-α·t) + ∫₀ᵗ σ·e^(-α·(t-s))·dW(s)

    **Moments:**
    Mean:
        E[X(t)] = X₀·e^(-α·t) + (u/α)·(1 - e^(-α·t))

    Variance:
        Var[X(t)] = (σ²/2α)·(1 - e^(-2α·t))

    **Asymptotic (t → ∞):**
    Mean: E[X(∞)] = u/α
    Variance: Var[X(∞)] = σ²/(2α)
    Distribution: N(u/α, σ²/(2α))

    **Autocorrelation:**
    For stationary process:
        Cov[X(t), X(t+τ)] = (σ²/2α)·e^(-α·τ)

    Exponential decay with rate α.

    Physical Interpretation
    -----------------------
    **Mean Reversion Rate α:**
    - Controls speed of return to equilibrium
    - Units: [1/time]
    - Time constant: τ = 1/α
    - Half-life: t₁/₂ = ln(2)/α ≈ 0.693/α

    **Examples:**
    - α = 0.1: Slow reversion, τ = 10s
    - α = 1.0: Moderate, τ = 1s  
    - α = 10.0: Fast, τ = 0.1s

    **Volatility σ:**
    - Instantaneous noise intensity
    - Units: [state]/√[time]
    - Stationary std: σ/√(2α)

    **Ratio σ/α:**
    Effective noise level:
    - Large: Weak reversion, large fluctuations
    - Small: Strong reversion, tight around mean

    **Stationary Standard Deviation:**
        σ_stat = σ/√(2α)

    Balance between noise injection (σ) and dissipation (α).

    State Space
    -----------
    State: x ∈ ℝ (unbounded)
        - Can take any real value
        - Equilibrium at x = u/α
        - Fluctuates around equilibrium

    Control: u ∈ ℝ (optional)
        - Shifts equilibrium to u/α
        - Examples: External force, policy intervention

    Parameters
    ----------
    alpha : float, default=1.0
        Mean reversion rate (must be positive for stability)
        - Larger α: Faster reversion, smaller steady-state variance
        - Time constant τ = 1/α
        - Typical range: 0.1 to 10.0

    sigma : float, default=1.0
        Volatility (must be positive)
        - Controls noise intensity
        - Stationary std: σ/√(2α)
        - Typical: 0.1 to 2.0

    Stochastic Properties
    ---------------------
    - Noise Type: ADDITIVE
    - Diffusion: g(x) = σ (constant, state-independent)
    - SDE Type: Itô (standard)
    - Noise Dimension: nw = 1
    - Stationary: Yes (unique equilibrium distribution)
    - Ergodic: Yes (time averages = ensemble averages)

    Applications
    ------------
    **1. Physics:**
    - Langevin equation (particle velocity in fluid)
    - Thermal equilibrium (velocity distribution)
    - Optical tweezers (trapped particle)
    - Molecular dynamics

    **2. Finance:**
    - Vasicek interest rate model
    - Commodity price spreads
    - Pairs trading (spread between correlated assets)
    - Credit spreads

    **3. Neuroscience:**
    - Neural membrane potential between spikes
    - Leaky integrate-and-fire neurons
    - Synaptic conductances

    **4. Biology:**
    - Population fluctuations around carrying capacity
    - Gene expression dynamics
    - Ecological systems with noise

    **5. Control & Signal Processing:**
    - Colored noise generation
    - Autoregressive processes (continuous AR(1))
    - Kalman filtering benchmark

    Numerical Simulation
    --------------------
    **Euler-Maruyama:**
        X[k+1] = (1 - α·Δt)·X[k] + u·Δt + σ·√Δt·Z[k]

    Requires α·Δt < 1 for stability.

    **Exact Discretization (Preferred):**
        X[k+1] = e^(-α·Δt)·X[k] + (u/α)·(1-e^(-α·Δt)) 
                 + σ_eff·Z[k]

    where σ_eff = √((σ²/2α)·(1-e^(-2α·Δt)))

    **Advantages:**
    - Exact (no discretization error)
    - Unconditionally stable
    - Preserves stationary distribution

    Statistical Analysis
    --------------------
    **Parameter Estimation:**
    From discrete observations, estimate:
        α̂ = -ln(autocorr(Δt)) / Δt
        σ̂ = sample_std · √(2α̂)

    **Model Validation:**
    - Check exponential autocorrelation
    - Verify Gaussian residuals
    - Test stationarity (KPSS)
    - Test mean reversion (ADF unit root)

    Comparison with Other Processes
    --------------------------------
    **vs. Brownian Motion:**
    - BM: No reversion (non-stationary)
    - OU: Mean-reverting (stationary)

    **vs. Geometric Brownian Motion:**
    - GBM: Multiplicative noise, for prices
    - OU: Additive noise, for rates/deviations

    **vs. CIR Process:**
    - CIR: Multiplicative noise √X (stays positive)
    - OU: Additive noise (can be negative)

    Limitations
    -----------
    - Can be negative (problem for rates/prices)
    - Constant volatility (unrealistic for some applications)
    - Linear mean reversion (may be non-linear in reality)
    - Gaussian (real data may have fat tails)

    **Solutions:**
    - CIR for positive quantities
    - State-dependent diffusion
    - Non-linear drift functions
    - Jump extensions

    See Also
    --------
    BrownianMotion : No mean reversion (α=0 limit)
    GeometricBrownianMotion : Multiplicative noise
    CoxIngersollRoss : Mean-reverting with √X diffusion
    """

    def define_system(self, alpha: float = 1.0, sigma: float = 1.0):
        """
        Define Ornstein-Uhlenbeck process dynamics.

        Sets up the stochastic differential equation:
            dX = (-α·X + u)·dt + σ·dW

        with mean reversion and additive noise.

        Parameters
        ----------
        alpha : float, default=1.0
            Mean reversion rate (should be positive)
            - α > 0: Stable (mean-reverting)
            - α = 0: Brownian motion (no reversion)
            - α < 0: Unstable (explosive)

        sigma : float, default=1.0
            Volatility (must be positive)
            - Controls noise intensity
            - Stationary std: σ/√(2α)

        Raises
        ------
        ValueError
            If sigma ≤ 0
        UserWarning
            If alpha ≤ 0 (unstable/non-reverting)

        Notes
        -----
        **Stability Condition:**
        Require α > 0 for mean reversion and stationarity.
        - α > 0: Process stable, reverts to mean
        - α = 0: Becomes Brownian motion
        - α < 0: Unstable, diverges exponentially

        **Time Scales:**
        - Time constant: τ = 1/α
        - Half-life: t₁/₂ = ln(2)/α ≈ 0.693/α
        - Settling time: t_settle ≈ 5/α (1% of initial deviation)

        After time τ: Deviation reduced by factor e ≈ 0.368
        After time 5τ: Deviation reduced by factor e⁻⁵ ≈ 0.007 (~99% reverted)

        **Stationary Statistics:**
        For u = 0:
        - Mean: E[X(∞)] = 0
        - Variance: Var[X(∞)] = σ²/(2α)
        - Std: σ/√(2α)

        For u ≠ 0:
        - Mean: E[X(∞)] = u/α
        - Variance: σ²/(2α) (same)

        **Parameter Selection:**
        - Fast reversion: α = 5-10 (τ = 0.1-0.2s)
        - Moderate: α = 1-2 (τ = 0.5-1s)
        - Slow: α = 0.1-0.5 (τ = 2-10s)

        **Noise Level:**
        Effective noise (stationary std): σ/√(2α)
        - To achieve desired std s: set σ = s·√(2α)

        **Fluctuation-Dissipation:**
        In physical systems at temperature T:
            σ² = 2α·kᵦ·T/m
        Balance between thermal noise and friction.
        """
        # Validate parameters
        if alpha <= 0:
            import warnings
            warnings.warn(
                f"alpha={alpha} ≤ 0 leads to unstable/non-reverting process. "
                f"Use alpha > 0 for mean reversion. "
                f"alpha = 0 gives Brownian motion, alpha < 0 is explosive.",
                UserWarning,
            )

        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        # Define symbolic variables
        x = sp.symbols("x", real=True)
        u = sp.symbols("u", real=True)

        # Define symbolic parameters
        alpha_sym = sp.symbols("alpha", positive=True)
        sigma_sym = sp.symbols("sigma", positive=True)

        # System definition
        self.state_vars = [x]
        self.control_vars = [u]

        # Drift: f(x, u) = -α·x + u
        # Mean-reverting drift toward u/α
        self._f_sym = sp.Matrix([[-alpha_sym * x + u]])

        self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
        self.order = 1

        # Diffusion: g(x, u) = σ (constant - additive noise)
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = "ito"

    def get_stationary_std(self) -> float:
        """
        Compute theoretical stationary standard deviation.

        For OU process, stationary distribution is:
            X(∞) ~ N(u/α, σ²/(2α))

        Returns
        -------
        float
            Stationary standard deviation: σ/√(2α)

        Notes
        -----
        **Interpretation:**
        This is the long-term standard deviation around the mean,
        reached after transient dies out (t >> 1/α).

        **Relationship to Parameters:**
        - Increases with σ (more noise)
        - Decreases with α (stronger reversion)
        - Ratio σ/α sets scale

        **Comparison with σ:**
        - σ: Instantaneous noise intensity
        - σ/√(2α): Accumulated noise effect
        - Always: σ_stat < σ (for α > 0.5)

        **Design:**
        To achieve target std s:
            σ = s · √(2α)
        """
        # Extract parameter values
        alpha = None
        sigma = None

        for key, val in self.parameters.items():
            if str(key) == "alpha":
                alpha = val
            elif str(key) == "sigma":
                sigma = val

        return sigma / np.sqrt(2.0 * alpha)

    def get_time_constant(self) -> float:
        """
        Get mean reversion time constant τ = 1/α.

        The time constant is the characteristic time scale for
        the process to revert to its mean. After time τ, the
        deviation from mean is reduced by factor e ≈ 0.368.

        Returns
        -------
        float
            Time constant [time units]

        Notes
        -----
        **Physical Meaning:**
        - Time to reduce deviation by ~63%
        - Reciprocal of mean reversion rate
        - Sets time scale of dynamics

        **Related Quantities:**
        - Half-life: t₁/₂ = ln(2)·τ ≈ 0.693·τ
        - 99% settling: t_99 ≈ 5·τ
        - 1% settling: t_01 ≈ 0.01·τ

        **Rule of Thumb:**
        Process reaches stationarity after 3-5 time constants.
        - t < τ: Still in transient
        - t ≈ 5τ: Essentially stationary

        **Examples:**
        - τ = 0.1s: Fast dynamics (α = 10)
        - τ = 1.0s: Moderate (α = 1)
        - τ = 10s: Slow (α = 0.1)
        """
        for key, val in self.parameters.items():
            if str(key) == "alpha":
                return 1.0 / val

        raise RuntimeError("alpha parameter not found")

    def get_half_life(self) -> float:
        """
        Get half-life t₁/₂ = ln(2)/α ≈ 0.693/α.

        Time for deviation from mean to reduce by 50%.
        More intuitive than time constant for some applications.

        Returns
        -------
        float
            Half-life [time units]
        """
        return np.log(2.0) / self.get_time_constant()


# ============================================================================
# Convenience Factory Functions
# ============================================================================


def create_ou_process(
    time_constant: float = 1.0, 
    volatility: float = 1.0
) -> OrnsteinUhlenbeck:
    """
    Create OU process with specified time constant and volatility.

    More intuitive parameterization using time constant τ instead
    of mean reversion rate α = 1/τ.

    Parameters
    ----------
    time_constant : float, default=1.0
        Time constant τ [time units]
        - Mean reversion rate: α = 1/τ
        - Typical: 0.1 to 10 seconds
        
    volatility : float, default=1.0
        Noise intensity σ
        - Stationary std: σ/√(2/τ)

    Returns
    -------
    OrnsteinUhlenbeck
        OU process with α = 1/τ, σ = volatility

    Notes
    -----
    **Time Constant Interpretation:**
    - τ = 0.1s: Fast reversion (settles in ~0.5s)
    - τ = 1.0s: Moderate (settles in ~5s)
    - τ = 10s: Slow (settles in ~50s)

    **Design Pattern:**
    Choose τ based on desired response time, then
    choose σ based on desired fluctuation magnitude.
    """
    alpha = 1.0 / time_constant
    return OrnsteinUhlenbeck(alpha=alpha, sigma=volatility)


def create_vasicek_model(
    mean_reversion: float = 0.5,
    long_term_rate: float = 0.05,
    volatility: float = 0.01,
) -> OrnsteinUhlenbeck:
    """
    Create Vasicek interest rate model.

    The Vasicek model (1977) is an OU process for short-term
    interest rates with mean reversion to a long-term level.

    Mathematical Form:
        dr = κ·(θ - r)·dt + σ·dW

    Equivalent to:
        dr = -κ·r·dt + κ·θ·dt + σ·dW
        
    which is OU with control u = κ·θ.

    Parameters
    ----------
    mean_reversion : float, default=0.5
        Mean reversion speed κ (1/year)
        - Typical: 0.1 to 2.0
        - Higher: Faster reversion to long-term rate
        
    long_term_rate : float, default=0.05
        Long-term mean interest rate θ
        - As decimal: 0.05 = 5% annual rate
        - Typical: 0.02 to 0.08
        
    volatility : float, default=0.01
        Interest rate volatility σ (1/√year)
        - Typical: 0.005 to 0.02 (0.5% to 2%)

    Returns
    -------
    OrnsteinUhlenbeck
        Vasicek model (OU process with parameters κ, σ)

    Notes
    -----
    **Historical Context:**
    Introduced by Oldřich Vašíček in 1977, this was the first
    equilibrium model of the term structure. Revolutionary for:
    - Analytical bond pricing formulas
    - Mean reversion (rates don't wander arbitrarily)
    - Stochastic calculus in finance

    **Advantages:**
    - Tractable analytical formulas
    - Mean reversion captures rate behavior
    - Simple to estimate from data

    **Limitations:**
    - Can produce negative rates (problematic)
    - Constant volatility (unrealistic)
    - Normal distribution (fat tails in reality)

    **Modern Usage:**
    Still used for:
    - Teaching and intuition
    - Baseline comparisons
    - Simple scenarios

    Superseded by:
    - CIR model (positive rates)
    - Hull-White (time-varying parameters)
    - LIBOR market models

    **Implementation Note:**
    This returns centered OU process. To match Vasicek exactly
    with long-term mean θ, set control:
        u = κ·θ

    Then equilibrium rate is u/κ = θ.

    **Stationary Distribution:**
        r(∞) ~ N(θ, σ²/(2κ))

    **Bond Pricing:**
    Zero-coupon bond price:
        P(t,T) = A(t,T)·exp(-B(t,T)·r(t))
    
    where B(t,T) and A(t,T) have closed-form expressions.
    See Also
    --------
    OrnsteinUhlenbeck : Base class
    CoxIngersollRoss : Positive-rates alternative
    """
    # Note: Returns centered OU. User must apply control u = κ·θ for correct mean.
    return OrnsteinUhlenbeck(alpha=mean_reversion, sigma=volatility)