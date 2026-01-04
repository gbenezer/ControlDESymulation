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
Discrete Random Walk - Non-Stationary Stochastic Process
=========================================================

This module provides implementations of the discrete random walk, one of the
most fundamental stochastic processes in probability theory. The random walk
is distinguished by:

- The canonical example of a non-stationary process (unit root)
- The discrete-time analog of Brownian motion
- Foundation for understanding martingales and diffusion limits
- The AR(1) process with φ = 1 (critical boundary)
- The cumulative sum of white noise innovations

The random walk is the limiting case between stationary (|φ| < 1) and
explosive (|φ| > 1) processes, exhibiting unique properties that make
it essential for understanding non-stationary time series.

Mathematical Background
-----------------------
**Physical Interpretation:**
Random walks naturally arise in:
- Particle diffusion (molecule in fluid)
- Stock prices (cumulative returns)
- Polymer chains (end-to-end distance)
- Gambler's fortune (cumulative wins/losses)
- Search algorithms (random search)
- Genetic drift (allele frequency changes)

Mathematical Definition
-----------------------

**Standard Form:**
    X[k+1] = X[k] + w[k]

where w[k] ~ N(0, σ²) are iid innovations (white noise).

**With Drift:**
    X[k+1] = X[k] + μ + w[k]

Adds deterministic trend μ per step.

**Equivalent Cumulative Form:**
    X[k] = X[0] + Σ_{j=0}^{k-1} w[j]

This shows random walk as cumulative sum of white noise.

**Connection to AR(1):**
Random walk is AR(1) with φ = 1:
    X[k+1] = 1·X[k] + w[k]

This is the "unit root" case - boundary between stationary and explosive.

**Relationship to Brownian Motion:**
Scaling limit (Donsker's theorem):
    (1/√n)·X[⌊nt⌋] → σ·W(t) as n → ∞

where W(t) is standard Brownian motion. Random walk converges to
continuous Brownian motion in distribution.

Key Properties
--------------

**1. Non-Stationarity:**
Unlike stationary processes, moments grow with time.

Mean: E[X[k]] = X[0] + k·μ (grows linearly with k)
Variance: Var[X[k]] = k·σ² (grows linearly with k)

Both unbounded as k → ∞.

**2. Unit Root:**
Characteristic equation: 1 - z = 0 has root at z = 1.
This "unit root" makes the process non-stationary.

**Stationarity Test:**
Dickey-Fuller test: H₀: unit root vs H₁: stationary

**3. Perfect Memory:**
All past shocks permanently affect current value:
    X[k] = X[0] + w[0] + w[1] + ... + w[k-1]

Shocks never die out (contrast with AR(1) where shocks decay).

**4. Increments are Stationary:**
While X[k] non-stationary, differences are stationary:
    ΔX[k] = X[k] - X[k-1] = w[k] ~ N(0, σ²)

This is called "integrated of order 1" or I(1).

**5. Martingale Property (μ = 0):**
For zero drift:
    E[X[k+1] | X[k], X[k-1], ...] = X[k]

Fair game - best prediction of future is current value.

**6. Recurrence (1D and 2D):**
Pólya's theorem:
- 1D: Returns to any point infinitely often (recurrent)
- 2D: Returns to any point infinitely often (recurrent)
- 3D+: Escapes to infinity (transient)

**7. No Long-Term Mean:**
Unlike stationary processes, random walk has no equilibrium distribution.
Wanders arbitrarily far from origin.

**8. Unbounded Variance:**
Variance grows linearly: Var[X[k]] = k·σ²

Standard deviation grows as √k (square root of time).

Mathematical Properties
-----------------------

**Exact Solution:**
For X[0] = X₀ and drift μ:
    X[k] = X₀ + k·μ + Σ_{j=0}^{k-1} w[j]

**Moments (μ = 0):**
Mean:
    E[X[k]] = X₀

Variance:
    Var[X[k]] = k·σ²

Standard Deviation:
    Std[X[k]] = σ·√k

**Distribution:**
    X[k] ~ N(X₀ + k·μ, k·σ²)

Normal with growing variance.

**Autocorrelation:**
For k ≥ j:
    Cov[X[k], X[j]] = j·σ²
    Corr[X[k], X[j]] = √(j/k)

Correlation decreases slowly (does not decay to zero).

**First Passage Time:**
Time to first reach level a > X₀:
- Mean: E[τ_a] = (a - X₀)/μ (with drift μ > 0)
- Variance: Var[τ_a] = σ²·(a - X₀)/μ³

Without drift (μ = 0): E[τ_a] = ∞ (infinite expected time!)

Physical and Statistical Interpretation
----------------------------------------

**Step Size σ:**
- Dimension: [state]
- Interpretation: Standard deviation of each step
- Typical range: Application-dependent

**RMS Displacement:**
After k steps: RMS = σ·√k

This √k scaling is characteristic of diffusion processes.

**Diffusion Coefficient:**
In continuous limit: D = σ²/(2·Δt)

where Δt is time step.

**Einstein Relation:**
For particle with friction γ at temperature T:
    σ² = 2·D·Δt = (2·k_B·T/γ)·Δt

Connects noise to thermodynamics.

**Drift μ:**
- Dimension: [state]/[step]
- Interpretation: Expected change per step
- Creates deterministic trend

**Hurst Exponent:**
For random walk: H = 0.5

Displacement scales as t^H. This is:
- Subdiffusion: H < 0.5 (slower than random walk)
- Normal diffusion: H = 0.5 (random walk)
- Superdiffusion: H > 0.5 (faster than random walk)

Comparison with Other Processes
--------------------------------

**vs. Stationary AR(1):**
- AR(1): φ < 1, variance bounded
- Random walk: φ = 1, variance grows
- Random walk is critical boundary

**vs. White Noise:**
- White noise: Independent samples, bounded variance
- Random walk: Cumulative sum, growing variance
- Random walk = Integrated white noise

**vs. Brownian Motion:**
- Brownian motion: Continuous time
- Random walk: Discrete time
- Random walk → Brownian motion in limit

**vs. AR(1) with φ > 1:**
- Random walk: Variance grows linearly
- Explosive AR(1): Variance grows exponentially
- Random walk is stable boundary

**vs. Trend-Stationary:**
- Trend-stationary: X[k] = α + β·k + stationary
- Random walk: X[k] = X[0] + Σ w[j]
- Shocks permanent in RW, transient in trend-stationary

Applications
------------

**1. Financial Economics:**

**Efficient Market Hypothesis:**
If markets efficient, prices should follow random walk:
    P[k] = P[k-1] + ε[k]

Returns (ε[k]) unpredictable from past.

**Stock Prices:**
Log-prices often modeled as random walk (with drift):
    ln(P[k]) = ln(P[k-1]) + μ + σ·w[k]

**Exchange Rates:**
Currency rates may exhibit random walk behavior.

**2. Physics:**

**Particle Diffusion:**
Position of particle undergoing Brownian motion.

**Polymer Chains:**
End-to-end distance of random coil polymer.

**2D Random Walk:**
Models diffusion on surfaces, membranes.

**3. Biology:**

**Genetic Drift:**
Allele frequency changes in finite populations.

**Foraging:**
Animal search patterns (Lévy flights generalization).

**Epidemic Spread:**
Spatial diffusion of disease.

**4. Computer Science:**

**Random Algorithms:**
Random search, randomized optimization.

**Page Rank:**
Random walk on web graph.

**Monte Carlo:**
Random walk in state space.

**5. Econometrics:**

**Unit Root Testing:**
Distinguish random walk from stationary process.

**Cointegration:**
Multiple random walks with stationary linear combination.

**Spurious Regression:**
Regressing one random walk on another gives spurious correlation.

**6. Gambling:**

**Gambler's Ruin:**
Cumulative winnings follow random walk.

**Betting Systems:**
Can't beat fair game (martingale property).

Numerical Simulation
--------------------

**Direct Generation:**
    X[0] = X₀
    for k in range(N):
        X[k+1] = X[k] + σ·randn()

**Vectorized (Efficient):**
    w = σ·np.random.randn(N)
    X = X₀ + np.cumsum(w)

Uses cumulative sum of white noise.

**With Drift:**
    X = X₀ + μ·np.arange(N) + np.cumsum(σ·np.random.randn(N))

**Typical Behavior:**
- Starts at X₀
- Wanders increasingly far from origin
- No tendency to return
- Can reach arbitrarily large/small values

**Visualization:**
Plot looks like "wandering" with no equilibrium.
Standard deviation envelope: X₀ ± σ·√k (grows with time).

Statistical Analysis
--------------------

**Parameter Estimation:**
Given observations X[0], X[1], ..., X[N]:

Estimate σ² from differences:
    ΔX[k] = X[k] - X[k-1]
    σ̂² = (1/(N-1))·Σ (ΔX[k] - μ̂)²

where μ̂ = (X[N] - X[0])/N.

**Unit Root Tests:**

1. **Dickey-Fuller Test:**
   Regress ΔX[k] on X[k-1]:
   - H₀: Random walk (unit root)
   - H₁: Stationary (|φ| < 1)

2. **Augmented Dickey-Fuller (ADF):**
   Adds lagged differences to account for serial correlation.

3. **Phillips-Perron:**
   Non-parametric correction for serial correlation.

**Spurious Regression Problem:**
Regressing one random walk on another can give:
- High R² (often > 0.5)
- Significant t-statistics
- But relationship is spurious!

Always test for unit roots before regression.

**Differencing:**
Transform to stationarity:
    ΔX[k] = X[k] - X[k-1] = w[k]

Differences are stationary white noise (if σ constant).

Extensions and Generalizations
-------------------------------

**1. Random Walk with Drift:**
    X[k+1] = X[k] + μ + σ·w[k]

Adds deterministic trend.

**2. Random Walk with Barrier:**
- Reflecting: Bounces off boundaries
- Absorbing: Stops at boundaries
- Applications: option pricing, gambler's ruin

**3. Biased Random Walk:**
    P(up) = p ≠ 0.5
    P(down) = 1 - p

Asymmetric steps.

**4. Correlated Random Walk:**
Direction depends on previous direction.

**5. Continuous-Time Random Walk:**
Lévy flights (heavy-tailed steps).

**6. Fractional Random Walk:**
Long-range dependence (fractional integration).

**7. Multivariate:**
Vector random walk (multiple dimensions).

**8. Non-Gaussian:**
Steps from non-Gaussian distribution (fat tails).

Limitations
-----------

**1. Non-Stationarity:**
Standard statistical methods assume stationarity.
Need special treatment (differencing, cointegration).

**2. Unbounded Variance:**
Grows without limit. Real systems often have bounds.

**3. No Mean Reversion:**
Unlike real prices which may revert to fundamentals.

**4. Constant Step Size:**
Real volatility varies over time (GARCH effects).

**5. Independence:**
Real data may have dependence in differences.

**6. Normality:**
Real innovations may have fat tails.

Common Pitfalls
---------------

1. **Treating as Stationary:**
   - Random walk is NOT stationary
   - Standard inference breaks down
   - Must difference or use unit root methods

2. **Spurious Regression:**
   - Two independent random walks can appear correlated
   - Always test for unit roots first

3. **Confusing with Trend:**
   - Random walk: Stochastic trend (shocks permanent)
   - Deterministic trend: X[k] = α + β·k + stationary
   - Different implications for forecasting

4. **Ignoring Drift:**
   - Zero drift: Wanders both directions
   - Positive drift: Trend upward
   - Changes long-term behavior

5. **Forecasting:**
   - Best forecast: Current value (if no drift)
   - Forecast uncertainty grows: σ²·h for h steps ahead
   - Not mean-reverting like AR(1)

6. **Sample Size:**
   - Need long samples to distinguish from near-unit-root AR(1)
   - Power of unit root tests low in small samples

Testing and Validation
-----------------------

**Diagnostic Tests:**

1. **Unit Root Test:**
   - ADF test on levels
   - Should not reject unit root

2. **Stationarity of Differences:**
   - Test that ΔX[k] is stationary
   - Should reject unit root for differences

3. **Variance Growth:**
   - Check Var[X[k]] ∝ k
   - Plot variance vs time (should be linear)

4. **ACF of Levels:**
   - Should decay very slowly
   - Not drop to zero

5. **ACF of Differences:**
   - Should be white noise (zero except lag 0)

**Monte Carlo Validation:**
Generate random walks, verify:
- Mean constant (if no drift)
- Variance linear in k
- Returns to origin infinitely often (1D)
- ADF test fails to reject unit root
"""

import numpy as np
import sympy as sp

from src.systems.base.core.discrete_stochastic_system import DiscreteStochasticSystem


class DiscreteRandomWalk(DiscreteStochasticSystem):
    """
    Random walk - non-stationary process with unit root.

    The fundamental non-stationary process, representing cumulative sum
    of white noise. This is the discrete-time analog of Brownian motion
    and the AR(1) process with φ = 1 (unit root).

    Difference Equation
    -------------------
    Standard form (no drift):
        X[k+1] = X[k] + σ·w[k]

    With drift:
        X[k+1] = X[k] + μ + σ·w[k]

    where w[k] ~ N(0,1) are iid.

    **Cumulative Form:**
        X[k] = X[0] + μ·k + Σ_{j=0}^{k-1} σ·w[j]

    Shows random walk as cumulative sum of innovations.

    Key Features
    ------------
    **Unit Root:**
    AR(1) with φ = 1 (critical boundary):
    - Stationary: |φ| < 1
    - Random walk: φ = 1
    - Explosive: |φ| > 1

    **Non-Stationarity:**
    Variance grows with time:
        Var[X[k]] = k·σ²

    Unlike stationary processes with bounded variance.

    **Perfect Memory:**
    All past shocks permanently affect current value.
    Shocks never decay (contrast with AR(1)).

    **Martingale (μ=0):**
    Best prediction of future is present:
        E[X[k+1] | past] = X[k]

    "Fair game" property.

    **Recurrence (1D):**
    Returns to any value infinitely often (Pólya's theorem).
    But transient in 3D and higher.

    Mathematical Properties
    -----------------------
    **Moments (μ = 0):**
    Mean: E[X[k]] = X[0]
    Variance: Var[X[k]] = k·σ²
    Std Dev: σ·√k

    **Distribution:**
        X[k] ~ N(X[0] + μ·k, k·σ²)

    **Autocorrelation:**
    For k ≥ j:
        Corr[X[k], X[j]] = √(j/k)

    Decays slowly (does not go to zero).

    **Stationary Differences:**
        ΔX[k] = X[k] - X[k-1] = μ + σ·w[k]

    Differences are stationary (integrated of order 1).

    Physical Interpretation
    -----------------------
    **Step Size σ:**
    - Standard deviation of each step
    - Units: [state]
    - RMS displacement after k steps: σ·√k

    **Drift μ:**
    - Expected change per step
    - Units: [state]/[step]
    - Creates deterministic trend

    **Scaling:**
    Displacement scales as √k (characteristic of diffusion).

    **Diffusion Coefficient:**
        D = σ²/(2·Δt)

    State Space
    -----------
    State: x ∈ ℝ (unbounded)
        - No equilibrium (non-stationary)
        - Wanders arbitrarily far
        - No tendency to return

    Control: None (autonomous)
        - Pure stochastic dynamics
        - No external forcing

    Parameters
    ----------
    sigma : float, default=1.0
        Step size (innovation std dev)
        - Must be positive
        - Controls diffusion rate
        - Typical: 0.1 to 10.0

    mu : float, default=0.0
        Drift (trend per step)
        - Can be positive, negative, or zero
        - μ = 0: Symmetric random walk
        - μ ≠ 0: Biased random walk

    dt : float, default=1.0
        Time step [units]
        - Required for discrete system
        - Sets time scale

    Stochastic Properties
    ---------------------
    - Type: Unit root (φ = 1)
    - Stationary: No (non-stationary)
    - Martingale: Yes (if μ = 0)
    - Memory: Perfect (all shocks permanent)
    - Variance: Unbounded (grows linearly)

    Applications
    ------------
    **1. Financial Economics:**
    - Stock prices (efficient markets)
    - Exchange rates
    - Interest rates (sometimes)
    - Null hypothesis for predictability tests

    **2. Physics:**
    - Brownian motion (discrete time)
    - Particle diffusion
    - Polymer chains

    **3. Econometrics:**
    - Unit root testing
    - Cointegration analysis
    - Non-stationary time series

    **4. Biology:**
    - Genetic drift
    - Population dispersal
    - Foraging behavior

    **5. Computer Science:**
    - Random algorithms
    - Page rank
    - Monte Carlo methods

    Numerical Simulation
    --------------------
    **Efficient Generation:**
        w = σ·np.random.randn(N)
        X = X₀ + μ·np.arange(N) + np.cumsum(w)

    Uses cumulative sum for efficiency.

    **Typical Behavior:**
    - Starts at X₀
    - Wanders with no equilibrium
    - Standard deviation: σ·√k
    - Never settles down

    Statistical Analysis
    --------------------
    **Unit Root Testing:**
    - Dickey-Fuller test
    - Augmented Dickey-Fuller (ADF)
    - Phillips-Perron

    **Differencing:**
    Transform to stationarity:
        ΔX[k] = X[k] - X[k-1] ~ N(μ, σ²)

    **Parameter Estimation:**
    From differences:
        μ̂ = mean(ΔX)
        σ̂² = var(ΔX)

    Comparison with Other Processes
    --------------------------------
    **vs. AR(1):**
    - AR(1): Stationary (|φ| < 1)
    - Random walk: Unit root (φ = 1)
    - RW is critical boundary

    **vs. White Noise:**
    - White noise: Independent, stationary
    - Random walk: Cumulative sum of white noise

    **vs. Brownian Motion:**
    - Brownian motion: Continuous time
    - Random walk: Discrete time
    - RW → BM in scaling limit

    **vs. Trend-Stationary:**
    - Trend-stationary: Detrend → stationary
    - Random walk: Difference → stationary
    - Different implications

    Limitations
    -----------
    - Non-stationary (need special methods)
    - Unbounded variance (unrealistic)
    - No mean reversion (prices may revert)
    - Constant volatility (often time-varying)
    - Normal innovations (may have fat tails)

    See Also
    --------
    DiscreteWhiteNoise : Differences of random walk
    DiscreteAR1 : Stationary version (|φ| < 1)
    BrownianMotion : Continuous-time analog
    """

    def define_system(
        self, 
        sigma: float = 1.0, 
        mu: float = 0.0,
        dt: float = 1.0
    ):
        """
        Define random walk dynamics.

        Sets up the difference equation:
            X[k+1] = X[k] + μ + σ·w[k]

        Parameters
        ----------
        sigma : float, default=1.0
            Step size (innovation standard deviation)
            - Must be positive
            - Controls diffusion rate
            - Variance grows as k·σ²

        mu : float, default=0.0
            Drift (expected change per step)
            - Can be any real number
            - μ = 0: Symmetric random walk
            - μ > 0: Upward drift
            - μ < 0: Downward drift

        dt : float, default=1.0
            Time step [units]
            - Required for discrete system
            - Sets time scale

        Raises
        ------
        ValueError
            If sigma ≤ 0

        Notes
        -----
        **Unit Root Property:**
        This is AR(1) with φ = 1:
            X[k+1] = 1·X[k] + μ + σ·w[k]

        Critical boundary between stationary and explosive.

        **Non-Stationarity:**
        Unlike AR(1) with |φ| < 1:
        - Variance grows: Var[X[k]] = k·σ²
        - No equilibrium distribution
        - Wanders arbitrarily far

        **Perfect Memory:**
        All past shocks permanently affect current value:
            X[k] = X[0] + μ·k + Σ σ·w[j]

        **Martingale (μ=0):**
        Best forecast: E[X[k+1] | X[k]] = X[k]

        Fair game - expected change is zero.

        **With Drift (μ≠0):**
        Expected value: E[X[k]] = X[0] + μ·k
        Linear trend in addition to random fluctuations.

        **Scaling to Brownian Motion:**
        As Δt → 0 with σ² ∝ Δt:
            X[t/Δt] / √(Δt) → Brownian motion

        This is Donsker's invariance principle.

        **Diffusion Coefficient:**
        In continuous limit:
            D = σ²/(2·Δt)

        **Parameter Selection:**
        - Financial returns: σ ≈ 0.01-0.03 (daily)
        - Physics: σ from Einstein relation
        - Generic: σ = 1 for baseline

        **Comparison with AR(1):**
        AR(1) with φ close to 1 approximates random walk:
        - φ = 0.99: Nearly unit root
        - φ = 1.00: Exact random walk
        - Hard to distinguish in finite samples

        **First Difference:**
        ΔX[k] = X[k] - X[k-1] = μ + σ·w[k]

        Differences are stationary white noise (with mean μ).
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        # Define symbolic variables
        x = sp.symbols("x", real=True)
        sigma_sym = sp.symbols("sigma", positive=True)
        mu_sym = sp.symbols("mu", real=True)

        # System definition
        self.state_vars = [x]
        self.control_vars = []  # Autonomous

        # Random walk: X[k+1] = X[k] + μ
        # Persistence with drift
        self._f_sym = sp.Matrix([[x + mu_sym]])

        # Additive noise
        self.diffusion_expr = sp.Matrix([[sigma_sym]])

        self.parameters = {sigma_sym: sigma, mu_sym: mu}
        self.order = 1
        self._dt = dt  # Required for discrete system
        self.sde_type = "ito"  # Discrete analog

    def get_variance(self, n_steps: int) -> float:
        """
        Get theoretical variance after n steps.

        For random walk: Var[X[k]] = k·σ²

        Parameters
        ----------
        n_steps : int
            Number of steps (must be non-negative)

        Returns
        -------
        float
            Variance after n steps

        Notes
        -----
        Variance grows linearly with time - hallmark of diffusion.
        """
        if n_steps < 0:
            raise ValueError(f"n_steps must be non-negative, got {n_steps}")

        sigma = next(v for k, v in self.parameters.items() if str(k) == "sigma")
        return n_steps * sigma**2

    def get_std(self, n_steps: int) -> float:
        """
        Get theoretical standard deviation after n steps.

        For random walk: Std[X[k]] = σ·√k

        Parameters
        ----------
        n_steps : int
            Number of steps

        Returns
        -------
        float
            Standard deviation after n steps

        Notes
        -----
        √k scaling characteristic of diffusion.
        """
        return np.sqrt(self.get_variance(n_steps))

    def get_mean(self, n_steps: int) -> float:
        """
        Get expected value after n steps (from X[0]=0).

        For random walk: E[X[k]] = μ·k

        Parameters
        ----------
        n_steps : int
            Number of steps

        Returns
        -------
        float
            Expected value
        """
        mu = next(v for k, v in self.parameters.items() if str(k) == "mu")
        return n_steps * mu


# ============================================================================
# Convenience Factory Functions
# ============================================================================


def create_symmetric_random_walk(sigma: float = 1.0, dt: float = 1.0) -> DiscreteRandomWalk:
    """
    Create symmetric random walk (no drift).

    X[k+1] = X[k] + σ·w[k]

    Parameters
    ----------
    sigma : float
        Step size
    dt : float
        Time step

    Returns
    -------
    DiscreteRandomWalk
        Symmetric random walk

    Examples
    --------
    >>> # Standard symmetric random walk
    >>> rw = create_symmetric_random_walk(sigma=1.0, dt=1.0)
    >>> # Equal probability of moving up or down
    """
    return DiscreteRandomWalk(sigma=sigma, mu=0.0, dt=dt)


def create_biased_random_walk(
    sigma: float = 1.0,
    drift: float = 0.1,
    dt: float = 1.0
) -> DiscreteRandomWalk:
    """
    Create biased random walk (with drift).

    X[k+1] = X[k] + μ + σ·w[k]

    Parameters
    ----------
    sigma : float
        Step size (volatility)
    drift : float
        Expected change per step
    dt : float
        Time step

    Returns
    -------
    DiscreteRandomWalk
        Biased random walk

    Examples
    --------
    >>> # Upward-drifting random walk
    >>> rw = create_biased_random_walk(
    ...     sigma=1.0,
    ...     drift=0.05,  # Positive drift
    ...     dt=1.0
    ... )
    """
    return DiscreteRandomWalk(sigma=sigma, mu=drift, dt=dt)