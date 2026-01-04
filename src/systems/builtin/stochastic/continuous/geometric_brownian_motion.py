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
Geometric Brownian Motion - Multiplicative Noise Stochastic System
===================================================================

This module provides implementations of Geometric Brownian Motion (GBM),
the fundamental model for processes exhibiting exponential growth with
multiplicative (proportional) noise. GBM is distinguished by:

- The canonical stochastic process in mathematical finance
- Foundation of the Black-Scholes-Merton option pricing model (Nobel Prize 1997)
- Standard model for asset price dynamics
- Ensures positivity and naturally expresses percentage changes
- Bridge between continuous-time stochastic calculus and discrete returns

GBM represents the continuous-time limit of discrete compound returns,
making it natural for modeling processes where percentage changes (not
absolute changes) are the fundamental random quantity.

Mathematical Background
-----------------------
Geometric Brownian Motion differs fundamentally from arithmetic Brownian
motion by having multiplicative (state-dependent) rather than additive
(state-independent) noise.

**Why Multiplicative Noise?**
Consider a $100 stock with $10 volatility (additive):
- At S=$100: 10% volatility (reasonable)
- At S=$10: 100% volatility (unrealistic)
- Can become negative

With multiplicative noise σ·S:
- Percentage volatility constant at all price levels
- Dollar volatility scales appropriately
- Always positive if started positive

Mathematical Formulation
------------------------
**Stochastic Differential Equation:**
    dX = μ·X·dt + σ·X·dW

where:
    - X(t): State variable (price, population) > 0
    - μ ∈ ℝ: Drift (expected growth rate, dimension: 1/time)
    - σ > 0: Volatility (noise intensity, dimension: 1/√time)
    - W(t): Standard Wiener process
    - dW ~ N(0, dt)

**With Control:**
    dX = (μ·X + u)·dt + σ·X·dW

**Itô vs Stratonovich:**
Itô form (standard in finance):
    dX = μ·X·dt + σ·X·dW

Stratonovich equivalent:
    dX = (μ - σ²/2)·X·dt + σ·X∘dW

The drift differs by the Itô correction -σ²/2·X.

Analytical Solution
-------------------
For autonomous case (u=0) with X(0)=X₀>0:

**Explicit Solution:**
    X(t) = X₀·exp((μ - σ²/2)·t + σ·W(t))

**Log Process:**
    ln(X(t)) = ln(X₀) + (μ - σ²/2)·t + σ·W(t)
    ln(X(t)/X₀) ~ N((μ - σ²/2)·t, σ²·t)

Therefore X(t) is **log-normally distributed**.

**The Itô Correction:**
The drift in ln(X) is μ - σ²/2, not μ. This -σ²/2 term arises
from quadratic variation (dW)² = dt.

**Moments:**
Mean: E[X(t)] = X₀·exp(μ·t)
Variance: Var[X(t)] = X₀²·exp(2μ·t)·(exp(σ²·t) - 1)
Median: Median[X(t)] = X₀·exp((μ - σ²/2)·t) < E[X(t)]

**Asymptotic Behavior:**
- If μ > σ²/2: X(t) → ∞ almost surely (growth)
- If μ = σ²/2: X(t) oscillates (critical)
- If μ < σ²/2: X(t) → 0 almost surely (extinction)

Key Properties
--------------
1. **Positivity:** X(t) > 0 for all t if X₀ > 0
2. **Log-Normality:** X(t) ~ LogNormal (right-skewed, heavy tails)
3. **Markov Property:** Future independent of past given present
4. **Martingale:** e^(-μt)·X(t) is martingale
5. **Scale Invariance:** Dynamics independent of price level
6. **Multiplicative Noise:** Percentage volatility constant

Physical Interpretation
-----------------------
**Drift μ:**
- Expected continuous compound growth rate [1/time]
- μ=0.10: 10% annual growth
- μ=0: Fair game (martingale)
- μ<0: Expected decay

**Volatility σ:**
- Standard deviation of log returns [1/√time]
- σ=0.20: 20% annual volatility (typical stock)
- σ=0.50: High volatility
- Time scaling: σ_daily = σ_annual/√252

**Sharpe Ratio:**
Risk-adjusted return: S = (μ - r)/σ
Typical values: 0.3-0.5

Applications
------------
1. **Finance:** Stock prices, option pricing, portfolio dynamics
2. **Biology:** Population with environmental noise, extinction analysis
3. **Economics:** GDP growth, commodity prices
4. **Physics:** Multiplicative noise processes
5. **Engineering:** Component degradation models

Numerical Simulation
--------------------
**Exact Discretization (Preferred):**
    X[k+1] = X[k]·exp((μ - σ²/2)·Δt + σ·√Δt·Z[k])

where Z ~ N(0,1). This is **exact** with no discretization error.

**Advantages:**
- Always positive
- Matches all moments exactly
- Can use large Δt

**Euler-Maruyama (Not Recommended):**
    X[k+1] = X[k]·(1 + μ·Δt + σ·√Δt·Z[k])

Can become negative; biased for finite Δt.

Limitations
-----------
Real financial data often shows:
- Fat tails (excess kurtosis beyond log-normal)
- Volatility clustering (time-varying σ)
- Leverage effect (σ increases when price falls)
- Mean reversion (especially commodities)
- Jumps (discontinuous movements)

**Extensions:** Stochastic volatility (Heston), jump diffusion (Merton),
local volatility, regime-switching models.
"""

import numpy as np
import sympy as sp

from src.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem


class GeometricBrownianMotion(ContinuousStochasticSystem):
    """
    Geometric Brownian motion with multiplicative (state-dependent) noise.

    Foundation of Black-Scholes model. Ensures positive states and
    naturally expresses percentage returns.

    Stochastic Differential Equation
    ---------------------------------
    dX = (μ·X + u)·dt + σ·X·dW

    where:
        X(t) ∈ ℝ₊: State (price, population)
        μ: Drift (expected growth rate)
        σ > 0: Volatility (noise intensity)
        u: Control (optional, additive)
        W(t): Standard Wiener process

    Key Features
    ------------
    - **Multiplicative Noise:** σ·X scales with state
    - **Positivity:** X(t) > 0 if X₀ > 0
    - **Log-Normality:** X(t) ~ LogNormal
    - **Exponential Growth:** E[X(t)] = X₀·exp(μ·t)

    Mathematical Properties
    -----------------------
    **Exact Solution (u=0):**
        X(t) = X₀·exp((μ - σ²/2)·t + σ·W(t))

    **Moments:**
    - Mean: E[X(t)] = X₀·exp(μ·t)
    - Variance: Var[X(t)] = X₀²·exp(2μ·t)·(exp(σ²·t) - 1)
    - Median: X₀·exp((μ - σ²/2)·t)

    **Asymptotic Behavior:**
    - μ > σ²/2: Growth to ∞
    - μ = σ²/2: Oscillates
    - μ < σ²/2: Decay to 0

    Parameters
    ----------
    mu : float, default=0.1
        Drift coefficient (expected growth rate)
        Typical: -0.1 to 0.3 for stocks
        
    sigma : float, default=0.2
        Volatility (must be positive)
        Typical stocks: 0.15-0.30 (15-30% annual)

    State Space
    -----------
    State: x ∈ ℝ₊ = (0, ∞)
    Control: u ∈ ℝ (optional)

    Stochastic Properties
    ---------------------
    - Noise Type: MULTIPLICATIVE
    - Diffusion: g(x) = σ·x (state-dependent)
    - SDE Type: Itô (standard)
    - Noise Dimension: nw = 1

    Applications
    ------------
    **Finance:** Stock prices, Black-Scholes model
    **Biology:** Population dynamics with noise
    **Economics:** GDP growth models
    **Physics:** Multiplicative noise processes

    Numerical Simulation
    --------------------
    **Exact Scheme (Recommended):**
        X[k+1] = X[k]·exp((μ - σ²/2)·Δt + σ·√Δt·Z[k])
    
    Always positive, no discretization error.

    Limitations
    -----------
    - Assumes constant volatility
    - No jumps (continuous paths only)
    - No mean reversion
    - Empirical returns show fat tails

    See Also
    --------
    BrownianMotion : Additive noise version
    OrnsteinUhlenbeck : Mean-reverting process
    CoxIngersollRoss : Mean-reverting with multiplicative noise
    """

    def define_system(self, mu: float = 0.1, sigma: float = 0.2):
        """
        Define geometric Brownian motion dynamics.

        Parameters
        ----------
        mu : float, default=0.1
            Drift coefficient (growth rate)
            μ > 0: Growth, μ = 0: Fair game, μ < 0: Decay
            
        sigma : float, default=0.2
            Volatility (must be positive)
            Typical stocks: 0.15-0.30

        Raises
        ------
        ValueError
            If sigma ≤ 0

        Notes
        -----
        **Parameter Guidelines:**
        - Large cap stocks: μ≈0.08, σ≈0.15-0.25
        - Small cap stocks: μ≈0.12, σ≈0.30-0.50
        - Sharpe ratio: (μ-r)/σ ≈ 0.3-0.5

        **State Positivity:**
        Multiplicative noise g(x) = σ·x ensures X(t) > 0
        for all t if X(0) > 0.

        **Itô Correction:**
        Expected log-return is μ - σ²/2, not μ, due to
        quadratic variation.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        # Symbolic variables
        x = sp.symbols("x", positive=True)
        u = sp.symbols("u", real=True)
        mu_sym = sp.symbols("mu", real=True)
        sigma_sym = sp.symbols("sigma", positive=True)

        # System definition
        self.state_vars = [x]
        self.control_vars = [u]

        # Drift: f(x,u) = μ·x + u
        self._f_sym = sp.Matrix([[mu_sym * x + u]])

        self.parameters = {mu_sym: mu, sigma_sym: sigma}
        self.order = 1

        # Diffusion: g(x,u) = σ·x (multiplicative!)
        self.diffusion_expr = sp.Matrix([[sigma_sym * x]])
        self.sde_type = "ito"

    def get_expected_value(self, x0: float, t: float, u: float = 0.0) -> float:
        """
        Compute analytical expected value E[X(t)].

        For u=0: E[X(t)] = X₀·exp(μ·t)

        Parameters
        ----------
        x0 : float
            Initial state (must be positive)
        t : float
            Time (non-negative)
        u : float, default=0.0
            Control (assumed constant)

        Returns
        -------
        float
            Expected value E[X(t)]

        Notes
        -----
        Mean grows exponentially at rate μ:
        - Doubling time: ln(2)/μ (if μ > 0)
        - Half-life: ln(2)/|μ| (if μ < 0)

        Examples
        --------
        >>> gbm = GeometricBrownianMotion(mu=0.10, sigma=0.20)
        >>> E_1yr = gbm.get_expected_value(x0=100, t=1.0)
        >>> print(f"Expected: ${E_1yr:.2f}")  # $110.52
        """
        if x0 <= 0:
            raise ValueError(f"x0 must be positive, got {x0}")
        if t < 0:
            raise ValueError(f"t must be non-negative, got {t}")

        mu = next(val for key, val in self.parameters.items() if str(key) == "mu")

        if u == 0:
            return x0 * np.exp(mu * t)
        return x0 * np.exp(mu * t) + u * t * np.exp(mu * t)

    def get_variance(self, x0: float, t: float) -> float:
        """
        Compute analytical variance Var[X(t)].

        Var[X(t)] = X₀²·exp(2μ·t)·(exp(σ²·t) - 1)

        Parameters
        ----------
        x0 : float
            Initial state (positive)
        t : float
            Time (non-negative)

        Returns
        -------
        float
            Variance Var[X(t)]

        Notes
        -----
        Coefficient of variation grows unboundedly:
            CV = √(exp(σ²·t) - 1) → ∞ as t → ∞

        Examples
        --------
        >>> gbm = GeometricBrownianMotion(mu=0.05, sigma=0.20)
        >>> var = gbm.get_variance(x0=100, t=1.0)
        >>> std = np.sqrt(var)
        >>> print(f"Std Dev: ${std:.2f}")
        """
        if x0 <= 0:
            raise ValueError(f"x0 must be positive, got {x0}")
        if t < 0:
            raise ValueError(f"t must be non-negative, got {t}")

        mu = next(val for key, val in self.parameters.items() if str(key) == "mu")
        sigma = next(val for key, val in self.parameters.items() if str(key) == "sigma")

        return x0**2 * np.exp(2 * mu * t) * (np.exp(sigma**2 * t) - 1)


class BrownianMotionWithDrift(GeometricBrownianMotion):
    """
    Alias for GeometricBrownianMotion.

    Alternative name sometimes used in literature.
    Mathematically identical to GBM.
    """
    pass


def create_stock_price_model(
    expected_return: float = 0.07,
    annual_volatility: float = 0.20,
) -> GeometricBrownianMotion:
    """
    Create GBM model for stock price dynamics.

    Uses financial conventions: annual parameters, time in years.

    Parameters
    ----------
    expected_return : float, default=0.07
        Expected annual return (e.g., 0.07 = 7%)
        Typical: 0.05-0.15
        
    annual_volatility : float, default=0.20
        Annual volatility (e.g., 0.20 = 20%)
        Typical: 0.15-0.50

    Returns
    -------
    GeometricBrownianMotion
        Stock price model

    Examples
    --------
    >>> # S&P 500 typical
    >>> sp500 = create_stock_price_model(
    ...     expected_return=0.10,
    ...     annual_volatility=0.18
    ... )
    >>> 
    >>> # Conservative large cap
    >>> large_cap = create_stock_price_model(
    ...     expected_return=0.08,
    ...     annual_volatility=0.15
    ... )
    """
    return GeometricBrownianMotion(mu=expected_return, sigma=annual_volatility)