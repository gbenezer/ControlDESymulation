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
Discrete ARMA Process - Autoregressive Moving Average Models
=============================================================

This module provides ARMA (Autoregressive Moving Average) models, which combine
AR and MA components to create the most flexible linear time series models.
ARMA processes serve as:

- The foundation of the Box-Jenkins methodology (Nobel Prize 2003, Granger & Engle)
- The most parsimonious linear models for stationary time series
- A framework unifying AR (infinite MA) and MA (infinite AR) representations
- The building blocks for ARIMA, SARIMA, and state-space models
- The standard for forecasting in econometrics and signal processing

ARMA models extend AR by adding moving average terms, creating models that:
1. Require fewer parameters than pure AR for same flexibility
2. Better capture sharp transients and impulse responses
3. Provide both recursive (AR) and smoothing (MA) components
4. Enable exact representation of many physical systems

The ARMA(1,1) model is the simplest non-trivial ARMA, combining one AR lag
with one MA lag to efficiently model a wide range of time series behaviors.

Mathematical Background
-----------------------

**Pure AR(p):**
    X[k] = Σφᵢ·X[k-i] + w[k]

Advantages: Simple recursion, only past X needed
Disadvantages: May need many lags (large p) for complex dynamics

**Pure MA(q):**
    X[k] = Σθⱼ·w[k-j] + w[k]

Advantages: Finite impulse response, always stationary
Disadvantages: Need past innovations w (not observed directly)

**ARMA(p,q): Best of Both Worlds**
    X[k] = Σφᵢ·X[k-i] + Σθⱼ·w[k-j] + w[k]

Combines:
- AR component: Recursive structure (memory)
- MA component: Smoothing structure (filtering)

**ARMA(1,1) - Simplest Non-Trivial:**
    X[k] = φ·X[k-1] + θ·w[k-1] + w[k]

Three parameters (φ, θ, σ) provide flexibility of AR(∞) or MA(∞).

**Historical Development:**

- **1927**: Yule introduces AR models
- **1937**: Wold proves decomposition theorem (any stationary = linear filter)
- **1938**: Slutsky introduces MA models
- **1970**: Box & Jenkins systematize ARMA methodology
  - Identification: Determine (p,q) from data
  - Estimation: MLE for parameters
  - Diagnostics: Check residuals
- **Present**: Industry standard (economics, finance, engineering)

Mathematical Formulation
------------------------

**ARMA(1,1) Process:**

Standard form:
    X[k] = φ·X[k-1] + w[k] + θ·w[k-1]

where:
    - X[k]: State (observed time series)
    - φ: AR coefficient (persistence)
    - θ: MA coefficient (smoothing)
    - w[k]: White noise innovation ~ N(0, σ²)

**With Control:**
    X[k] = φ·X[k-1] + u[k] + w[k] + θ·w[k-1]

Control u[k] provides external forcing.

**State-Space Representation:**

ARMA can be written as state-space (enables Kalman filter):
    [X[k]  ] = [φ  θ]·[X[k-1]] + [1]·w[k]
    [w[k-1]]   [0  0] [w[k-2]]   [1]

Augmented state includes past innovation.

**Lag Operator Form:**
    (1 - φ·L)·X[k] = (1 + θ·L)·w[k]

where L is lag operator: L·X[k] = X[k-1]

Transfer function (z-domain):
    H(z) = (1 + θ·z⁻¹)/(1 - φ·z⁻¹)

Analytical Properties
---------------------

**Stationarity:**

ARMA(1,1) is stationary if and only if |φ| < 1.

MA part always stationary (finite memory).
AR part requires |φ| < 1 for stability.

**Invertibility:**

ARMA(1,1) is invertible if and only if |θ| < 1.

Invertibility means can write as AR(∞):
    X[k] = Σπⱼ·X[k-j] + w[k]

Important for:
- Forecasting (express in terms of past X)
- Identification (uniqueness)
- Estimation (likelihood computation)

**Both Conditions:**
For well-defined ARMA(1,1): |φ| < 1 AND |θ| < 1

**Moments (Stationary):**

Mean: E[X[k]] = 0 (or μ if non-centered)

Variance:
    γ(0) = σ²·(1 + 2θφ + θ²)/(1 - φ²)

Autocovariance (lag 1):
    γ(1) = σ²·(φ + θ)·(1 + θφ)/(1 - φ²)

Autocovariance (lag h ≥ 2):
    γ(h) = φ·γ(h-1) = φ^(h-1)·γ(1)

**Autocorrelation Function:**

    ρ(1) = (φ + θ)·(1 + θφ)/(1 + 2θφ + θ²)
    ρ(h) = φ^(h-1)·ρ(1) for h ≥ 2

Geometric decay after lag 1 (AR signature).
But ρ(1) not necessarily φ (MA effect).

**Key Insight:**
ARMA(1,1) ACF looks like AR(1) for h ≥ 2, but ρ(1) is different.
This is the MA component's signature.

Identification and Estimation
------------------------------

**Box-Jenkins Methodology:**

**1. Identification (Determine p, q):**

Use ACF and PACF:
- **ACF (Autocorrelation Function):** ρ(h) vs h
- **PACF (Partial Autocorrelation Function):** φ_hh vs h

**Patterns:**
- AR(p): ACF decays, PACF cuts off after lag p
- MA(q): ACF cuts off after lag q, PACF decays
- ARMA(p,q): Both ACF and PACF decay (no clean cutoff)

For ARMA(1,1):
- ACF: Decays geometrically after lag 1
- PACF: Decays geometrically (no cutoff)

**2. Estimation (Fit parameters):**

**Maximum Likelihood:**
Most common for ARMA.

Log-likelihood (Gaussian):
    ℓ(φ,θ,σ²) = -(N/2)·ln(2π·σ²) - (1/2σ²)·Σε²[k]

where ε[k] are innovations (one-step prediction errors).

Maximize numerically (nonlinear optimization).

**Conditional Least Squares:**
Minimize Σε²[k] given initial conditions.

**Yule-Walker:**
Not applicable to ARMA (only AR).

**3. Diagnostics (Check model adequacy):**

Residuals ε[k] should be white noise:
- ACF of residuals: ≈ 0 for all h > 0
- Ljung-Box test: Q-statistic
- Normality: Jarque-Bera test

If residuals not white: Need higher order (p,q).

**Model Selection:**

Compare ARMA(p,q) for different (p,q):
- AIC (Akaike): Penalizes parameters moderately
- BIC (Bayesian): Stronger penalty (prefers parsimony)
- Out-of-sample forecast error

ARMA(1,1) often wins: Parsimonious yet flexible.

Key Properties
--------------

**1. Parsimony:**

ARMA(1,1) with 3 parameters (φ, θ, σ) can match:
- AR(∞): Infinite AR representation
- MA(∞): Infinite MA representation

More economical than high-order AR or MA.

**2. Duality:**

Can represent same process as:
- AR(∞): X[k] = Σπⱼ·X[k-j] + w[k]
- MA(∞): X[k] = ΣψⱼÂ·w[k-j]

Coefficients decay geometrically.

**3. Stationarity + Invertibility:**

Both required for well-behaved ARMA:
- |φ| < 1: Stationary (stability)
- |θ| < 1: Invertible (uniqueness)

**4. Spectral Density:**

Power spectrum:
    S(f) = σ²·|1 + θ·e^(-2πif·Δt)|² / |1 - φ·e^(-2πif·Δt)|²

Ratio of MA and AR parts creates flexible shapes.

**5. Impulse Response:**

Response to w[k] = 1, w[j] = 0 for j ≠ k:
    h[0] = 1
    h[1] = φ + θ
    h[j] = φ^(j-1)·(φ + θ) for j ≥ 2

Geometric decay after initial transient.

**6. Markov (in Augmented State):**

Not Markov in X alone, but Markov in (X, w_prev).

Applications
------------

**1. Econometrics:**

**GDP Growth:**
Quarterly GDP often ARMA(1,1):
- AR: Persistence from economic momentum
- MA: Smoothing from measurement/aggregation

**Inflation:**
Monthly inflation well-modeled by ARMA.

**Interest Rate Spreads:**
Mean-reverting with MA component.

**2. Finance:**

**Asset Returns:**
Daily returns often have MA component:
- AR: Momentum, feedback trading
- MA: Bid-ask bounce, non-synchronous trading

**Volatility:**
GARCH models often combined with ARMA for returns.

**3. Signal Processing:**

**ARMA Filter:**
Rational transfer function:
- Poles from AR (φ)
- Zeros from MA (θ)
- Efficient IIR filter

**4. Control Systems:**

**Disturbance Model:**
ARMA for colored noise disturbances.

**System Identification:**
Fit ARMA to input-output data.

**5. Weather & Climate:**

**Temperature Anomalies:**
Daily/monthly temperatures.

**Precipitation:**
Rainfall models.

**6. Engineering:**

**Vibration:**
Mechanical systems with ARMA response.

**Communication:**
Channel models.

Comparison with AR and MA
--------------------------

**AR(1) Only:**
    X[k] = φ·X[k-1] + w[k]

Simple but may need high p for complex ACF.

**MA(1) Only:**
    X[k] = θ·w[k-1] + w[k]

Always stationary but not invertible if |θ| > 1.

**ARMA(1,1):**
    X[k] = φ·X[k-1] + θ·w[k-1] + w[k]

More flexible:
- Can match wider range of ACFs
- Fewer parameters than high-order AR or MA
- Industry standard for parsimony

**Equivalences:**

ARMA(1,1) reduces to:
- AR(1): Set θ = 0
- MA(1): Set φ = 0
- White noise: Set φ = θ = 0
- Random walk: Set φ = 1, θ = 0 (non-stationary)

ARMA(1,1) nests all simpler models.

Numerical Simulation
--------------------

**Challenge:**

Cannot directly simulate X[k] = φ·X[k-1] + θ·w[k-1] + w[k]
because w[k-1] not stored in state.

**Solution 1: State-Space Form**

Augmented state Z = [X, w_prev]:
    [X[k]  ] = [φ  θ]·[X[k-1]  ] + [1]·w[k]
    [w[k-1]]   [0  0] [w[k-2]]    [1]

Then X[k] is first component of Z[k].

**Solution 2: Direct Recursion**

Store past innovation:
```python
X = np.zeros(N)
w_prev = 0
for k in range(N):
    w_k = sigma * np.random.randn()
    X[k] = phi * X[k-1] + theta * w_prev + w_k
    w_prev = w_k
```

**This Implementation:** State-space form (cleaner, enables Kalman filter).

Common Pitfalls
---------------

1. **Confusing φ and θ:**
   - φ: AR coefficient (past observations)
   - θ: MA coefficient (past innovations)
   - Different interpretations!

2. **Non-Invertibility:**
   - If |θ| > 1: Multiple ARMA give same ACF
   - Non-unique representation
   - Estimation problems

3. **Overfitting:**
   - High (p,q): More parameters, better fit
   - But poor out-of-sample
   - Use information criteria (AIC, BIC)

4. **Identification Errors:**
   - ACF/PACF don't always clearly indicate (p,q)
   - Try multiple models, compare
   - Use automated selection

5. **Initial Conditions:**
   - Need w[0], w[-1], ... for MA terms
   - Typically assume zero (transient effect)
   - Or estimate from data

6. **Computational:**
   - MLE requires nonlinear optimization
   - Can have multiple local minima
   - Need good starting values

Historical Significance
-----------------------

**Box-Jenkins Revolution (1970):**

George Box and Gwilym Jenkins systematized ARMA modeling:

**Three-Step Methodology:**
1. **Identification:** Use ACF/PACF to select (p,q)
2. **Estimation:** MLE to fit parameters
3. **Diagnostic Checking:** Test residuals for white noise

**Impact:**
- Made time series analysis systematic
- Accessible to practitioners (not just theorists)
- Foundation of modern forecasting
- Used in business, economics, engineering

**Nobel Prizes:**

- **2003**: Granger (cointegration) & Engle (ARCH/GARCH)
  - Both built on ARMA foundation
  - Extensions to non-stationary and heteroskedastic

**Legacy:**
ARMA models remain standard despite:
- Neural networks (black box, no interpretation)
- State-space models (more general but complex)
- Nonlinear methods (need more data)

ARMA often wins for:
- Parsimony (few parameters)
- Interpretability (clear ACF/PACF signatures)
- Forecasting accuracy (linear often sufficient)

"""

import numpy as np
import sympy as sp
from typing import Optional

from src.systems.base.core.discrete_stochastic_system import DiscreteStochasticSystem


class DiscreteARMA11(DiscreteStochasticSystem):
    """
    ARMA(1,1) process - combines autoregressive and moving average.

    The simplest non-trivial ARMA model, combining one AR lag with one MA lag
    to create a flexible yet parsimonious time series model. This is the
    workhorse of Box-Jenkins methodology and the foundation of modern
    forecasting in econometrics.

    Difference Equation
    -------------------
    Standard ARMA(1,1) form:
        X[k] = φ·X[k-1] + w[k] + θ·w[k-1]

    With control:
        X[k] = φ·X[k-1] + u[k] + w[k] + θ·w[k-1]

    where:
        - X[k]: Observed series (state)
        - φ: AR coefficient (persistence, -∞ < φ < ∞)
        - θ: MA coefficient (smoothing, -∞ < θ < ∞)
        - w[k]: White noise innovation ~ N(0, σ²)
        - u[k]: Control/external input

    **State-Space Form (for simulation):**
        Z[k] = [X[k], w[k-1]]ᵀ (augmented state)
        
        Z[k+1] = [φ  θ]·Z[k] + [1]·w[k+1]
                 [0  0]        [1]

    Output: X[k] = [1, 0]·Z[k]

    Physical Interpretation
    -----------------------
    **AR Component (φ):**
    - Past observation influences current
    - φ > 0: Positive persistence (typical)
    - φ < 0: Alternation (rare)
    - Recursive: Memory of all past (exponentially weighted)

    **MA Component (θ):**
    - Past innovation influences current
    - θ > 0: Smoothing (positive autocorrelation at lag 1)
    - θ < 0: Overshoot (negative autocorrelation)
    - Finite: Only immediate past innovation matters

    **Combined Effect:**
    - AR provides long memory (slow decay)
    - MA adjusts short-term behavior (lag 1)
    - Together: More flexible than either alone

    **Why ARMA(1,1) is Popular:**

    Can approximate:
    - AR(∞): Via MA(∞) representation
    - MA(∞): Via AR(∞) representation
    - Complex ACF: With just 3 parameters

    More parsimonious than high-order AR or MA.

    Key Features
    ------------
    **Stationarity:**
    Requires |φ| < 1 (from AR part).

    **Invertibility:**
    Requires |θ| < 1 (from MA part).

    **Parsimony:**
    3 parameters (φ, θ, σ) provide flexibility of high-order models.

    **Autocorrelation:**
    - ρ(1): Affected by both φ and θ
    - ρ(h ≥ 2): Geometric decay at rate φ

    **Spectral Density:**
    Rational function (ratio of polynomials).

    **Markov (in Augmented State):**
    State-space form is Markov in (X, w_prev).

    Mathematical Properties
    -----------------------
    **Stationarity Condition:**
        |φ| < 1

    **Invertibility Condition:**
        |θ| < 1

    **Variance:**
        Var[X] = σ²·(1 + 2θφ + θ²)/(1 - φ²)

    **Autocorrelation:**
        ρ(1) = (φ + θ)·(1 + θφ)/(1 + 2θφ + θ²)
        ρ(h) = φ·ρ(h-1) for h ≥ 2

    **Impulse Response:**
        ψ₀ = 1
        ψ₁ = φ + θ
        ψⱼ = φ^(j-1)·(φ + θ) for j ≥ 2

    **MA(∞) Representation:**
        X[k] = Σψⱼ·w[k-j]

    Coefficients decay geometrically: ψⱼ ~ φ^j

    **AR(∞) Representation:**
        w[k] = Σπⱼ·X[k-j]

    Coefficients decay geometrically: πⱼ ~ θ^j (if |θ| < 1)

    Physical Interpretation
    -----------------------
    **AR Coefficient φ:**
    - Dimensionless
    - Fraction of past persisting
    - Typical: 0.3-0.9 (positive persistence)

    **MA Coefficient θ:**
    - Dimensionless
    - Weight on past shock
    - Typical: -0.5 to +0.5
    - θ > 0: Smoothing (positive lag-1 correlation)
    - θ < 0: Overshoot (negative lag-1 correlation)

    **Innovation Variance σ²:**
    - Units: [state]²
    - Unpredictable component each period

    **Special Cases:**

    φ = 0.8, θ = 0:    Pure AR(1)
    φ = 0, θ = -0.5:   Pure MA(1)
    φ = 0.8, θ = -0.6: Typical ARMA(1,1)

    State Space
    -----------
    Augmented state: Z = [X, w_prev] ∈ ℝ²
        - X: Observed series
        - w_prev: Past innovation (latent)

    Observable state: X ∈ ℝ
        - What we actually measure

    Control: u ∈ ℝ (optional)
        - External input

    Parameters
    ----------
    phi : float, default=0.7
        AR coefficient
        - Typical: 0.3-0.9
        - Must satisfy |φ| < 1 for stationarity

    theta : float, default=0.3
        MA coefficient
        - Typical: -0.5 to +0.5
        - Should satisfy |θ| < 1 for invertibility

    sigma : float, default=0.1
        Innovation standard deviation
        - Must be positive
        - Sets scale of unpredictable component

    dt : float, default=1.0
        Sampling period
        - Required for discrete system

    Stochastic Properties
    ---------------------
    - System Type: LINEAR (ARMA)
    - Noise Type: ADDITIVE (white noise)
    - Markov: In augmented state (X, w_prev)
    - Stationary: If |φ| < 1
    - Invertible: If |θ| < 1
    - Gaussian: If w ~ N(0,σ²)

    Applications
    ------------
    **1. Economic Forecasting:**
    - GDP growth
    - Inflation
    - Unemployment
    - Sales data

    **2. Financial Time Series:**
    - Asset returns
    - Interest rate spreads
    - Volume data

    **3. Signal Processing:**
    - Digital filter design
    - Prediction algorithms
    - Spectral estimation

    **4. Engineering:**
    - Vibration analysis
    - Quality control
    - Process monitoring

    **5. Environmental:**
    - Temperature series
    - Rainfall
    - Pollution levels

    Numerical Simulation
    --------------------
    **State-Space Implementation:**

    Augment state with past innovation:
        Z[k+1] = F·Z[k] + G·w[k+1]

    where:
        Z = [X, w_prev]ᵀ
        F = [φ  θ]
            [0  0]
        G = [1]
            [1]

    Extract: X[k] = Z[k][0]

    Forecasting
    -----------
    **One-Step-Ahead:**
        X̂[k+1|k] = φ·X[k] + θ·ε[k]

    where ε[k] = X[k] - X̂[k|k-1] is innovation.

    **h-Step-Ahead:**
        X̂[k+h|k] = φ^h·X[k] + θ·Σφ^(h-1-j)·ε[k-j]

    Forecast variance increases with horizon.

    Comparison with Other Models
    -----------------------------
    **vs. AR(1):**
    - AR(1): Only φ parameter
    - ARMA(1,1): Adds θ (more flexible)
    - Same asymptotic ACF (h → ∞)
    - Different ρ(1)

    **vs. MA(1):**
    - MA(1): Only θ parameter
    - ARMA(1,1): Adds φ (long memory)

    **vs. AR(2):**
    - AR(2): Two AR lags (4 parameters with σ, μ)
    - ARMA(1,1): One AR, one MA (3 parameters)
    - Often ARMA(1,1) more parsimonious

    **vs. ARMA(p,q):**
    - ARMA(1,1): Simplest non-trivial
    - Higher order: More flexible but more parameters
    - Start with ARMA(1,1), increase if needed

    Limitations
    -----------
    - Linear only
    - Constant parameters
    - Gaussian innovations (typically)
    - Stationary (no trends)
    - Low order (1,1 may not suffice)

    Extensions
    ----------
    - ARMA(p,q): Higher order
    - ARIMA: Integrated (for non-stationary)
    - SARIMA: Seasonal patterns
    - GARCH: Time-varying variance
    - Regime-switching ARMA

    Examples
    --------
    Standard ARMA(1,1):
    
    >>> # Typical configuration
    >>> arma = DiscreteARMA11(
    ...     phi=0.7,    # Moderate persistence
    ...     theta=0.3,  # Positive MA (smoothing)
    ...     sigma=0.1,
    ...     dt=1.0
    ... )
    >>> 
    >>> # Check stationarity and invertibility
    >>> print(f"|φ| < 1: {abs(0.7) < 1}")  # Stationary
    >>> print(f"|θ| < 1: {abs(0.3) < 1}")  # Invertible

    Different configurations:
    
    >>> # High persistence, low MA
    >>> high_ar = DiscreteARMA11(phi=0.9, theta=0.1, sigma=0.1)
    >>> 
    >>> # Moderate AR, negative MA (overshoot)
    >>> negative_ma = DiscreteARMA11(phi=0.5, theta=-0.3, sigma=0.1)
    >>> 
    >>> # Balanced AR and MA
    >>> balanced = DiscreteARMA11(phi=0.6, theta=0.4, sigma=0.1)

    Simulation:
    
    >>> # Generate ARMA series
    >>> z0 = np.array([0.0, 0.0])  # [X[0], w[-1]]
    >>> u_seq = np.zeros((100, 1))
    >>> 
    >>> result = arma.simulate(z0, u_seq, n_steps=100)
    >>> 
    >>> # Extract observed series (first component of augmented state)
    >>> X_series = result['states'][:, 0]
    >>> 
    >>> # Compute sample ACF
    >>> from statsmodels.tsa.stattools import acf
    >>> acf_values = acf(X_series, nlags=20)

    See Also
    --------
    DiscreteAR1 : Pure autoregressive (θ=0)
    DiscreteMA1 : Pure moving average (φ=0, future)
    DiscreteARMA : General ARMA(p,q) (future)

    References
    ----------
    .. [1] Box, G.E.P. & Jenkins, G.M. (1970). "Time Series Analysis: 
           Forecasting and Control"
    .. [2] Hamilton, J.D. (1994). "Time Series Analysis"
    .. [3] Brockwell, P.J. & Davis, R.A. (2016). "Introduction to Time 
           Series and Forecasting"
    .. [4] Shumway, R.H. & Stoffer, D.S. (2017). "Time Series Analysis 
           and Its Applications"
    """

    def define_system(
        self,
        phi: float = 0.7,
        theta: float = 0.3,
        sigma: float = 0.1,
        dt: float = 1.0,
    ):
        """
        Define ARMA(1,1) process dynamics in state-space form.

        Parameters
        ----------
        phi : float, default=0.7
            AR coefficient (persistence)
            - Typical: 0.3-0.9
            - Should satisfy |φ| < 1 for stationarity
            - Similar to AR(1)

        theta : float, default=0.3
            MA coefficient (smoothing)
            - Typical: -0.5 to +0.5
            - Should satisfy |θ| < 1 for invertibility
            - θ > 0: Smoothing (common)
            - θ < 0: Overshoot (less common)

        sigma : float, default=0.1
            Innovation standard deviation
            - Must be positive
            - Sets scale of unpredictable shocks

        dt : float, default=1.0
            Sampling period
            - Required for discrete system

        Raises
        ------
        ValueError
            If sigma ≤ 0

        UserWarning
            If |φ| ≥ 1 (non-stationary)
            If |θ| ≥ 1 (non-invertible)

        Notes
        -----
        **Stationarity:**
        Requires |φ| < 1 (from AR component).
        - φ = 1: Unit root (ARIMA needed)
        - |φ| > 1: Explosive (unstable)

        **Invertibility:**
        Requires |θ| < 1 (from MA component).
        - |θ| ≥ 1: Non-unique representation
        - Multiple ARMA models give same ACF
        - Estimation problems

        **Parameter Space:**
        Valid region: |φ| < 1 AND |θ| < 1

        **State-Space Formulation:**

        To simulate ARMA, augment state with past innovation:
            Z = [X, w_prev]ᵀ

        Dynamics:
            Z[k+1] = F·Z[k] + G·w[k+1]

        where:
            F = [φ  θ]
                [0  0]
            
            G = [1]
                [1]

        **Why Augmented State?**
        Cannot write X[k+1] = f(X[k]) alone because depends on w[k].
        Need to track w[k] as auxiliary state.

        **Variance (Stationary):**
            Var[X] = σ²·(1 + 2θφ + θ²)/(1 - φ²)

        **Autocorrelation:**
            ρ(1) = (φ + θ)·(1 + θφ)/(1 + 2θφ + θ²)
            ρ(h) = φ·ρ(h-1) for h ≥ 2

        **Special Cases:**

        θ = 0: Reduces to AR(1)
            X[k] = φ·X[k-1] + w[k]

        φ = 0: Reduces to MA(1)
            X[k] = θ·w[k-1] + w[k]

        φ = -θ: Simplifies significantly
            X[k] = φ·(X[k-1] - w[k-1]) + w[k]
            (Differencing form)

        **Typical Configurations:**

        Economic data:
        - φ = 0.6-0.9 (high persistence)
        - θ = 0.1-0.5 (moderate smoothing)

        Financial returns:
        - φ = 0.0-0.3 (low persistence)
        - θ = -0.3-0.0 (slight overshoot or none)

        Physical measurements:
        - φ = 0.5-0.8 (moderate persistence)
        - θ = 0.2-0.5 (smoothing from measurement)

        Examples
        --------
        >>> # Standard economic series
        >>> economic = DiscreteARMA11(
        ...     phi=0.7,    # High persistence
        ...     theta=0.3,  # Smoothing
        ...     sigma=0.1
        ... )
        >>> 
        >>> # Financial returns (low persistence)
        >>> returns = DiscreteARMA11(
        ...     phi=0.1,     # Weak persistence
        ...     theta=-0.2,  # Slight overshoot
        ...     sigma=0.2
        ... )
        >>> 
        >>> # Physical sensor (measurement smoothing)
        >>> sensor = DiscreteARMA11(
        ...     phi=0.6,
        ...     theta=0.4,   # Strong smoothing
        ...     sigma=0.05
        ... )
        """
        # Validate parameters
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        # Check stationarity
        if abs(phi) >= 1:
            import warnings
            warnings.warn(
                f"|phi| = {abs(phi)} >= 1 creates non-stationary ARMA. "
                f"For stationarity, require |phi| < 1. "
                f"Consider ARIMA if unit root intended.",
                UserWarning
            )

        # Check invertibility
        if abs(theta) >= 1:
            import warnings
            warnings.warn(
                f"|theta| = {abs(theta)} >= 1 creates non-invertible ARMA. "
                f"Non-unique representation, estimation problems. "
                f"For invertibility, require |theta| < 1.",
                UserWarning
            )

        # Augmented state: [X, w_previous]
        X, w_prev = sp.symbols("X w_prev", real=True)
        u = sp.symbols("u", real=True)

        # Parameters
        phi_sym = sp.symbols("phi", real=True)
        theta_sym = sp.symbols("theta", real=True)
        sigma_sym = sp.symbols("sigma", positive=True)

        # State-space form
        # Z[k+1] = F·Z[k] + G·w[k+1] + B·u[k]
        # where Z = [X, w_prev]ᵀ and w[k+1] is new innovation

        # For ARMA(1,1): X[k] = φ·X[k-1] + u[k] + w[k] + θ·w[k-1]
        # In state-space:
        # X[k+1] = φ·X[k] + θ·w[k] + u[k+1] + w[k+1]
        # w[k] = w[k] (store for next iteration)

        # State update (deterministic part before adding new innovation)
        # This represents the conditional mean given past
        X_next = phi_sym * X + theta_sym * w_prev + u
        w_next = 0  # Will be replaced by new innovation w[k+1]

        self.state_vars = [X, w_prev]
        self.control_vars = [u]

        # Deterministic part (before adding new innovation)
        self._f_sym = sp.Matrix([X_next, w_next])

        self.parameters = {
            phi_sym: phi,
            theta_sym: theta,
            sigma_sym: sigma,
        }
        self.order = 1
        self._dt = dt

        # STOCHASTIC PART
        # New innovation w[k+1] added to both components:
        # - X equation gets w[k+1]
        # - w_prev equation stores w[k+1] for next iteration
        self.diffusion_expr = sp.Matrix([
            [sigma_sym],  # Innovation to X
            [sigma_sym],  # Innovation becomes w_prev
        ])

        self.sde_type = "ito"

        # Output: Observable is X only (not w_prev)
        self._h_sym = sp.Matrix([X])

    def get_stationary_variance(self) -> float:
        """
        Get theoretical stationary variance of X.

        Var[X] = σ²·(1 + 2θφ + θ²)/(1 - φ²)

        Returns
        -------
        float
            Stationary variance

        Raises
        ------
        ValueError
            If |φ| ≥ 1 (non-stationary)

        Examples
        --------
        >>> arma = DiscreteARMA11(phi=0.7, theta=0.3, sigma=0.1)
        >>> var = arma.get_stationary_variance()
        >>> print(f"Stationary variance: {var:.4f}")
        """
        phi = self.parameters[sp.symbols('phi')]
        theta = self.parameters[sp.symbols('theta')]
        sigma = self.parameters[sp.symbols('sigma')]

        if abs(phi) >= 1:
            raise ValueError(
                f"Stationary variance undefined for |phi| = {abs(phi)} >= 1"
            )

        numerator = 1 + 2*theta*phi + theta**2
        denominator = 1 - phi**2

        return sigma**2 * numerator / denominator

    def get_acf_lag1(self) -> float:
        """
        Get theoretical autocorrelation at lag 1.

        ρ(1) = (φ + θ)·(1 + θφ)/(1 + 2θφ + θ²)

        Returns
        -------
        float
            Lag-1 autocorrelation

        Notes
        -----
        This is where ARMA differs from AR(1).
        For AR(1): ρ(1) = φ
        For ARMA(1,1): ρ(1) ≠ φ (MA effect)

        Examples
        --------
        >>> arma = DiscreteARMA11(phi=0.7, theta=0.3, sigma=0.1)
        >>> rho1 = arma.get_acf_lag1()
        >>> print(f"ρ(1) = {rho1:.3f}")
        >>> print(f"Compare with φ = 0.7")
        """
        phi = self.parameters[sp.symbols('phi')]
        theta = self.parameters[sp.symbols('theta')]

        numerator = (phi + theta) * (1 + theta*phi)
        denominator = 1 + 2*theta*phi + theta**2

        return numerator / denominator

    def get_acf(self, max_lag: int = 20) -> np.ndarray:
        """
        Get theoretical autocorrelation function.

        Parameters
        ----------
        max_lag : int, default=20
            Maximum lag

        Returns
        -------
        np.ndarray
            ACF values [ρ(0), ρ(1), ..., ρ(max_lag)]

        Notes
        -----
        ρ(0) = 1
        ρ(1) = (φ + θ)·(1 + θφ)/(1 + 2θφ + θ²)
        ρ(h) = φ·ρ(h-1) for h ≥ 2

        Geometric decay after lag 1.

        Examples
        --------
        >>> arma = DiscreteARMA11(phi=0.7, theta=0.3, sigma=0.1)
        >>> acf = arma.get_acf(max_lag=10)
        >>> print(f"ACF: {acf}")
        """
        phi = self.parameters[sp.symbols('phi')]

        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0
        acf[1] = self.get_acf_lag1()

        # Geometric decay for h ≥ 2
        for h in range(2, max_lag + 1):
            acf[h] = phi * acf[h-1]

        return acf


# Convenience functions
def create_economic_arma(
    persistence: str = 'moderate',
    smoothing: str = 'moderate',
) -> DiscreteARMA11:
    """
    Create ARMA(1,1) for economic time series.

    Parameters
    ----------
    persistence : str, default='moderate'
        'low', 'moderate', or 'high'
    smoothing : str, default='moderate'
        'low', 'moderate', or 'high'

    Returns
    -------
    DiscreteARMA11

    Notes
    -----
    Typical economic data:
    - GDP, inflation: High persistence, moderate smoothing
    - Interest rates: High persistence, low smoothing
    - Employment: Moderate persistence, high smoothing

    Examples
    --------
    >>> # GDP growth (high persistence)
    >>> gdp = create_economic_arma(
    ...     persistence='high',
    ...     smoothing='moderate'
    ... )
    """
    persistence_map = {'low': 0.3, 'moderate': 0.6, 'high': 0.9}
    smoothing_map = {'low': 0.1, 'moderate': 0.3, 'high': 0.5}

    phi = persistence_map.get(persistence, 0.6)
    theta = smoothing_map.get(smoothing, 0.3)

    return DiscreteARMA11(phi=phi, theta=theta, sigma=0.1, dt=1.0)


def create_sensor_arma(
    measurement_smoothing: float = 0.4,
    dynamics_persistence: float = 0.7,
) -> DiscreteARMA11:
    """
    Create ARMA(1,1) for sensor data with smoothing.

    Parameters
    ----------
    measurement_smoothing : float, default=0.4
        MA coefficient (sensor averaging)
    dynamics_persistence : float, default=0.7
        AR coefficient (physical dynamics)

    Returns
    -------
    DiscreteARMA11

    Notes
    -----
    Physical system with sensor smoothing naturally creates ARMA:
    - AR: True dynamics persistence
    - MA: Sensor averaging/filtering

    Examples
    --------
    >>> # Temperature sensor with low-pass filter
    >>> temp_sensor = create_sensor_arma(
    ...     measurement_smoothing=0.3,
    ...     dynamics_persistence=0.8
    ... )
    """
    return DiscreteARMA11(
        phi=dynamics_persistence,
        theta=measurement_smoothing,
        sigma=0.05,
        dt=1.0
    )