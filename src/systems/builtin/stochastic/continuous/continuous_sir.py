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
Stochastic SIR Model - Epidemic Dynamics with Demographic Stochasticity
========================================================================

This module provides the stochastic Susceptible-Infected-Recovered (SIR) model,
the fundamental framework for understanding disease spread under uncertainty.
The stochastic SIR model serves as:

- The canonical model for epidemic dynamics with demographic noise
- A framework for understanding outbreak extinction vs persistence
- A benchmark for stochastic population dynamics and rare event analysis
- A test case for parameter estimation from noisy epidemic data
- An illustration of how discreteness (individual events) creates randomness

The stochastic SIR extends the deterministic epidemic model by including
demographic stochasticity - the inherent randomness from discrete infection
and recovery events in finite populations. This creates phenomena impossible
in deterministic models:
- Small outbreaks can randomly die out (stochastic extinction)
- Large outbreaks have uncertain timing and peak size
- Threshold behavior becomes probabilistic (not deterministic)
- Rare superspreading events can dominate dynamics

Physical Context
----------------

**Kermack-McKendrick Model (1927):**

The deterministic SIR model divides population N into:
- S(t): Susceptible (can be infected)
- I(t): Infected (currently infectious)
- R(t): Recovered (immune, no longer susceptible)

**Deterministic ODEs:**
    dS/dt = -β·S·I/N
    dI/dt = β·S·I/N - γ·I
    dR/dt = γ·I

where:
- β: Transmission rate (contacts per time × transmission probability)
- γ: Recovery rate (1/γ = average infectious period)
- S + I + R = N (conservation)

**Why Stochastic Extension Needed:**

Real epidemics exhibit randomness from:

1. **Demographic Stochasticity:**
   - Infection events are discrete (integer people)
   - Small numbers: Poisson fluctuations
   - Each event is random (contact with infected)

2. **Individual Variability:**
   - Heterogeneous infectiousness (superspreaders)
   - Variable recovery times
   - Contact network structure

3. **Environmental Fluctuations:**
   - Seasonal variation (temperature, behavior)
   - Policy changes (interventions)
   - Mobility patterns

4. **Measurement Noise:**
   - Underreporting (asymptomatic cases)
   - Testing capacity limits
   - Reporting delays

**Small Population Effects:**

Deterministic valid when N → ∞ (Law of Large Numbers).

For finite N:
- Fluctuations scale as √N (Central Limit Theorem)
- Relative fluctuations: 1/√N
- Small populations: Large relative fluctuations

**Example:**
- N = 1,000,000: Fluctuations ~0.1% (deterministic good approximation)
- N = 10,000: Fluctuations ~1%
- N = 100: Fluctuations ~10% (stochastic essential)

Mathematical Formulation
------------------------

**Continuous-Time Stochastic SIR:**

Diffusion approximation of discrete Markov chain:
    dS = -β·S·I/N·dt + √(β·S·I/N)·dW_S
    dI = (β·S·I/N - γ·I)·dt + √(β·S·I/N + γ·I)·dW_I
    dR = γ·I·dt + √(γ·I)·dW_R

where:
- Drift: Deterministic SIR (mean behavior)
- Diffusion: √(rate) from Poisson fluctuations
- W_S, W_I, W_R: Correlated Wiener processes

**Note on Noise Structure:**

Not independent! Constraints:
- Conservation: dS + dI + dR = 0
- Correlations: Infection event decreases S, increases I

This implementation uses **simplified independent noise** (approximate).

**Square-Root Diffusion:**

Similar to CIR process: σ(X) = √X
- Ensures positivity (approximately)
- Noise vanishes as X → 0
- Physical: Poisson counting statistics

**Alternative: Additive Noise (This Implementation):**

For simplicity, use additive noise:
    dS = -β·S·I/N·dt + σ_S·dW_S
    dI = (β·S·I/N - γ·I)·dt + σ_I·dW_I
    dR = γ·I·dt + σ_R·dW_R

Less physical but easier to simulate and analyze.

Stochastic Extinction
----------------------

**Critical Phenomenon:**

Small outbreaks can randomly die out before epidemic threshold.

**Deterministic Threshold:**

Basic reproduction number R₀ = β/γ
- R₀ < 1: Disease dies out (subcritical)
- R₀ > 1: Epidemic occurs (supercritical)

**Stochastic Reality:**

Even if R₀ > 1, outbreak can die out by chance:
- Early infections: I small (few infected individuals)
- Random recovery before transmission
- Probability of extinction: P_ext(I₀) > 0

**Extinction Probability:**

For outbreak starting with I₀ infected:
    P_ext ≈ (1/R₀)^{I₀} if R₀ > 1

**Examples:**
- R₀ = 2, I₀ = 1: P_ext = 0.5 (50% die out!)
- R₀ = 2, I₀ = 10: P_ext ≈ 0.001 (0.1%)
- R₀ = 5, I₀ = 1: P_ext = 0.2 (20%)

**Implications:**

1. **Early Intervention Critical:**
   Small I₀ → higher extinction probability
   Quarantine early infections maximizes stochastic extinction

2. **Superspreader Events:**
   One individual → I₀ = 10-100 (extinction unlikely)

3. **Import Risk:**
   Travel introduces I₀ = 1 → often dies out naturally

**Time to Extinction:**

For R₀ < 1 (subcritical), mean time:
    E[τ_ext] ≈ ln(N)/(γ(1 - R₀))

Logarithmic in population (fast extinction).

Epidemic Variability
--------------------

**Final Size Distribution:**

For R₀ > 1, final number infected R(∞) is random.

Deterministic: R_∞ = N - S₀·exp(-R₀·R_∞/N)

Stochastic: Distribution around deterministic prediction.
- Mean ≈ deterministic
- Variance from demographic noise
- Can be bimodal (extinction vs major outbreak)

**Peak Timing and Size:**

Time to peak t_peak and peak infections I_max are random:
- Deterministic: Single values
- Stochastic: Distributions (sometimes wide)
- Early extinction → no peak (bimodal)

**Duration Uncertainty:**

Outbreak duration τ_outbreak random:
- Fast die-out: τ small (extinction)
- Major outbreak: τ ~ (1/γ)·ln(N/I₀)
- Variance: Considerable (factor 2-3)

Basic Reproduction Number R₀
------------------------------

**Definition:**
    R₀ = β/γ

Average number of secondary infections from one infected individual
in fully susceptible population.

**Interpretation:**
- R₀ < 1: Each infected infects <1 others → dies out
- R₀ = 1: Critical (boundary)
- R₀ > 1: Each infected infects >1 others → epidemic

**Examples (Real Diseases):**
- Measles: R₀ = 12-18 (highly contagious)
- COVID-19 (original): R₀ = 2-3
- Influenza: R₀ = 1-2
- Ebola: R₀ = 1.5-2.5

**Herd Immunity Threshold:**
    H = 1 - 1/R₀

Fraction immune needed to prevent epidemic.
- Measles (R₀=15): H = 93% (need high vaccination)
- COVID (R₀=3): H = 67%

**With Stochasticity:**
R₀ becomes probabilistic threshold - even R₀ > 1 can die out.

Applications
------------

**1. Epidemiology:**

**COVID-19 Modeling:**
- Stochastic early phase (cluster dynamics)
- Parameter uncertainty (β, γ not exactly known)
- Intervention timing (when to lock down)
- Outbreak risk (probability of major epidemic)

**Outbreak Investigation:**
- Probability origin in cluster
- Superspreader identification
- Contact tracing effectiveness

**2. Parameter Estimation:**

**Inference from Data:**
- Observed: I(t) with noise
- Estimate: β, γ, I₀
- Methods: MLE, Bayesian (MCMC), particle filter

**3. Intervention Design:**

**Optimal Control:**
- Vaccination strategy: Who to vaccinate first?
- Social distancing: When to implement?
- Testing: Optimal allocation?

**Stochastic Optimal Control:**
- Minimize: E[infections] + cost(intervention)
- Constraints: P(I > I_crit) < ε (ICU capacity)
- Rare event: Avoid overwhelming hospitals

**4. Ecological Models:**

**Host-Pathogen Dynamics:**
SIR framework applies to:
- Wildlife diseases
- Plant pathogens
- Aquatic epidemics

**5. Information Spread:**

**Social Contagion:**
- Memes, rumors, information
- S: Haven't seen
- I: Actively sharing
- R: Lost interest

**6. Cyber Security:**

**Malware Spread:**
- S: Vulnerable computers
- I: Infected (spreading)
- R: Patched/removed

Numerical Integration
---------------------

**Challenges:**

1. **Positivity:**
   - S, I, R ≥ 0 (number of people)
   - Standard SDE integrators can violate
   - Need projection or positivity-preserving schemes

2. **Conservation:**
   - S + I + R = N (constant population)
   - Numerical drift can violate
   - Check conservation frequently

3. **Small I:**
   - Near extinction: I ≈ 0
   - Square-root diffusion √I creates issues
   - Need careful handling

4. **Nonlinearity:**
   - S·I interaction term
   - Coupled dynamics

**Recommended Methods:**

1. **Euler-Maruyama with Projection:**
   - Standard Euler-Maruyama
   - If any state < 0: Set to 0
   - If sum ≠ N: Renormalize

2. **Gillespie Algorithm (Exact):**
   - Discrete stochastic simulation
   - Exact for finite population
   - Gold standard (but discrete jumps, not continuous)

3. **Tau-Leaping:**
   - Approximate Gillespie with larger time steps
   - Poisson approximation

**For This Implementation:**
Euler-Maruyama with dt ~ 0.01-0.1 days.

Common Pitfalls
---------------

1. **Wrong Noise Scaling:**
   - Physical: σ ~ √(rate) from Poisson
   - This implementation: Additive (simplified)
   - For accurate: Use Gillespie or √ diffusion

2. **Conservation Violation:**
   - Noise can cause S+I+R ≠ N
   - Monitor and renormalize if needed

3. **Negativity:**
   - Can get S, I, R < 0
   - Physically impossible
   - Project to non-negative orthant

4. **R₀ Interpretation:**
   - Deterministic threshold not sharp in stochastic
   - Even R₀ > 1 can die out
   - Extinction probability finite

5. **Single Realization:**
   - One outbreak not typical
   - Need ensemble (Monte Carlo)
   - Bimodal: Extinction vs epidemic

6. **Parameter Estimation:**
   - Identifiability issues (β, γ correlated)
   - Need rich data (time series)
   - Bayesian methods handle uncertainty

Historical Context
------------------

**Development Timeline:**

- **1927**: Kermack & McKendrick deterministic SIR
  - Foundation of mathematical epidemiology
  - Threshold theorem (R₀)

- **1950s-60s**: Bailey introduces stochastic models
  - Branching process approximations
  - Extinction probabilities

- **1970s**: Gillespie algorithm
  - Exact stochastic simulation
  - Chemical kinetics → epidemiology

- **2000s**: Parameter estimation methods
  - Particle filters for epidemic inference
  - MCMC for Bayesian parameter estimation

- **2020**: COVID-19 pandemic
  - Massive application of stochastic SIR
  - Real-time parameter estimation
  - Policy evaluation under uncertainty

**Impact:**
Stochastic SIR demonstrated:
- Small populations: Randomness matters
- Extinction: Probabilistic, not deterministic
- Early intervention: Exploits stochastic extinction
- Forecasting: Must include uncertainty

"""

import numpy as np
import sympy as sp
from typing import Optional

from src.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem


class StochasticSIR(ContinuousStochasticSystem):
    """
    Stochastic SIR epidemic model with demographic noise.

    Models disease spread in a finite population with inherent randomness
    from discrete infection and recovery events. Essential for understanding
    outbreak extinction, variability in epidemic outcomes, and uncertainty
    in disease forecasting.

    Stochastic Differential Equations
    ----------------------------------
    Diffusion approximation of discrete SIR:

        dS = -β·S·I/N·dt + σ_S·dW_S
        dI = (β·S·I/N - γ·I)·dt + σ_I·dW_I
        dR = γ·I·dt + σ_R·dW_R

    where:
        - S(t): Number susceptible (can be infected)
        - I(t): Number infected (infectious)
        - R(t): Number recovered (immune)
        - N = S + I + R: Total population (constant)
        - β: Transmission rate [1/time]
        - γ: Recovery rate [1/time]
        - σ_S, σ_I, σ_R: Noise intensities
        - W_S, W_I, W_R: Wiener processes

    **Physical Meaning:**

    Transmission: β·S·I/N
    - Rate of new infections
    - Mass action: Proportional to S·I
    - β: Contacts per time × transmission probability

    Recovery: γ·I
    - Rate of recoveries
    - First-order: Proportional to I
    - γ = 1/(infectious period)

    **Demographic Noise:**

    True stochastic SIR has √ diffusion:
        Diffusion ~ √(rate)

    From Poisson statistics of discrete events.

    This implementation: Simplified additive noise for tractability.

    Physical Interpretation
    -----------------------
    **Susceptibles S:**

    Decrease only (monotonic):
    - Start: S(0) ≈ N (nearly all susceptible)
    - End: S(∞) > 0 (some escape infection)
    - Never increases (no loss of immunity modeled)

    **Infected I:**

    Non-monotonic (rise then fall):
    - Start: I(0) = small (index cases)
    - Peak: I_max at t_peak (outbreak peak)
    - End: I(∞) = 0 (disease dies out)

    **Recovered R:**

    Increase only (monotonic):
    - Start: R(0) = 0 (no immunity initially)
    - End: R(∞) = final outbreak size
    - Measure of epidemic impact

    **Conservation:**
    S + I + R = N (total population constant)

    Key Features
    ------------
    **Nonlinearity:**
    S·I term creates threshold behavior and epidemic curve.

    **Positivity:**
    Must have S, I, R ≥ 0 (counts of people).

    **Conservation:**
    S + I + R = N always (no births/deaths).

    **Extinction:**
    I → 0 eventually (disease dies out).
    Time random, final size random.

    **Threshold (R₀):**
    Probabilistic in stochastic model.

    **Finite Time:**
    Epidemic is transient (not steady state).

    Mathematical Properties
    -----------------------
    **Basic Reproduction Number:**
        R₀ = β/γ

    Critical threshold (deterministic):
    - R₀ < 1: Dies out
    - R₀ > 1: Epidemic

    **Stochastic Threshold:**
    Even R₀ > 1: Can die out with probability (1/R₀)^{I₀}

    **Final Size Relation:**

    Deterministic: R_∞ satisfies
        R_∞ = N - S₀·exp(-R₀·R_∞/N)

    Stochastic: Distribution around this value.

    **Epidemic Peak:**

    Deterministic:
        I_max ≈ I₀ + S₀ - N/R₀ - (N/R₀)·ln(S₀·R₀/N)
        t_peak ≈ (1/γ)·ln(R₀·S₀/N)

    Stochastic: Random variables (compute via simulation).

    Physical Interpretation
    -----------------------
    **Transmission Rate β:**
    - Units: [1/time]
    - β = contact rate × transmission probability
    - Typical: 0.2-2.0 per day

    **Examples:**
    - Influenza: β ≈ 0.5 per day
    - Measles: β ≈ 1.5 per day
    - COVID-19: β ≈ 0.3-0.6 per day (varies)

    **Recovery Rate γ:**
    - Units: [1/time]
    - γ = 1/(infectious period)
    - Typical: 0.1-1.0 per day

    **Examples:**
    - Influenza: γ ≈ 0.5 per day (2 days infectious)
    - COVID-19: γ ≈ 0.1 per day (10 days)
    - Measles: γ ≈ 0.1 per day (10 days)

    **Basic Reproduction Number:**
        R₀ = β/γ

    **Examples:**
    - Influenza: R₀ ≈ 1-2
    - COVID-19: R₀ ≈ 2-5 (variant dependent)
    - Measles: R₀ ≈ 12-18

    **Noise Intensity:**

    From demographic stochasticity (Poisson):
        σ ≈ √(rate/N)

    Relative noise: 1/√N

    State Space
    -----------
    State: X = [S, I, R] ∈ ℝ₊³
        - S, I, R ≥ 0 (non-negative counts)
        - S + I + R = N (conservation)
        - Bounded: 0 ≤ S, I, R ≤ N

    Control: u (optional, interventions)
        - Reduce β (social distancing)
        - Increase recovery (treatment)
        - Vaccination (move S → R)

    Noise: w = [w_S, w_I, w_R] ∈ ℝ³
        - Demographic stochasticity
        - Should be correlated (conservation)
        - This implementation: Simplified independent

    Parameters
    ----------
    beta : float, default=0.5
        Transmission rate [1/day]
        - Typical: 0.2-2.0
        - Disease and behavior dependent

    gamma : float, default=0.1
        Recovery rate [1/day]
        - 1/γ = infectious period
        - Typical: 0.1-1.0

    N : float, default=1000.0
        Total population
        - Larger N: More deterministic
        - Smaller N: More stochastic

    sigma_S : float, default=0.1
        Susceptible noise intensity [1/√day]
        - From √(rate/N) for demographic noise
        - Typical: √(β·S·I/N²)

    sigma_I : float, default=0.1
        Infected noise intensity [1/√day]

    sigma_R : float, default=0.1
        Recovered noise intensity [1/√day]

    Stochastic Properties
    ---------------------
    - System Type: NONLINEAR
    - Noise Type: ADDITIVE (simplified)
    - SDE Type: Itô
    - Noise Dimension: nw = 3
    - Stationary: No (epidemic is transient)
    - Positive: Should be (may need projection)
    - Conserved: S + I + R = N (approximately)

    Applications
    ------------
    **1. Epidemiology:**
    - COVID-19, influenza, measles modeling
    - Outbreak prediction with uncertainty
    - Intervention timing and effectiveness

    **2. Parameter Estimation:**
    - Estimate β, γ from outbreak data
    - Bayesian inference with uncertainty
    - Real-time estimation (particle filter)

    **3. Public Health Policy:**
    - Vaccination strategies
    - Social distancing timing
    - Resource allocation (ICU beds)

    **4. Extinction Analysis:**
    - Probability small outbreak dies out
    - Early intervention effectiveness
    - Import risk assessment

    **5. Rare Events:**
    - Superspreading events
    - Large outbreaks in small populations
    - Timing of peak (ICU planning)

    Numerical Integration
    ---------------------
    **Recommended:**
    - Euler-Maruyama: dt = 0.01-0.1 days
    - Project to non-negative: max(X, 0)
    - Check conservation: S+I+R ≈ N

    **Event Detection:**
    - Extinction: I < 0.5 (declare extinct)
    - Peak: Max I(t)
    - Duration: Time I > threshold

    Monte Carlo Guidelines
    -----------------------
    **Ensemble Analysis:**
    - N_runs = 100-1,000
    - Compute: Extinction probability, mean final size
    - Histogram: Final R (bimodal if near threshold)

    Comparison with Deterministic
    ------------------------------
    **Deterministic:**
    - Smooth epidemic curve
    - Single final size
    - R₀ sharp threshold

    **Stochastic:**
    - Variable epidemic curves
    - Distribution of final sizes
    - R₀ probabilistic threshold
    - Extinction possible

    Limitations
    -----------
    - Additive noise (not √ diffusion)
    - Independent noise (should be correlated)
    - Homogeneous mixing (no network structure)
    - Constant β, γ (no seasonality)
    - Closed population (no births/deaths)

    Extensions
    ----------
    - SEIR: Add Exposed class
    - Age structure: Multiple age groups
    - Spatial: Geographic spread
    - Network: Contact structure
    - Time-varying: Seasonal β(t)
    - Vaccination: Control via S → R
    See Also
    --------
    CoxIngersollRoss : Similar √ diffusion structure
    """

    def define_system(
        self,
        beta: float = 0.5,
        gamma: float = 0.1,
        N: float = 1000.0,
        sigma_S: float = 0.1,
        sigma_I: float = 0.1,
        sigma_R: float = 0.1,
    ):
        """
        Define stochastic SIR epidemic dynamics.

        Parameters
        ----------
        beta : float, default=0.5
            Transmission rate [1/day]
            - β = contact rate × transmission probability
            - Typical: 0.2-2.0 per day
            - Higher → faster spread

        gamma : float, default=0.1
            Recovery rate [1/day]
            - γ = 1/(infectious period)
            - Typical: 0.1-1.0 per day
            - Higher → faster recovery

        N : float, default=1000.0
            Total population (constant)
            - Larger N: More deterministic
            - Smaller N: More stochastic
            - Typical: 100-1,000,000

        sigma_S : float, default=0.1
            Susceptible noise intensity [1/√day]
            - From demographic noise: ~ √(β·S·I/N²)
            - Typical: 0.01-1.0

        sigma_I : float, default=0.1
            Infected noise intensity [1/√day]
            - From demographic noise: ~ √(β·S·I/N² + γ·I/N)
            - Most critical (affects outbreak dynamics)

        sigma_R : float, default=0.1
            Recovered noise intensity [1/√day]
            - From demographic noise: ~ √(γ·I/N)

        Notes
        -----
        **Basic Reproduction Number:**
            R₀ = β/γ

        Determines epidemic threshold:
        - R₀ < 1: Dies out (subcritical)
        - R₀ = 1: Critical (boundary)
        - R₀ > 1: Epidemic (supercritical)

        **Stochastic Extinction:**

        Even if R₀ > 1, can randomly die out:
            P(extinction | I₀) ≈ (1/R₀)^{I₀}

        **Example:** R₀ = 2, I₀ = 1 → P_ext = 50%

        **Noise Scaling:**

        Physical demographic noise:
            σ_physical ~ √(rate/N)

        For N = 1000, typical rates ~ 0.1-1:
            σ ~ 0.01-0.1

        Larger population → smaller relative noise.

        **Infectious Period:**
            T_inf = 1/γ

        **Examples:**
        - Influenza: 2-3 days (γ ≈ 0.4)
        - COVID-19: 7-14 days (γ ≈ 0.1)
        - Measles: 7-10 days (γ ≈ 0.12)

        **Herd Immunity:**
            H = 1 - 1/R₀

        Fraction immune needed to prevent epidemic.

        **Validation:**

        Check conservation: S + I + R should equal N.
        If drifts: Renormalize periodically.
        """
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if N <= 0:
            raise ValueError(f"N must be positive, got {N}")

        # Store population
        self.N = N
        self.R0 = beta / gamma

        # State variables
        S, I, R = sp.symbols("S I R", real=True, nonnegative=True)
        
        # No control in basic SIR (autonomous)
        self.control_vars = []

        # Parameters
        beta_sym = sp.symbols("beta", positive=True)
        gamma_sym = sp.symbols("gamma", positive=True)
        N_sym = sp.symbols("N", positive=True)
        sigma_S_sym = sp.symbols("sigma_S", nonnegative=True)
        sigma_I_sym = sp.symbols("sigma_I", nonnegative=True)
        sigma_R_sym = sp.symbols("sigma_R", nonnegative=True)

        self.state_vars = [S, I, R]

        # DRIFT (Deterministic SIR equations)
        transmission = beta_sym * S * I / N_sym
        recovery = gamma_sym * I

        self._f_sym = sp.Matrix([
            -transmission,           # dS/dt
            transmission - recovery, # dI/dt
            recovery                 # dR/dt
        ])

        self.parameters = {
            beta_sym: beta,
            gamma_sym: gamma,
            N_sym: N,
            sigma_S_sym: sigma_S,
            sigma_I_sym: sigma_I,
            sigma_R_sym: sigma_R,
        }
        self.order = 1

        # DIFFUSION (Demographic noise - simplified additive)
        # Physical: Should be √(rate) but additive for simplicity
        self.diffusion_expr = sp.Matrix([
            [sigma_S_sym, 0, 0],
            [0, sigma_I_sym, 0],
            [0, 0, sigma_R_sym]
        ])

        # Itô SDE
        self.sde_type = "ito"

        # Output: Infected count (what's typically measured)
        self._h_sym = sp.Matrix([I])

    def get_basic_reproduction_number(self) -> float:
        """
        Get basic reproduction number R₀ = β/γ.

        Returns
        -------
        float
            R₀

        Examples
        --------
        >>> sir = StochasticSIR(beta=0.5, gamma=0.1)
        >>> R0 = sir.get_basic_reproduction_number()
        >>> print(f"R₀ = {R0:.2f}")
        """
        return self.R0

    def get_herd_immunity_threshold(self) -> float:
        """
        Get herd immunity threshold H = 1 - 1/R₀.

        Returns
        -------
        float
            Herd immunity threshold (fraction)

        Examples
        --------
        >>> sir = StochasticSIR(beta=0.5, gamma=0.1)
        >>> H = sir.get_herd_immunity_threshold()
        >>> print(f"Herd immunity: {H:.1%}")
        """
        if self.R0 <= 1:
            return 0.0
        return 1.0 - 1.0/self.R0

    def estimate_extinction_probability(self, I0: float) -> float:
        """
        Estimate probability of stochastic extinction (approximation).

        For R₀ > 1: P_ext ≈ (1/R₀)^{I₀}

        Parameters
        ----------
        I0 : float
            Initial number infected

        Returns
        -------
        float
            Extinction probability

        Examples
        --------
        >>> sir = StochasticSIR(beta=0.6, gamma=0.2)  # R₀=3
        >>> P_ext_1 = sir.estimate_extinction_probability(I0=1)
        >>> P_ext_10 = sir.estimate_extinction_probability(I0=10)
        >>> print(f"1 infected: P_ext = {P_ext_1:.2%}")
        >>> print(f"10 infected: P_ext = {P_ext_10:.2%}")
        """
        if self.R0 <= 1:
            return 1.0  # Always dies out if subcritical
        
        return (1.0 / self.R0) ** I0

    def check_conservation(self, x: np.ndarray) -> float:
        """
        Check conservation constraint S + I + R = N.

        Parameters
        ----------
        x : np.ndarray
            State [S, I, R]

        Returns
        -------
        float
            Absolute error |S+I+R - N|

        Examples
        --------
        >>> sir = StochasticSIR(N=1000)
        >>> x = np.array([500, 200, 300])
        >>> error = sir.check_conservation(x)
        >>> print(f"Conservation error: {error:.2f}")
        """
        return np.abs(np.sum(x) - self.N)


# Convenience function
def create_epidemic_model(
    disease: str = 'covid',
    population: float = 10000,
    noise_level: str = 'medium',
) -> StochasticSIR:
    """
    Create SIR model for common diseases.

    Parameters
    ----------
    disease : str, default='covid'
        'influenza', 'covid', 'measles'
    population : float, default=10000
        Total population
    noise_level : str, default='medium'
        'low', 'medium', or 'high'

    Returns
    -------
    StochasticSIR

    Examples
    --------
    >>> # COVID-19 in medium population
    >>> covid = create_epidemic_model('covid', population=10000)
    >>> 
    >>> # Influenza in small population
    >>> flu = create_epidemic_model('influenza', population=1000)
    """
    disease_params = {
        'influenza': {'beta': 0.4, 'gamma': 0.5},  # R₀ ≈ 0.8
        'covid': {'beta': 0.3, 'gamma': 0.1},      # R₀ ≈ 3
        'measles': {'beta': 1.8, 'gamma': 0.12},   # R₀ ≈ 15
    }
    
    noise_presets = {
        'low': 0.01,
        'medium': 0.05,
        'high': 0.2,
    }
    
    params = disease_params.get(disease, disease_params['covid'])
    sigma = noise_presets.get(noise_level, 0.05)
    
    # Scale noise with population (demographic: ~1/√N)
    sigma_scaled = sigma * np.sqrt(1000 / population)
    
    return StochasticSIR(
        beta=params['beta'],
        gamma=params['gamma'],
        N=population,
        sigma_S=sigma_scaled,
        sigma_I=sigma_scaled,
        sigma_R=sigma_scaled,
    )