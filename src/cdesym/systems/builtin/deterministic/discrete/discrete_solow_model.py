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
Discrete-time Solow Growth Model - Macroeconomic Dynamics.

This module provides discrete-time implementations of the Solow-Swan growth model,
a foundational framework in macroeconomics for understanding long-run economic growth.
It serves as:
- The canonical model of capital accumulation and economic growth
- A benchmark for understanding steady-state convergence
- An illustration of diminishing returns and balanced growth
- A testbed for economic policy analysis
- A demonstration of nonlinear difference equations in economics

The Solow model represents:
- National economies and GDP growth
- Savings, investment, and capital formation
- Technological progress and productivity
- Population growth effects on per-capita income
- The transition from developing to developed economies

This is the discrete-time version, appropriate for:
- Annual economic data (yearly updates)
- Discrete-time policy interventions
- Numerical simulation and prediction
- Integration with discrete-time economic datasets
"""

from typing import Optional, Tuple

import numpy as np
import sympy as sp

from cdesym.systems.base.core.discrete_symbolic_system import DiscreteSymbolicSystem


class DiscreteSolowModel(DiscreteSymbolicSystem):
    """
    Discrete-time Solow-Swan neoclassical growth model.

    Physical System (Economic System):
    ----------------------------------
    The Solow model describes how an economy's capital stock evolves over time
    through the interplay of:
    - **Investment:** Savings channeled into new capital
    - **Depreciation:** Capital wears out and becomes obsolete
    - **Population growth:** Dilutes per-capita capital
    - **Technological progress:** Increases productivity

    **The Central Equation (Continuous Time):**
        k̇ = s·f(k) - (δ + n)·k

    where:
        k: Capital per worker (capital intensity)
        s: Savings rate (fraction of output saved)
        f(k): Production function (output per worker)
        δ: Depreciation rate
        n: Population growth rate

    **Production Function (Cobb-Douglas):**
    The standard form is:
        y = f(k) = A·k^α

    where:
        y: Output per worker (GDP per capita)
        A: Total factor productivity (TFP), technology level
        k: Capital per worker
        α: Output elasticity of capital (0 < α < 1)

    **Key Properties:**
    1. **Positive but diminishing marginal product:**
       f'(k) > 0, f''(k) < 0
       - More capital → more output
       - But each additional unit contributes less (diminishing returns)

    2. **Inada conditions:**
       f(0) = 0, f'(0) = ∞, f'(∞) = 0
       - No capital → no output
       - Initially very productive
       - Eventually saturates

    3. **Constant returns to scale (with labor):**
       F(K, L) = A·K^α·L^(1-α)
       - Doubling both K and L doubles output

    **Discrete-Time Dynamics:**
    The discrete version models year-to-year changes:

        k[t+1] = (1 - δ)·k[t] + s·f(k[t])

    Or with population growth:
        k[t+1] = [(1 - δ)·k[t] + s·f(k[t])] / (1 + n)

    State Space:
    -----------
    State: x[t] = [k[t]]
        Capital per worker:
        - k: Capital intensity [$/worker] or [capital units per worker]
          * k ≥ 0: Non-negative by definition
          * k = 0: No capital (subsistence economy)
          * k → k*: Converges to steady state
          * Typical range: 0-500 for normalized units

    **Physical/Economic Meaning:**
    Capital per worker represents:
    - Machines, equipment, tools available per person
    - Infrastructure (roads, buildings) per capita
    - Technology embodied in physical capital
    - Higher k → higher productivity → higher income

    Control: u[t] (optional, for policy intervention)
        - Can represent:
          * Temporary savings rate change: s → s + u[t]
          * Government investment/stimulus
          * Technology adoption rate change
          * Foreign aid or capital injection
        - Standard model: u = 0 (no intervention)

    Output: y[t] = [k[t]] or [k[t], y[t], c[t]]
        Standard outputs:
        - k[t]: Capital per worker (state variable)
        - y[t]: Output per worker (GDP per capita)
        - c[t]: Consumption per worker

        Where:
            y[t] = A·k[t]^α
            c[t] = (1 - s)·y[t]

    Dynamics (Economic Interpretation):
    -----------------------------------
    The discrete dynamics:
        k[t+1] = (1 - δ)·k[t] + s·A·k[t]^α

    **Capital Evolution Has Three Components:**

    1. **Undepreciated capital: (1 - δ)·k[t]**
       - Fraction (1 - δ) survives to next period
       - δ represents wearing out, obsolescence
       - If δ = 0.05: 5% of capital lost each year

    2. **New investment: s·f(k[t])**
       - Savings rate s fraction of output is invested
       - f(k) = A·k^α is output per worker
       - Higher s → more investment → faster growth

    3. **Population dilution (if n > 0):**
       - Growing population dilutes per-capita capital
       - k[t+1] must be divided by (1 + n)
       - Higher n → slower per-capita growth

    **Steady State (Golden Rule):**
    At steady state: k[t+1] = k[t] = k*

        (1 - δ)·k* + s·A·(k*)^α = k*
        s·A·(k*)^α = δ·k*
        k* = (s·A/δ)^(1/(1-α))

    The steady-state capital is unique and globally stable (under standard
    assumptions).

    **Convergence to Steady State:**
    - If k[0] < k*: Capital accumulates (growth)
    - If k[0] > k*: Capital decumulates (decline)
    - Convergence is monotonic (no oscillations for standard parameters)
    - Speed of convergence: Faster when far from k*, slower when near

    Parameters:
    ----------
    s : float, default=0.3
        Savings rate (fraction of output saved/invested)
        - Must satisfy: 0 < s < 1
        - Typical values: 0.15-0.40
        - Developed countries: ~0.20-0.25
        - Developing countries: ~0.30-0.40
        - China (historically): ~0.45-0.50

        **Economic Meaning:**
        - Higher s → more investment → higher steady-state k*
        - But also lower consumption (1-s)·y
        - Golden rule: s that maximizes steady-state consumption

    delta : float, default=0.05
        Depreciation rate (fraction of capital lost per period)
        - Must satisfy: 0 < δ < 1
        - Typical annual values: 0.03-0.10
        - Physical capital: ~0.05-0.08
        - Structures: ~0.03
        - Equipment: ~0.10

        **Economic Meaning:**
        - Higher δ → more replacement needed → lower k*
        - Represents obsolescence, wear and tear

    alpha : float, default=0.33
        Capital share of income (output elasticity of capital)
        - Must satisfy: 0 < α < 1
        - Typical values: 0.25-0.40
        - Empirically: ~0.33 (one-third)
        - Labor share: 1 - α ≈ 0.67 (two-thirds)

        **Economic Meaning:**
        - Measures importance of capital vs labor
        - α = 0.5: Equal importance (Cobb-Douglas symmetric)
        - α < 0.5: Labor more important
        - α > 0.5: Capital more important (rare)

    A : float, default=1.0
        Total Factor Productivity (TFP) / technology level
        - Must satisfy: A > 0
        - Captures: Technology, institutions, efficiency
        - Grows over time via technological progress
        - Typical growth: 1-2% per year

        **Economic Meaning:**
        - Higher A → more output for same k
        - Represents "technological level"
        - Solow residual: Growth not explained by capital/labor

    n : float, default=0.0
        Population/labor force growth rate
        - Can be positive, zero, or negative
        - Typical values: -0.01 to 0.03 (annually)
        - Developed countries: ~0.00-0.01
        - Developing countries: ~0.01-0.03

        **Economic Meaning:**
        - Higher n → dilutes per-capita capital
        - Must invest more just to maintain k
        - Effective depreciation: δ + n

    dt : float, default=1.0
        Time period (typically 1 year for macroeconomic data)
        - Units: years
        - dt = 1: Annual model (most common)
        - dt = 0.25: Quarterly model

    include_population_growth : bool, default=False
        If True, include population growth in dynamics
        If False, assume n = 0

    use_technology_growth : bool, default=False
        If True, allow A to grow over time
        If False, A is constant

    Equilibria:
    ----------
    **Steady State (k*):**
    The unique non-zero equilibrium where capital per worker is constant:

        k* = (s·A/δ)^(1/(1-α))

    At steady state:
    - Capital accumulation exactly balances depreciation
    - Investment = Depreciation: s·y* = δ·k*
    - No growth in per-capita variables (k, y, c constant)
    - But total output grows at rate n (population growth)

    **Stability:**
    The steady state is GLOBALLY ASYMPTOTICALLY STABLE:
    - For ANY initial k[0] > 0, system converges to k*
    - Convergence is monotonic (no oscillations)
    - Speed: |λ| = (1 - δ) + s·α·A·(k*)^(α-1) < 1

    **Output at Steady State:**
        y* = A·(k*)^α = A^(1/(1-α))·(s/δ)^(α/(1-α))

    **Consumption at Steady State:**
        c* = (1 - s)·y*

    **Golden Rule Capital Level:**
    The capital level that maximizes steady-state consumption:
        k_gold = (α·A/δ)^(1/(1-α))

    Golden rule savings rate:
        s_gold = α

    If s = α, the economy is at the golden rule (max consumption).
    If s > α, the economy is "over-saving" (dynamic inefficiency).
    If s < α, the economy could save more to increase consumption.

    Convergence Dynamics:
    --------------------
    **Conditional Convergence:**
    Countries with same parameters (s, δ, α, A) converge to same k*.
    - Poor countries (low k) grow faster initially
    - Rich countries (high k) grow slower
    - Convergence rate: β = (1 - α)(δ + n)
    - Half-life: t_half = ln(2)/β

    **Absolute Convergence:**
    Empirically NOT observed - poor countries don't always catch up.
    Reasons:
    - Different savings rates s
    - Different technology levels A
    - Different institutional quality
    - Different human capital

    **Speed of Convergence:**
    Empirical estimate: β ≈ 0.02 (2% per year)
    - Half-life: ~35 years
    - Slow convergence due to diminishing returns

    Economic Insights:
    -----------------
    **The Fundamental Question:**
    Why are some countries rich and others poor?

    Solow model answers:
    1. **Capital accumulation:** Rich countries have more k
    2. **Savings rates:** Higher s → higher k*
    3. **Technology:** Higher A → higher productivity
    4. **Population:** Lower n → easier to maintain high k

    **The Growth Puzzle:**
    Solow model predicts:
    - Long-run growth ONLY from technological progress (A growing)
    - Capital accumulation alone cannot sustain growth (diminishing returns)
    - Saving more raises level but not long-run growth rate

    **Policy Implications:**
    1. **Increase savings rate s:**
       - Raises steady-state income
       - Temporary growth boost during transition
       - Reduces consumption in short run

    2. **Reduce depreciation δ:**
       - Better infrastructure maintenance
       - Raises k* and y*

    3. **Improve technology A:**
       - Education, R&D, technology adoption
       - Permanent growth effects
       - Most important for long run

    4. **Control population n:**
       - Lower n → higher k* per capita
       - But total output growth slower

    Control Objectives (Policy):
    ----------------------------
    **1. Maximize Steady-State Consumption:**
       Goal: Choose s to maximize c* = (1 - s)·y*
       Solution: Golden rule s* = α

    **2. Convergence Acceleration:**
       Goal: Reach k* faster via temporary policy
       Method: Increase s temporarily, then reduce
       Challenge: Political economy (delayed gratification)

    **3. Poverty Trap Escape:**
       Goal: Overcome low-level equilibrium trap
       Method: Large temporary investment (foreign aid, stimulus)
       Extension: Add threshold effects, multiple equilibria

    **4. Optimal Savings Path:**
       Goal: Maximize discounted utility ∫e^(-ρt)·u(c[t])dt
       Solution: Ramsey-Cass-Koopmans model (extends Solow)

    **5. Technological Catch-Up:**
       Goal: Model technology diffusion from frontier
       Extension: A[t+1] = A[t] + φ(A_frontier - A[t])

    State Constraints:
    -----------------
    **1. Non-negativity: k[t] ≥ 0**
       - Capital cannot be negative
       - Economic necessity
       - k = 0 is absorbing state (poverty trap if no external help)

    **2. Feasibility: k[t] ≤ k_max**
       - Some maximum possible capital
       - Typically not binding (grows without bound theoretically)

    **3. Savings rate bounds: 0 < s < 1**
       - Cannot save more than 100% of output
       - Cannot save negative amount
       - Policy variable, not state constraint

    **4. Non-negative output: y[t] ≥ 0**
       - Automatic if k ≥ 0 and A > 0

    Numerical Considerations:
    ------------------------
    **Stability:**
    The discrete Solow model is always stable (under standard parameters):
    - Eigenvalue at k*: λ = (1 - δ) + s·α·A·(k*)^(α-1)
    - For typical parameters: 0 < λ < 1
    - Monotonic convergence (no oscillations)

    **Accuracy:**
    Discrete vs continuous models differ for large dt:
    - Annual data (dt = 1): Discrete appropriate
    - Continuous time limit as dt → 0

    **Numerical Precision:**
    No special considerations - well-conditioned problem.

    Example Usage:
    -------------
    >>> # Create Solow model with US-like parameters
    >>> model = DiscreteSolowModel(
    ...     s=0.25,      # 25% savings rate
    ...     delta=0.05,  # 5% annual depreciation
    ...     alpha=0.33,  # 1/3 capital share
    ...     A=1.0,       # Normalized technology
    ...     n=0.01,      # 1% population growth
    ...     dt=1.0,      # Annual periods
    ...     include_population_growth=True
    ... )
    >>>
    >>> # Compute steady state
    >>> k_star = model.compute_steady_state()
    >>> y_star = model.compute_output(k_star)
    >>> c_star = model.compute_consumption(k_star)
    >>>
    >>> print(f"Steady-state capital: k* = {k_star:.2f}")
    >>> print(f"Steady-state output: y* = {y_star:.2f}")
    >>> print(f"Steady-state consumption: c* = {c_star:.2f}")
    >>>
    >>> # Check golden rule
    >>> k_gold = model.compute_golden_rule_capital()
    >>> s_gold = model.alpha
    >>> print(f"\nGolden rule capital: k_gold = {k_gold:.2f}")
    >>> print(f"Golden rule savings rate: s_gold = {s_gold:.2f}")
    >>> print(f"Current vs optimal: {'Over-saving' if model.s > s_gold else 'Under-saving'}")
    >>>
    >>> # Simulate convergence from low initial capital (poor country)
    >>> k0_poor = np.array([10.0])  # Low starting capital
    >>> result_poor = model.simulate(
    ...     x0=k0_poor,
    ...     u_sequence=None,
    ...     n_steps=100  # 100 years
    ... )
    >>>
    >>> # Simulate from high initial capital (rich country)
    >>> k0_rich = np.array([100.0])  # High starting capital
    >>> result_rich = model.simulate(
    ...     x0=k0_rich,
    ...     u_sequence=None,
    ...     n_steps=100
    ... )
    >>>
    >>> # Plot convergence
    >>> import plotly.graph_objects as go
    >>> from plotly.subplots import make_subplots
    >>>
    >>> fig = make_subplots(
    ...     rows=2, cols=2,
    ...     subplot_titles=['Capital per Worker', 'Output per Worker',
    ...                     'Growth Rate', 'Consumption per Worker']
    ... )
    >>>
    >>> t = result_poor['time_steps']
    >>>
    >>> # Capital
    >>> fig.add_trace(go.Scatter(x=t, y=result_poor['states'][:, 0],
    ...                          name='Poor', line=dict(color='blue')), row=1, col=1)
    >>> fig.add_trace(go.Scatter(x=t, y=result_rich['states'][:, 0],
    ...                          name='Rich', line=dict(color='red')), row=1, col=1)
    >>> fig.add_hline(y=k_star, line_dash='dash', line_color='green', row=1, col=1)
    >>>
    >>> # Output
    >>> y_poor = model.compute_output(result_poor['states'][:, 0])
    >>> y_rich = model.compute_output(result_rich['states'][:, 0])
    >>> fig.add_trace(go.Scatter(x=t, y=y_poor, showlegend=False,
    ...                          line=dict(color='blue')), row=1, col=2)
    >>> fig.add_trace(go.Scatter(x=t, y=y_rich, showlegend=False,
    ...                          line=dict(color='red')), row=1, col=2)
    >>> fig.add_hline(y=y_star, line_dash='dash', line_color='green', row=1, col=2)
    >>>
    >>> # Growth rate
    >>> growth_poor = np.diff(result_poor['states'][:, 0]) / result_poor['states'][:-1, 0]
    >>> growth_rich = np.diff(result_rich['states'][:, 0]) / result_rich['states'][:-1, 0]
    >>> fig.add_trace(go.Scatter(x=t[:-1], y=growth_poor*100, showlegend=False,
    ...                          line=dict(color='blue')), row=2, col=1)
    >>> fig.add_trace(go.Scatter(x=t[:-1], y=growth_rich*100, showlegend=False,
    ...                          line=dict(color='red')), row=2, col=1)
    >>>
    >>> # Consumption
    >>> c_poor = model.compute_consumption(result_poor['states'][:, 0])
    >>> c_rich = model.compute_consumption(result_rich['states'][:, 0])
    >>> fig.add_trace(go.Scatter(x=t, y=c_poor, showlegend=False,
    ...                          line=dict(color='blue')), row=2, col=2)
    >>> fig.add_trace(go.Scatter(x=t, y=c_rich, showlegend=False,
    ...                          line=dict(color='red')), row=2, col=2)
    >>>
    >>> fig.update_xaxes(title_text='Time [years]')
    >>> fig.update_yaxes(title_text='k', row=1, col=1)
    >>> fig.update_yaxes(title_text='y', row=1, col=2)
    >>> fig.update_yaxes(title_text='Growth [%]', row=2, col=1)
    >>> fig.update_yaxes(title_text='c', row=2, col=2)
    >>> fig.update_layout(height=800, width=1000, title_text='Solow Model: Convergence Dynamics')
    >>> fig.show()
    >>>
    >>> # Phase diagram (Solow diagram)
    >>> fig_phase = model.plot_solow_diagram()
    >>> fig_phase.show()
    >>>
    >>> # Sensitivity analysis: Vary savings rate
    >>> s_values = np.linspace(0.1, 0.5, 20)
    >>> k_stars = []
    >>> y_stars = []
    >>> c_stars = []
    >>>
    >>> for s_test in s_values:
    ...     model_temp = DiscreteSolowModel(
    ...         s=s_test, delta=0.05, alpha=0.33, A=1.0
    ...     )
    ...     k_ss = model_temp.compute_steady_state()
    ...     y_ss = model_temp.compute_output(k_ss)
    ...     c_ss = model_temp.compute_consumption(k_ss)
    ...     k_stars.append(k_ss)
    ...     y_stars.append(y_ss)
    ...     c_stars.append(c_ss)
    >>>
    >>> fig_sens = go.Figure()
    >>> fig_sens.add_trace(go.Scatter(x=s_values, y=k_stars, name='k*'))
    >>> fig_sens.add_trace(go.Scatter(x=s_values, y=y_stars, name='y*'))
    >>> fig_sens.add_trace(go.Scatter(x=s_values, y=c_stars, name='c*'))
    >>> fig_sens.add_vline(x=model.alpha, line_dash='dash',
    ...                    annotation_text='Golden Rule')
    >>> fig_sens.update_layout(
    ...     title='Steady State vs Savings Rate',
    ...     xaxis_title='Savings Rate s',
    ...     yaxis_title='Steady-State Value'
    ... )
    >>> fig_sens.show()
    >>>
    >>> # Simulate policy experiment: Increase savings temporarily
    >>> k0 = np.array([20.0])  # Below steady state
    >>>
    >>> # Baseline: constant s = 0.25
    >>> result_baseline = model.simulate(x0=k0, u_sequence=None, n_steps=50)
    >>>
    >>> # Policy: Increase s to 0.35 for 10 years, then back to 0.25
    >>> def policy_intervention(x, k_time):
    ...     if 10 <= k_time < 20:
    ...         # Increase savings rate temporarily
    ...         model_temp = DiscreteSolowModel(
    ...             s=0.35, delta=model.delta, alpha=model.alpha, A=model.A
    ...         )
    ...         # Compute additional investment
    ...         extra_savings = 0.35 - 0.25
    ...         y_current = model.compute_output(x[0])
    ...         # This requires modifying the system...
    ...         # For now, return zero (standard model doesn't have control)
    ...         return np.array([])
    ...     return np.array([])
    >>>
    >>> # For policy experiments, better to simulate with modified parameters
    >>> # Create model with higher savings for intervention period
    >>> model_high_s = DiscreteSolowModel(s=0.35, delta=0.05, alpha=0.33, A=1.0)
    >>>
    >>> # Simulate in segments
    >>> # Years 0-10: Normal savings
    >>> result_seg1 = model.simulate(x0=k0, u_sequence=None, n_steps=10)
    >>> # Years 10-20: High savings
    >>> k_10 = result_seg1['states'][-1, :]
    >>> result_seg2 = model_high_s.simulate(x0=k_10, u_sequence=None, n_steps=10)
    >>> # Years 20-50: Back to normal
    >>> k_20 = result_seg2['states'][-1, :]
    >>> result_seg3 = model.simulate(x0=k_20, u_sequence=None, n_steps=30)
    >>>
    >>> # Combine results
    >>> k_policy = np.concatenate([
    ...     result_seg1['states'][:, 0],
    ...     result_seg2['states'][1:, 0],
    ...     result_seg3['states'][1:, 0]
    ... ])
    >>>
    >>> fig_policy = go.Figure()
    >>> fig_policy.add_trace(go.Scatter(
    ...     x=np.arange(51),
    ...     y=result_baseline['states'][:, 0],
    ...     name='Baseline (s=0.25)',
    ...     line=dict(color='blue')
    ... ))
    >>> fig_policy.add_trace(go.Scatter(
    ...     x=np.arange(51),
    ...     y=k_policy,
    ...     name='Policy (s=0.35 for years 10-20)',
    ...     line=dict(color='red')
    ... ))
    >>> fig_policy.add_vrect(x0=10, x1=20, fillcolor='gray', opacity=0.2,
    ...                      annotation_text='High Savings')
    >>> fig_policy.update_layout(
    ...     title='Policy Experiment: Temporary Savings Increase',
    ...     xaxis_title='Time [years]',
    ...     yaxis_title='Capital per Worker k'
    ... )
    >>> fig_policy.show()

    Economic Paradoxes and Puzzles:
    -------------------------------
    **1. Solow Paradox (Computer Age):**
    "You can see the computer age everywhere but in the productivity statistics"
    - Massive IT investment in 1980s-90s
    - But productivity growth didn't accelerate much
    - Possible explanations: Mismeasurement, adjustment lags, diminishing returns

    **2. Productivity Puzzle:**
    Why does α ≈ 0.33 (Kaldor fact)?
    - Empirically robust across countries and time
    - Suggests deep structural feature of production
    - Labor share stable despite technological change

    **3. Growth Miracle Puzzle:**
    Some countries grow much faster than Solow predicts:
    - East Asian tigers: 7-10% growth for decades
    - China: 10% growth for 30+ years
    - Solow: Convergence only, not sustained super-growth
    - Answer: Rapid A growth, human capital, institutions

    **4. Poverty Trap:**
    Basic Solow predicts convergence, but some countries stay poor:
    - Extensions: Threshold effects, multiple equilibria
    - S-shaped production function
    - Institutional quality matters

    Common Pitfalls:
    ---------------
    1. **Confusing levels with growth rates:**
       Higher s → higher k* (level effect)
       But NOT higher long-run growth rate
       Growth sustained only by technological progress

    2. **Ignoring diminishing returns:**
       Cannot grow forever by capital accumulation alone
       α < 1 ensures diminishing returns

    3. **Misinterpreting steady state:**
       k* constant doesn't mean economy stopped growing
       Total output Y = y·L grows at rate n (population)
       Per-capita y constant, but total Y growing

    4. **Assuming instant convergence:**
       Convergence takes decades (half-life ~35 years)
       Short-run dynamics matter for policy

    5. **Forgetting population dilution:**
       With n > 0, investment must first offset dilution
       Effective depreciation: δ + n

    6. **Using wrong α:**
       α should be capital share, empirically ~0.33
       α = 0.5 gives equal capital/labor importance (unrealistic)

    Extensions:
    ----------
    1. **Human capital:**
       k → (k_physical, k_human)
       Education and skills accumulation

    2. **Technology diffusion:**
       A[t+1] = A[t] + g·A[t] + φ(A_frontier - A[t])
       Catch-up dynamics

    3. **Multiple sectors:**
       Agriculture, manufacturing, services
       Structural transformation

    4. **Open economy:**
       Include international trade, capital flows
       Foreign investment, remittances

    5. **Endogenous growth:**
       Make technology growth endogenous
       R&D, learning-by-doing, spillovers

    6. **Resource constraints:**
       Add natural resources (oil, land)
       Environmental limits to growth

    See Also:
    --------
    LogisticMap : Similar mathematical structure (but very different interpretation)
    """

    def define_system(
        self,
        s: float = 0.3,
        delta: float = 0.05,
        alpha: float = 0.33,
        A: float = 1.0,
        n: float = 0.0,
        dt: float = 1.0,
        include_population_growth: bool = False,
        use_technology_growth: bool = False,
        g_A: float = 0.0,
    ):
        """
        Define discrete-time Solow growth model.

        Parameters
        ----------
        s : float
            Savings rate (0 < s < 1)
        delta : float
            Depreciation rate (0 < δ < 1)
        alpha : float
            Capital share / output elasticity (0 < α < 1)
        A : float
            Technology level / TFP (A > 0)
        n : float
            Population growth rate
        dt : float
            Time period [years]
        include_population_growth : bool
            If True, include population growth in dynamics
        use_technology_growth : bool
            If True, allow A to grow at rate g_A
        g_A : float
            Technology growth rate (if use_technology_growth=True)
        """
        # Store configuration
        self.s = s
        self.delta = delta
        self.alpha = alpha
        self.A = A
        self.n = n
        self.g_A = g_A
        self._include_pop_growth = include_population_growth
        self._include_tech_growth = use_technology_growth

        # Validate parameters
        if not (0 < s < 1):
            raise ValueError(f"Savings rate must satisfy 0 < s < 1, got s = {s}")
        if not (0 < delta < 1):
            raise ValueError(f"Depreciation must satisfy 0 < δ < 1, got δ = {delta}")
        if not (0 < alpha < 1):
            raise ValueError(f"Capital share must satisfy 0 < α < 1, got α = {alpha}")
        if A <= 0:
            raise ValueError(f"Technology must be positive, got A = {A}")

        # State variable
        k = sp.symbols("k", real=True, positive=True)

        # Symbolic parameters
        s_sym, delta_sym, alpha_sym, A_sym, n_sym = sp.symbols(
            "s delta alpha A n",
            real=True,
            positive=True,
        )

        self.state_vars = [k]
        self.control_vars = []  # No control in basic model
        self._dt = dt
        self.order = 1

        self.parameters = {
            s_sym: s,
            delta_sym: delta,
            alpha_sym: alpha,
            A_sym: A,
            n_sym: n if include_population_growth else 0.0,
        }

        # Production function (Cobb-Douglas)
        y = A_sym * k**alpha_sym

        # Capital accumulation equation
        # k[t+1] = (1 - δ)·k[t] + s·y[t]
        k_next = (1 - delta_sym) * k + s_sym * y

        # Include population growth if requested
        if include_population_growth:
            k_next = k_next / (1 + n_sym)

        self._f_sym = sp.Matrix([k_next])

        # Output function
        self._h_sym = sp.Matrix([k, y, (1 - s_sym) * y])  # [k, y, c]
        self.output_vars = []

    def setup_equilibria(self):
        """Set up steady-state equilibrium."""
        k_star = self.compute_steady_state()

        self.add_equilibrium(
            "steady_state",
            x_eq=np.array([k_star]),
            u_eq=np.array([]),
            verify=True,
            stability="stable",
            notes=f"Steady-state capital k* = {k_star:.2f}. "
            f"Globally stable: all k[0] > 0 converge to k*.",
        )

        self.set_default_equilibrium("steady_state")

    def compute_steady_state(self) -> float:
        """
        Compute steady-state capital per worker k*.

        Returns
        -------
        float
            Steady-state capital k*

        Notes
        -----
        At steady state: k[t+1] = k[t] = k*
            (1 - δ)·k* + s·A·(k*)^α = k*·(1 + n)
            s·A·(k*)^α = (δ + n)·k*
            k* = (s·A/(δ + n))^(1/(1-α))

        Examples
        --------
        >>> model = DiscreteSolowModel(s=0.25, delta=0.05, alpha=0.33)
        >>> k_star = model.compute_steady_state()
        >>> print(f"Steady-state capital: {k_star:.2f}")
        """
        effective_depreciation = self.delta + (self.n if self._include_pop_growth else 0.0)
        k_star = (self.s * self.A / effective_depreciation) ** (1.0 / (1.0 - self.alpha))
        return k_star

    def compute_output(self, k: np.ndarray) -> np.ndarray:
        """
        Compute output per worker y = A·k^α.

        Parameters
        ----------
        k : np.ndarray
            Capital per worker

        Returns
        -------
        np.ndarray
            Output per worker
        """
        return self.A * k**self.alpha

    def compute_consumption(self, k: np.ndarray) -> np.ndarray:
        """
        Compute consumption per worker c = (1-s)·y.

        Parameters
        ----------
        k : np.ndarray
            Capital per worker

        Returns
        -------
        np.ndarray
            Consumption per worker
        """
        y = self.compute_output(k)
        return (1.0 - self.s) * y

    def compute_investment(self, k: np.ndarray) -> np.ndarray:
        """
        Compute investment per worker i = s·y.

        Parameters
        ----------
        k : np.ndarray
            Capital per worker

        Returns
        -------
        np.ndarray
            Investment per worker
        """
        y = self.compute_output(k)
        return self.s * y

    def compute_golden_rule_capital(self) -> float:
        """
        Compute golden rule capital level k_gold.

        The golden rule maximizes steady-state consumption.

        Returns
        -------
        float
            Golden rule capital

        Notes
        -----
        At golden rule: MPK = δ + n
            α·A·k_gold^(α-1) = δ + n
            k_gold = (α·A/(δ + n))^(1/(1-α))

        Examples
        --------
        >>> model = DiscreteSolowModel()
        >>> k_gold = model.compute_golden_rule_capital()
        >>> k_star = model.compute_steady_state()
        >>> print(f"Current k*: {k_star:.2f}")
        >>> print(f"Golden rule k_gold: {k_gold:.2f}")
        """
        effective_depreciation = self.delta + (self.n if self._include_pop_growth else 0.0)
        k_gold = (self.alpha * self.A / effective_depreciation) ** (1.0 / (1.0 - self.alpha))
        return k_gold

    def compute_convergence_speed(self) -> float:
        """
        Compute speed of convergence β.

        Returns
        -------
        float
            Convergence coefficient β

        Notes
        -----
        Linearization around k* gives:
            k[t+1] - k* ≈ λ·(k[t] - k*)

        where λ = 1 - β and:
            β = (1 - α)(δ + n)

        Half-life: t_half = ln(2)/β

        Examples
        --------
        >>> model = DiscreteSolowModel()
        >>> beta = model.compute_convergence_speed()
        >>> print(f"Convergence rate: {beta:.4f}")
        >>> print(f"Half-life: {np.log(2)/beta:.1f} years")
        """
        effective_depreciation = self.delta + (self.n if self._include_pop_growth else 0.0)
        beta = (1.0 - self.alpha) * effective_depreciation
        return beta

    def plot_solow_diagram(
        self,
        k_range: Optional[Tuple[float, float]] = None,
    ) -> "go.Figure":
        """
        Plot classic Solow diagram (phase diagram).

        Shows:
        - Investment curve: s·f(k)
        - Depreciation line: δ·k (or (δ+n)·k)
        - Steady state at intersection

        Parameters
        ----------
        k_range : Optional[tuple]
            Range of k values to plot

        Returns
        -------
        go.Figure
            Solow diagram

        Examples
        --------
        >>> model = DiscreteSolowModel()
        >>> fig = model.plot_solow_diagram()
        >>> fig.show()
        """
        import plotly.graph_objects as go

        if k_range is None:
            k_star = self.compute_steady_state()
            k_range = (0, 2.5 * k_star)

        k_vals = np.linspace(k_range[0], k_range[1], 500)

        # Investment curve
        investment = self.s * self.A * k_vals**self.alpha

        # Depreciation line
        effective_depreciation = self.delta + (self.n if self._include_pop_growth else 0.0)
        depreciation = effective_depreciation * k_vals

        # Steady state
        k_star = self.compute_steady_state()

        fig = go.Figure()

        # Investment
        fig.add_trace(
            go.Scatter(
                x=k_vals,
                y=investment,
                mode="lines",
                name=f"s·f(k) = {self.s}·A·k^{self.alpha}",
                line=dict(color="blue", width=2),
            ),
        )

        # Depreciation
        dep_label = (
            f"(δ+n)·k = {effective_depreciation:.3f}·k"
            if self._include_pop_growth
            else f"δ·k = {self.delta}·k"
        )
        fig.add_trace(
            go.Scatter(
                x=k_vals,
                y=depreciation,
                mode="lines",
                name=dep_label,
                line=dict(color="red", width=2),
            ),
        )

        # Steady state
        fig.add_vline(
            x=k_star,
            line_dash="dash",
            line_color="green",
            annotation_text=f"k* = {k_star:.2f}",
        )

        # Shade regions
        idx_star = np.argmin(np.abs(k_vals - k_star))

        # Region where k is growing (investment > depreciation)
        if idx_star > 0:
            fig.add_trace(
                go.Scatter(
                    x=k_vals[:idx_star],
                    y=investment[:idx_star],
                    fill="tonexty",
                    fillcolor="rgba(0, 255, 0, 0.1)",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
            )

        fig.update_layout(
            title="Solow Diagram: Investment and Depreciation",
            xaxis_title="Capital per Worker k",
            yaxis_title="Investment, Depreciation",
            width=900,
            height=600,
            plot_bgcolor="white",
            annotations=[
                dict(
                    x=k_star * 0.5,
                    y=max(investment) * 0.8,
                    text="k growing<br>(s·f(k) > (δ+n)·k)",
                    showarrow=False,
                    font=dict(size=12, color="green"),
                ),
                dict(
                    x=k_star * 1.5,
                    y=max(investment) * 0.8,
                    text="k declining<br>(s·f(k) < (δ+n)·k)",
                    showarrow=False,
                    font=dict(size=12, color="red"),
                ),
            ],
        )

        return fig

    # def print_equations(self, simplify: bool = True):
    #     """Print symbolic equations."""
    #     print("=" * 70)
    #     print(f"{self.__class__.__name__} (Discrete-Time, dt={self.dt} years)")
    #     print("=" * 70)
    #     print("Solow-Swan Neoclassical Growth Model")

    #     print("\nEconomic Parameters:")
    #     print(f"  Savings rate: s = {self.s} ({self.s*100:.0f}% of output saved)")
    #     print(f"  Depreciation rate: δ = {self.delta} ({self.delta*100:.0f}% per year)")
    #     print(f"  Capital share: α = {self.alpha}")
    #     print(f"  Technology level: A = {self.A}")
    #     if self._include_pop_growth:
    #         print(f"  Population growth: n = {self.n} ({self.n*100:.1f}% per year)")
    #     if self._include_tech_growth:
    #         print(f"  Technology growth: g_A = {self.g_A} ({self.g_A*100:.1f}% per year)")

    #     print(f"\nState: x = [k] (capital per worker)")
    #     print(f"No control (autonomous model)")
    #     print(f"Dimensions: nx={self.nx}, nu={self.nu}")

    #     print("\nProduction Function (Cobb-Douglas):")
    #     print(f"  y = A·k^α = {self.A}·k^{self.alpha}")
    #     print(f"  Output per worker as function of capital per worker")

    #     print("\nCapital Accumulation Equation:")
    #     if self._include_pop_growth:
    #         print("  k[t+1] = [(1 - δ)·k[t] + s·A·k[t]^α] / (1 + n)")
    #     else:
    #         print("  k[t+1] = (1 - δ)·k[t] + s·A·k[t]^α")

    #     print("\nDiscrete-Time Dynamics:")
    #     expr = self._f_sym[0]
    #     expr_sub = self.substitute_parameters(expr)
    #     if simplify:
    #         expr_sub = sp.simplify(expr_sub)
    #     print(f"  k[t+1] = {expr_sub}")

    #     print("\nOutput Variables:")
    #     print("  y[t] = A·k[t]^α (output per worker)")
    #     print("  c[t] = (1 - s)·y[t] (consumption per worker)")
    #     print("  i[t] = s·y[t] (investment per worker)")

    #     # Steady state
    #     k_star = self.compute_steady_state()
    #     y_star = self.compute_output(k_star)
    #     c_star = self.compute_consumption(k_star)

    #     print("\nSteady-State Values:")
    #     print(f"  Capital: k* = {k_star:.2f}")
    #     print(f"  Output: y* = {y_star:.2f}")
    #     print(f"  Consumption: c* = {c_star:.2f}")
    #     print(f"  Investment: i* = {self.s * y_star:.2f}")

    #     # Golden rule
    #     k_gold = self.compute_golden_rule_capital()
    #     c_gold = self.compute_consumption(k_gold)

    #     print("\nGolden Rule (Maximum Consumption):")
    #     print(f"  Optimal capital: k_gold = {k_gold:.2f}")
    #     print(f"  Optimal savings: s_gold = α = {self.alpha}")
    #     print(f"  Maximum consumption: c_gold = {c_gold:.2f}")

    #     if abs(self.s - self.alpha) < 0.01:
    #         print("  ✓ Economy is at golden rule")
    #     elif self.s > self.alpha:
    #         print(f"  ⚠ Economy is over-saving (s > α)")
    #         print(f"    Could increase consumption by reducing s")
    #     else:
    #         print(f"  Economy is under-saving (s < α)")
    #         print(f"    Could reach higher consumption with higher s")

    #     # Convergence
    #     beta = self.compute_convergence_speed()
    #     half_life = np.log(2) / beta if beta > 0 else np.inf

    #     print("\nConvergence Dynamics:")
    #     print(f"  Convergence rate: β = {beta:.4f}")
    #     print(f"  Half-life: {half_life:.1f} years")
    #     print(f"  (Time for gap to k* to halve)")

    #     # Linearization
    #     Ad, Bd = self.linearize(np.array([k_star]), np.array([]))
    #     eigenvalue = Ad[0, 0]
    #     print(f"  Eigenvalue at k*: λ = {eigenvalue:.4f}")
    #     print(f"  Stable: {abs(eigenvalue) < 1} (|λ| < 1)")

    #     print("\nEconomic Interpretation:")
    #     print("  - k: Capital per worker [$/worker or machines/worker]")
    #     print("  - y: GDP per capita")
    #     print("  - c: Consumption per capita")
    #     print("  - Diminishing returns: f''(k) < 0")
    #     print("  - Convergence: Poor countries grow faster (conditional)")

    #     print("\nKey Predictions:")
    #     print("  1. Economies converge to steady state k*")
    #     print("  2. Higher savings → higher income level (but not growth rate)")
    #     print("  3. Long-run growth requires technological progress")
    #     print("  4. Capital accumulation alone cannot sustain growth")

    #     print("\nApplications:")
    #     print("  - Long-run economic growth analysis")
    #     print("  - Cross-country income differences")
    #     print("  - Development economics and poverty traps")
    #     print("  - Savings and investment policy")
    #     print("  - Convergence hypothesis testing")

    #     print("=" * 70)


# Aliases
SolowGrowthModel = DiscreteSolowModel
SolowSwanModel = DiscreteSolowModel
NeoclassicalGrowthModel = DiscreteSolowModel
