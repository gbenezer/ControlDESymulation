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

from typing import Optional

import numpy as np
import sympy as sp

from src.systems.base.core.continuous_symbolic_system import ContinuousSymbolicSystem


class ContinuousBatchReactor(ContinuousSymbolicSystem):
    """
    Continuous-time chemical batch reactor with temperature control.

    Physical System:
    ---------------
    A well-mixed batch reactor where chemical species A converts to B,
    which then converts to product C. The reactor operates in continuous
    time with differential equations governing the evolution of concentrations
    and temperature.

    The reaction sequence is:
        A → B → C

    Both reactions are first-order and temperature-dependent following
    Arrhenius kinetics. Temperature affects reaction rates exponentially,
    creating strong nonlinear coupling between composition and thermal
    dynamics.

    State Space:
    -----------
    State: x = [Cₐ, Cᵦ, T]
        Concentration states:
        - Cₐ: Concentration of reactant A [mol/L]
          * Initial concentration typically Cₐ(0) = 1.0 mol/L
          * Decreases monotonically (consumed by first reaction)
          * Must remain non-negative: Cₐ ≥ 0

        - Cᵦ: Concentration of intermediate B [mol/L]
          * Produced from A, consumed to form C
          * Non-monotonic: rises then falls
          * Maximum occurs when r₁ = r₂ (production = consumption)
          * Must remain non-negative: Cᵦ ≥ 0

        Temperature state:
        - T: Reactor temperature [K]
          * Typical range: 300-400 K (27-127°C)
          * Affects reaction rates exponentially via Arrhenius
          * Subject to heat loss to ambient (cooling)
          * Controlled by external heating Q

    Control: u = [Q]
        - Q: Heating/cooling rate [K/s]
          * Q > 0: Heating applied
          * Q < 0: Active cooling
          * Q = 0: Natural heat loss only
          * Typical range: -50 to +50 K/s

    Output: y = [Cₐ, Cᵦ, T]
        - Full state measurement (all concentrations and temperature)
        - In practice, concentration may be measured via:
          * Spectroscopy (UV-Vis, IR)
          * Chromatography (GC, HPLC)
          * Online analyzers
        - Temperature measured via thermocouple or RTD

    Dynamics:
    --------
    The continuous-time dynamics are:

        dCₐ/dt = -r₁
        dCᵦ/dt = r₁ - r₂
        dT/dt = Q - α·(T - Tₐₘᵦ)

    **Reaction Rates (Arrhenius kinetics)**:
        r₁ = k₁·Cₐ·exp(-E₁/T)    [mol/(L·s)]
        r₂ = k₂·Cᵦ·exp(-E₂/T)    [mol/(L·s)]

    where:
    - k₁, k₂: Pre-exponential factors (frequency factors)
    - E₁, E₂: Activation energies [K] (using Eₐ/R as temperature)
    - exp(-E/T): Arrhenius temperature dependence

    **Physical Interpretation**:

    Reaction 1 (A → B):
    - Rate r₁ proportional to Cₐ (first-order kinetics)
    - Exponentially increases with temperature
    - Higher E₁ → more temperature sensitive
    - Depletes reactant A, produces intermediate B

    Reaction 2 (B → C):
    - Rate r₂ proportional to Cᵦ (first-order kinetics)
    - Exponentially increases with temperature
    - Higher E₂ → more temperature sensitive
    - Consumes intermediate B, produces final product C

    Temperature dynamics:
    - Q: External heating/cooling control
    - -α·(T - Tₐₘᵦ): Heat loss to ambient (Newton's cooling)
    - α: Heat transfer coefficient [1/s]
    - Tₐₘᵦ: Ambient temperature [K]

    **Nonlinear Coupling**:
    The system exhibits strong nonlinear coupling:
    1. Temperature affects reaction rates exponentially
    2. Reactions may be exothermic/endothermic (not modeled here)
    3. Competing reactions create non-monotonic Cᵦ profile

    Parameters:
    ----------
    k1 : float, default=0.5
        Pre-exponential factor for reaction 1 (A→B) [1/s]
        Higher k₁ → faster depletion of A
        Typical range: 0.1 - 10.0

    k2 : float, default=0.3
        Pre-exponential factor for reaction 2 (B→C) [1/s]
        Higher k₂ → faster conversion of B to C
        Typical range: 0.1 - 10.0

    E1 : float, default=1000.0
        Activation energy for reaction 1 [K] (actually Eₐ/R)
        Higher E₁ → more sensitive to temperature
        Physical Eₐ typically 8,000 - 30,000 K

    E2 : float, default=1500.0
        Activation energy for reaction 2 [K] (actually Eₐ/R)
        E₂ > E₁ means reaction 2 is more temperature-sensitive
        Creates selectivity control via temperature

    alpha : float, default=0.1
        Heat transfer coefficient [1/s]
        Characterizes cooling rate to ambient
        Higher α → faster heat loss, harder to maintain temperature

    T_amb : float, default=300.0
        Ambient temperature [K] (27°C)
        System equilibrium temperature with Q = 0

    Equilibria:
    ----------
    **Steady-state (complete conversion)**:
        x_eq = [0, 0, Tₐₘᵦ]  (all reactants consumed, cooled to ambient)
        u_eq = 0  (no heating needed)

    This equilibrium is reached asymptotically as t → ∞ when:
    - All A has converted to B: Cₐ → 0
    - All B has converted to C: Cᵦ → 0
    - Temperature equilibrates with ambient: T → Tₐₘᵦ

    This is a **stable equilibrium** (globally attracting).

    **Optimal operating point** (maximum B yield):
        If goal is to maximize Cᵦ at a specific time, equilibrium
        concept doesn't apply. Instead, use optimal control to find
        temperature trajectory Q(t) that maximizes Cᵦ at final time.

    **Temperature setpoint equilibrium** (partial reaction):
        For constant T* > Tₐₘᵦ maintained by control:
        - Requires Q_eq = α·(T* - Tₐₘᵦ) to balance heat loss
        - Concentrations evolve according to reaction kinetics at T*
        - Not a true equilibrium (Cₐ, Cᵦ still changing)

    Control Objectives:
    ------------------
    Common control goals for batch reactors:

    1. **Temperature tracking**: Maintain T(t) ≈ T_ref(t)
       - Maximize reaction rate
       - Ensure safety (prevent runaway)
       - PID/LQR/MPC controllers typical

    2. **Yield optimization**: Maximize Cᵦ at final time
       - Requires optimal temperature trajectory
       - May involve heating → cooling profile
       - Calculus of variations or optimal control

    3. **Batch time minimization**: Reach Cₐ < ε in minimum time
       - Subject to temperature constraints (T_min ≤ T ≤ T_max)
       - Bang-bang control often optimal

    4. **Selectivity control**: Maximize ratio Cᵦ/Cᶜ
       - Exploit different activation energies (E₁ vs E₂)
       - Intermediate temperature maximizes B

    State Constraints:
    -----------------
    Physical constraints that must be enforced:

    1. **Non-negativity**: Cₐ(t) ≥ 0, Cᵦ(t) ≥ 0
       - Concentrations cannot be negative
       - Physical meaning: species present or absent

    2. **Conservation**: Cₐ(t) + Cᵦ(t) + Cᶜ(t) = Cₐ(0)
       - Total moles conserved (if C tracked)
       - Useful for validation

    3. **Temperature limits**: T_min ≤ T(t) ≤ T_max
       - Safety: prevent runaway or solidification
       - Typical: 280 K ≤ T ≤ 450 K

    4. **Actuation limits**: Q_min ≤ Q(t) ≤ Q_max
       - Physical heating/cooling capacity
       - Typical: -50 ≤ Q ≤ 50 K/s

    Numerical Integration:
    ---------------------
    **Stiffness**: This system can be **moderately stiff** due to:
    - Exponential temperature dependence (Arrhenius)
    - Different time scales (fast reactions at high T, slow cooling)
    - Stiffness ratio ≈ exp((E₂ - E₁)/T)

    **Recommended Solvers**:
    - **Moderate stiffness**: RK45 (adaptive Runge-Kutta)
    - **High stiffness**: Radau, BDF (implicit methods)
    - **High accuracy**: Vern7, Vern9 (Julia DiffEq)
    - **GPU acceleration**: JAX with diffrax

    **Tolerance Selection**:
    - Standard: rtol=1e-6, atol=1e-8
    - High accuracy: rtol=1e-9, atol=1e-11
    - Tighter tolerances needed for optimization

    Example Usage:
    -------------
    >>> # Create reactor with default parameters
    >>> reactor = ContinuousBatchReactor()
    >>> 
    >>> # Initial condition: fresh batch
    >>> x0 = np.array([1.0, 0.0, 350.0])  # [Cₐ, Cᵦ, T]
    >>> 
    >>> # Simulate with constant heating
    >>> def controller(x, t):
    ...     return np.array([10.0])  # Constant Q = 10 K/s
    >>> 
    >>> result = reactor.simulate(
    ...     x0=x0,
    ...     controller=controller,
    ...     t_span=(0, 100),
    ...     dt=0.1
    ... )
    >>> 
    >>> # Plot concentration profiles
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(result['time'], result['states'][:, 0], label='Cₐ')
    >>> plt.plot(result['time'], result['states'][:, 1], label='Cᵦ')
    >>> plt.xlabel('Time (s)')
    >>> plt.ylabel('Concentration [mol/L]')
    >>> plt.legend()
    >>> 
    >>> # Design LQR temperature controller
    >>> T_ref = 360.0  # Reference temperature
    >>> x_ref = np.array([0.5, 0.3, T_ref])
    >>> u_ref = reactor._calculate_steady_heating(T_ref)
    >>> 
    >>> A, B = reactor.linearize(x_ref, u_ref)
    >>> Q_lqr = np.diag([0, 0, 100])  # Only care about temperature
    >>> R_lqr = np.array([[1.0]])
    >>> lqr_result = reactor.control.design_lqr(A, B, Q_lqr, R_lqr, 
    ...                                          system_type='continuous')
    >>> K = lqr_result['gain']
    >>> 
    >>> # Simulate with LQR control
    >>> def lqr_controller(x, t):
    ...     return -K @ (x - x_ref) + u_ref
    >>> 
    >>> result_lqr = reactor.simulate(x0, lqr_controller, t_span=(0, 100), dt=0.1)
    >>> 
    >>> # Or use integrate() for adaptive time stepping
    >>> result_adaptive = reactor.integrate(
    ...     x0=x0,
    ...     u=lqr_controller,
    ...     t_span=(0, 100),
    ...     method='Radau',  # Stiff solver
    ...     rtol=1e-8,
    ...     atol=1e-10
    ... )

    Physical Insights:
    -----------------
    **Reaction Selectivity**:
    Since E₂ > E₁ (default), reaction 2 is more temperature-sensitive.
    This means:
    - Low T: Slow r₂, Cᵦ accumulates (favors intermediate)
    - High T: Fast r₂, Cᵦ depletes quickly (favors product)

    **Temperature Control Strategy**:
    To maximize Cᵦ yield:
    1. Heat initially to accelerate reaction 1 (produce B)
    2. Cool before reaction 2 becomes too fast (preserve B)
    3. Optimal trajectory: heating → plateau → cooling

    **Batch Time vs. Yield Tradeoff**:
    - High temperature: Fast reactions, short batch time, but may
      overshoot optimal Cᵦ (too much conversion to C)
    - Low temperature: Slow reactions, long batch time, but can
      maintain high Cᵦ for longer
    - Economic optimum balances these factors

    **Safety Considerations**:
    - Exothermic reactions (not modeled) can cause thermal runaway
    - High temperature reduces selectivity, may form byproducts
    - Emergency cooling (Q < 0) must be available
    - Temperature constraints critical for safe operation

    **Comparison with Discrete Version**:
    This continuous-time model is the "ground truth" that discrete
    systems approximate:
    - Discrete system: Uses Euler/RK4 discretization with fixed dt
    - Continuous system: Adaptive time stepping, arbitrary accuracy
    - Use discretize() method to create discrete version:
```python
        reactor_discrete = reactor.discretize(dt=1.0, method='rk4')
```

    See Also:
    --------
    DiscreteBatchReactor : Discrete-time version of this system
    ContinuousCSTR : Continuous stirred-tank reactor (continuous flow)
    Lorenz : Another nonlinear continuous system with multiple equilibria
    VanDerPolOscillator : Continuous nonlinear oscillator with limit cycle

    Notes
    -----
    **Stiffness Detection**: If integration is slow or fails, try:
    1. Check condition number of Jacobian
    2. Use stiff solver (Radau, BDF)
    3. Reduce temperature range
    4. Use Julia backend for better stiff solvers

    **Optimal Control**: For batch optimization:
    1. Define cost functional: J = -Cᵦ(t_f) + ∫(Q²/R)dt
    2. Solve using Pontryagin's maximum principle
    3. Or use direct methods (collocation, multiple shooting)
    4. Result: Bang-bang or singular arc control

    **Parameter Estimation**: If fitting to data:
    1. Minimize ||data - model(θ)||²
    2. Use scipy.optimize.minimize with integrate()
    3. May need to estimate k₁, k₂, E₁, E₂, α
    4. Ensure identifiability (different θ → different output)
    """

    def define_system(
        self,
        k1_val: float = 0.5,
        k2_val: float = 0.3,
        E1_val: float = 1000.0,
        E2_val: float = 1500.0,
        alpha_val: float = 0.1,
        T_amb_val: float = 300.0,
        C_A0: Optional[float] = None,
        T0: Optional[float] = None,
    ):
        """
        Define symbolic continuous-time batch reactor dynamics.

        Parameters
        ----------
        k1_val : float
            Pre-exponential factor for A→B reaction [1/s]
        k2_val : float
            Pre-exponential factor for B→C reaction [1/s]
        E1_val : float
            Activation energy for reaction 1 [K]
        E2_val : float
            Activation energy for reaction 2 [K]
        alpha_val : float
            Heat transfer coefficient [1/s]
        T_amb_val : float
            Ambient temperature [K]
        C_A0 : Optional[float]
            Initial concentration of A for equilibrium setup [mol/L]
        T0 : Optional[float]
            Initial temperature for equilibrium setup [K]
        """
        # Store initial conditions for equilibrium setup
        self.C_A0 = C_A0
        self.T0 = T0

        # State variables
        C_A, C_B, T = sp.symbols("C_A C_B T", real=True, positive=True)
        Q = sp.symbols("Q", real=True)

        # Parameters
        k1, k2, E1, E2, alpha, T_amb = sp.symbols(
            "k1 k2 E1 E2 alpha T_amb", real=True, positive=True
        )

        self.parameters = {
            k1: k1_val,
            k2: k2_val,
            E1: E1_val,
            E2: E2_val,
            alpha: alpha_val,
            T_amb: T_amb_val,
        }

        self.state_vars = [C_A, C_B, T]
        self.control_vars = [Q]
        self.output_vars = []
        self.order = 1

        # Reaction rates (Arrhenius kinetics)
        # r1 = k1 * C_A * exp(-E1/T)  [A -> B]
        # r2 = k2 * C_B * exp(-E2/T)  [B -> C]
        r1 = k1 * C_A * sp.exp(-E1 / T)
        r2 = k2 * C_B * sp.exp(-E2 / T)

        # Continuous-time dynamics
        # dC_A/dt = -r1
        dC_A_dt = -r1

        # dC_B/dt = r1 - r2
        dC_B_dt = r1 - r2

        # dT/dt = Q - alpha * (T - T_amb)
        dT_dt = Q - alpha * (T - T_amb)

        self._f_sym = sp.Matrix([dC_A_dt, dC_B_dt, dT_dt])
        self._h_sym = sp.Matrix([C_A, C_B, T])

    def setup_equilibria(self):
        """
        Set up equilibrium points for the batch reactor.

        Adds two equilibria:
        1. 'complete': Complete conversion (Cₐ=0, Cᵦ=0, T=Tₐₘᵦ)
        2. 'initial': Optional initial state if C_A0 and T0 specified
        """
        # Get parameter values
        T_amb = self.parameters[sp.symbols("T_amb")]

        # Complete conversion equilibrium (stable, global attractor)
        self.add_equilibrium(
            "complete",
            x_eq=np.array([0.0, 0.0, T_amb]),
            u_eq=np.array([0.0]),
            verify=True,
            stability="stable",
            notes="Complete conversion: all reactants consumed, cooled to ambient",
        )

        # If initial conditions specified, add as reference point
        if self.C_A0 is not None and self.T0 is not None:
            # Calculate required heating to maintain initial temperature
            alpha = self.parameters[sp.symbols("alpha")]
            Q_init = alpha * (self.T0 - T_amb)

            self.add_equilibrium(
                "initial",
                x_eq=np.array([self.C_A0, 0.0, self.T0]),
                u_eq=np.array([Q_init]),
                verify=False,  # Not a true equilibrium (C_A still changes)
                stability="unstable",
                notes=f"Initial fresh batch state with T maintained at {self.T0} K",
            )

            self.set_default_equilibrium("initial")
        else:
            self.set_default_equilibrium("complete")

    def calculate_steady_heating(self, T_setpoint: float) -> float:
        """
        Calculate steady-state heating required to maintain temperature setpoint.

        Parameters
        ----------
        T_setpoint : float
            Desired reactor temperature [K]

        Returns
        -------
        float
            Required heating rate Q [K/s]

        Notes
        -----
        At steady state (constant T), heat input must balance heat loss:
            Q = α·(T - T_amb)
        """
        alpha = self.parameters[sp.symbols("alpha")]
        T_amb = self.parameters[sp.symbols("T_amb")]

        return alpha * (T_setpoint - T_amb)

    def compute_conversion(self, C_A: float, C_A0: float) -> float:
        """
        Compute fractional conversion of reactant A.

        Parameters
        ----------
        C_A : float
            Current concentration of A [mol/L]
        C_A0 : float
            Initial concentration of A [mol/L]

        Returns
        -------
        float
            Conversion fraction X_A (0 = no conversion, 1 = complete)

        Examples
        --------
        >>> reactor = ContinuousBatchReactor()
        >>> X = reactor.compute_conversion(C_A=0.3, C_A0=1.0)
        >>> print(f"Conversion: {X*100:.1f}%")
        Conversion: 70.0%
        """
        return (C_A0 - C_A) / C_A0

    def compute_selectivity(self, C_B: float, C_A: float, C_A0: float) -> float:
        """
        Compute selectivity to intermediate B.

        Parameters
        ----------
        C_B : float
            Current concentration of B [mol/L]
        C_A : float
            Current concentration of A [mol/L]
        C_A0 : float
            Initial concentration of A [mol/L]

        Returns
        -------
        float
            Selectivity S_B = C_B / (C_A0 - C_A) (moles B per mole A converted)

        Notes
        -----
        Selectivity measures how much intermediate B is produced per
        mole of A consumed. Values:
        - S_B = 1.0: Perfect selectivity (all A → B, no B → C yet)
        - S_B < 1.0: Some B has already converted to C
        - S_B → 0: Most B has converted to C (over-reacted)

        Examples
        --------
        >>> reactor = ContinuousBatchReactor()
        >>> S = reactor.compute_selectivity(C_B=0.5, C_A=0.3, C_A0=1.0)
        >>> print(f"Selectivity: {S:.2f} mol B / mol A converted")
        """
        A_consumed = C_A0 - C_A
        if A_consumed < 1e-10:
            return 0.0  # No conversion yet
        return C_B / A_consumed

    def compute_yield(self, C_B: float, C_A0: float) -> float:
        """
        Compute yield of intermediate B.

        Parameters
        ----------
        C_B : float
            Current concentration of B [mol/L]
        C_A0 : float
            Initial concentration of A [mol/L]

        Returns
        -------
        float
            Yield Y_B = C_B / C_A0 (moles B per initial mole A)

        Notes
        -----
        Yield is the most important metric for batch optimization.
        Combines both conversion and selectivity:
            Y_B = X_A · S_B

        Examples
        --------
        >>> reactor = ContinuousBatchReactor()
        >>> Y = reactor.compute_yield(C_B=0.4, C_A0=1.0)
        >>> print(f"Yield: {Y*100:.1f}%")
        Yield: 40.0%
        """
        return C_B / C_A0

    # def print_equations(self, simplify: bool = True):
    #     """
    #     Print symbolic equations using continuous-time notation.

    #     Parameters
    #     ----------
    #     simplify : bool
    #         If True, simplify expressions before printing
    #     """
    #     print("=" * 70)
    #     print(f"{self.__class__.__name__} (Continuous-Time)")
    #     print("=" * 70)
    #     print(f"State Variables: {self.state_vars}")
    #     print(f"Control Variables: {self.control_vars}")
    #     print(f"Dimensions: nx={self.nx}, nu={self.nu}, ny={self.ny}")

    #     # Extract parameter values for display
    #     k1_val = self.parameters[sp.symbols("k1")]
    #     k2_val = self.parameters[sp.symbols("k2")]
    #     E1_val = self.parameters[sp.symbols("E1")]
    #     E2_val = self.parameters[sp.symbols("E2")]
    #     alpha_val = self.parameters[sp.symbols("alpha")]
    #     T_amb_val = self.parameters[sp.symbols("T_amb")]

    #     print("\nPhysical Parameters:")
    #     print(f"  k₁ = {k1_val} 1/s (pre-exponential, A→B)")
    #     print(f"  k₂ = {k2_val} 1/s (pre-exponential, B→C)")
    #     print(f"  E₁ = {E1_val} K (activation energy, A→B)")
    #     print(f"  E₂ = {E2_val} K (activation energy, B→C)")
    #     print(f"  α = {alpha_val} 1/s (heat transfer coefficient)")
    #     print(f"  T_amb = {T_amb_val} K (ambient temperature)")

    #     print("\nChemical Reactions:")
    #     print("  A → B  (first-order, rate r₁ = k₁·Cₐ·exp(-E₁/T))")
    #     print("  B → C  (first-order, rate r₂ = k₂·Cᵦ·exp(-E₂/T))")

    #     print("\nDynamics: dx/dt = f(x, u)")
    #     for var, expr in zip(self.state_vars, self._f_sym):
    #         expr_sub = self.substitute_parameters(expr)
    #         if simplify:
    #             expr_sub = sp.simplify(expr_sub)
    #         print(f"  d{var}/dt = {expr_sub}")

    #     if self._h_sym is not None:
    #         print("\nOutput: y = h(x)")
    #         for i, expr in enumerate(self._h_sym):
    #             expr_sub = self.substitute_parameters(expr)
    #             if simplify:
    #                 expr_sub = sp.simplify(expr_sub)
    #             print(f"  y[{i}] = {expr_sub}")

    #     print("\nPhysical Interpretation:")
    #     print("  - Cₐ: Concentration of reactant A [mol/L]")
    #     print("  - Cᵦ: Concentration of intermediate B [mol/L]")
    #     print("  - T: Reactor temperature [K]")
    #     print("  - Q: Heating/cooling rate [K/s]")

    #     print("\nTypical Operating Range:")
    #     print("  - Cₐ: 0 - 1.0 mol/L")
    #     print("  - Cᵦ: 0 - 0.5 mol/L")
    #     print("  - T: 300 - 400 K")
    #     print("  - Q: -50 to +50 K/s")

    #     print("=" * 70)


class ContinuousCSTR(ContinuousSymbolicSystem):
    """
    Continuous-time Continuous Stirred-Tank Reactor (CSTR) with cooling jacket.

    Physical System:
    ---------------
    A continuous flow reactor where reactant A converts to product B in
    an exothermic reaction. The CSTR is one of the most studied nonlinear
    systems in chemical engineering, exhibiting rich dynamics including:
    - Multiple steady states (multiplicity)
    - Sustained oscillations (limit cycles)
    - Bifurcations and hysteresis
    - Thermal runaway behavior

    Unlike batch reactors, CSTRs operate continuously with:
    - Continuous feed stream entering at Cₐ,feed, T_feed
    - Continuous product stream leaving at reactor conditions
    - Perfect mixing assumption (uniform concentration and temperature)
    - External cooling/heating via jacket

    The CSTR represents a fundamental model in:
    - Process control (benchmark nonlinear system)
    - Nonlinear dynamics (canonical example of multiplicity)
    - Chemical reaction engineering (industrial reactor design)
    - Bifurcation theory (illustrates saddle-node, Hopf bifurcations)

    State Space:
    -----------
    State: x = [Cₐ, T]
        Concentration state:
        - Cₐ: Concentration of reactant A in reactor [mol/L]
          * Governed by material balance: in - out - reaction
          * 0 ≤ Cₐ ≤ Cₐ,feed (bounded by feed concentration)
          * Low Cₐ → high conversion (desired but challenging to control)
          * High Cₐ → low conversion (safe but inefficient)

        Temperature state:
        - T: Reactor temperature [K]
          * Governed by energy balance: in - out + generation - removal
          * Typically T > T_feed for exothermic reactions
          * Exhibits strong nonlinear coupling with concentration
          * Critical for safety (runaway prevention)
          * Small changes can cause large rate changes (Arrhenius)

    Control: u = [T_jacket]
        - T_jacket: Cooling jacket temperature [K]
          * Primary manipulated variable for temperature control
          * Affects heat removal rate via UA·(T - T_jacket)
          * Typically T_jacket < T (cooling mode)
          * Can be T_jacket > T (heating mode for startup/cold days)
          * Physical constraints: chiller/heater capacity limits
          * Rate constraints: jacket dynamics (not modeled here)

    Output: y = [Cₐ, T]
        - Full state measurement (common in modern plants)
        - In practice:
          * Cₐ: Online analyzer (GC, HPLC, NIR spectroscopy)
          * T: Thermocouple or RTD (fast, reliable)
          * Both have measurement noise and potential delays

    Dynamics:
    --------
    The continuous-time CSTR dynamics are:

        dCₐ/dt = (F/V)·(Cₐ,feed - Cₐ) - r
        dT/dt = (F/V)·(T_feed - T) + (-ΔH/ρCₚ)·r + UA/(VρCₚ)·(T_jacket - T)

    **Reaction Rate (Arrhenius kinetics)**:
        r = k₀·Cₐ·exp(-E/T)  [mol/(L·s)]

    where:
    - k₀: Pre-exponential factor [1/s]
      * Collision frequency in Arrhenius equation
      * Typical range: 10⁶-10¹² for liquid-phase reactions
      * Material and reaction-specific constant
    - E: Activation energy [K] (dimensionless Eₐ/R)
      * Energy barrier for reaction to occur
      * Typical range: 5,000-15,000 K for Eₐ/R
      * Higher E → more temperature-sensitive reaction
      * Physical activation energy Eₐ typically 40-120 kJ/mol
    - exp(-E/T): Arrhenius temperature dependence
      * Exponential sensitivity creates strong nonlinearity
      * 10°C change can double/triple reaction rate
      * Source of multiple steady states and instability

    **Physical Interpretation of Each Term**:

    Material Balance (dCₐ/dt):
    1. **(F/V)·(Cₐ,feed - Cₐ)**: Convective in/out
       - F/V = 1/τ: Inverse residence time [1/s]
       - τ = V/F: Average time molecule spends in reactor [s]
       - Positive when Cₐ < Cₐ,feed (dilution effect)
       - Acts as "restoring force" toward feed concentration
       - Time scale: τ (seconds to minutes)

    2. **-r**: Consumption by reaction
       - Always negative (reactant consumed)
       - Exponentially increases with temperature
       - Depends on current concentration (first-order)
       - Time scale: 1/k (fast at high T, slow at low T)

    At steady state: inflow rate = outflow rate + reaction rate

    Energy Balance (dT/dt):
    1. **(F/V)·(T_feed - T)**: Convective heat in/out
       - Negative when T > T_feed (typical for exothermic)
       - Same time scale as material balance (1/τ)
       - Acts as "restoring force" toward feed temperature

    2. **(-ΔH/ρCₚ)·r**: Heat generation from reaction
       - Positive for exothermic reaction (ΔH < 0)
       - Couples concentration to temperature
       - Creates positive feedback: higher T → faster r → more heat
       - This term causes thermal runaway if unchecked
       - Magnitude: |ΔH|/(ρCₚ) is adiabatic temperature rise per mol/L

    3. **UA/(VρCₚ)·(T_jacket - T)**: Heat removal via jacket
       - Negative when T > T_jacket (cooling)
       - Only term controlled by manipulated variable
       - UA: Overall heat transfer coefficient × area [J/(s·K)]
       - Larger UA → better temperature control
       - Time scale: VρCₚ/UA (thermal time constant)

    At steady state: heat in + heat generated = heat out + heat removed

    **Nonlinear Coupling and Feedback**:

    The CSTR exhibits strong positive feedback that can lead to instability:

    1. **Thermal Feedback Loop** (Runaway Mechanism):
       T ↑ → r ↑ (Arrhenius) → heat generation ↑ → T ↑ (positive feedback)

       This loop is stabilized by:
       - Convective cooling: higher T → more heat removal to feed
       - Jacket cooling: higher T → more heat removal via jacket
       - Reactant depletion: higher r → lower Cₐ → lower r (negative feedback)

    2. **Material-Thermal Coupling**:
       - High T → fast reaction → low Cₐ (depletion)
       - Low Cₐ → slow reaction → less heat generation → lower T
       - This coupling creates multiple possible steady states

    3. **Competition Between Time Scales**:
       - Residence time τ = V/F (convective transport)
       - Reaction time 1/k (chemical kinetics)
       - Thermal time VρCₚ/UA (heat transfer)
       - Relative magnitudes determine stability and multiplicity

    Parameters:
    ----------
    F : float, default=100.0
        Volumetric flow rate [L/s]
        - Controls residence time τ = V/F
        - Higher F → shorter τ → lower conversion but more stable
        - Lower F → longer τ → higher conversion but less stable
        - Typical range: 10-1000 L/s depending on reactor size
        - Often kept constant (set by upstream/downstream constraints)

    V : float, default=100.0
        Reactor volume [L]
        - Combined with F to give residence time τ
        - Larger V → more material holdup → slower dynamics
        - Typical range: 100-10,000 L for industrial reactors
        - Design parameter (fixed once reactor is built)

    C_A_feed : float, default=1.0
        Feed concentration [mol/L]
        - Upper bound for reactor concentration
        - Higher feed → more product but more heat generation
        - Typical range: 0.1-10 mol/L
        - Often a disturbance variable (feed composition changes)

    T_feed : float, default=350.0
        Feed temperature [K]
        - Inlet stream temperature
        - Typical range: 280-360 K (ambient to pre-heated)
        - Can be manipulated for control but usually fixed
        - Feed pre-heating can improve conversion but reduces stability margin

    k0 : float, default=7.2e10
        Pre-exponential factor [1/s]
        - Arrhenius equation parameter
        - Determines reaction speed at given temperature
        - Typical range: 10⁶-10¹² for liquid phase
        - Reaction and catalyst specific
        - Obtained from kinetic experiments or literature

    E : float, default=8750.0
        Activation energy [K] (actually Eₐ/R, dimensionless)
        - Energy barrier for reaction to occur
        - Higher E → more temperature-sensitive
        - Typical range: 5,000-15,000 K for Eₐ/R
        - Physical Eₐ typically 40-120 kJ/mol
        - Key parameter determining multiplicity region
        - Strong influence on stability

    delta_H : float, default=-5e4
        Heat of reaction [J/mol]
        - Energy released (negative) or absorbed (positive) per mole reacted
        - Negative = exothermic (releases heat) - most common case
        - Positive = endothermic (absorbs heat) - rare, simpler control
        - Typical for exothermic: -20,000 to -200,000 J/mol
        - Magnitude determines thermal coupling strength
        - Larger |ΔH| → stronger feedback → more multiplicity/instability

    rho : float, default=1000.0
        Density [kg/L]
        - Fluid density (assumed constant)
        - Typical for aqueous solutions: 900-1,100 kg/L
        - Affects thermal mass (heat capacity of reactor contents)

    Cp : float, default=0.239
        Specific heat capacity [J/(kg·K)]
        - Heat required to raise 1 kg by 1 K
        - Typical for aqueous: 0.2-0.5 J/(kg·K)
        - Note: Using J/(kg·K) not kJ/(kg·K), hence small value
        - Combined with ρ gives volumetric heat capacity ρCₚ
        - Higher Cₚ → larger thermal inertia → slower temperature changes

    UA : float, default=5e4
        Overall heat transfer coefficient × area [J/(s·K)]
        - Lumped parameter combining:
          * Jacket-side film coefficient
          * Wall thermal conductivity
          * Reactor-side film coefficient
          * Heat transfer area
        - Typical range: 10³-10⁵ J/(s·K)
        - Higher UA → better temperature control, faster cooling
        - Limited by physical design (jacket size, flow rate, area)
        - Critical parameter for preventing runaway

    Equilibria and Multiple Steady States:
    -------------------------------------
    **The Hallmark of CSTR Dynamics**

    The CSTR is famous for exhibiting **multiple steady states** - a phenomenon
    called **multiplicity**. For a given set of operating conditions (feed
    conditions and jacket temperature), the reactor can have:

    1. **One steady state** (unique solution)
    2. **Three steady states** (two stable, one unstable)
    3. **Five steady states** (rare, special parameter combinations)

    **Three Steady State Scenario** (most common interesting case):

    1. **Low-Conversion State (Stable)**:
       - Low temperature (T ≈ T_feed + 10-30 K)
       - High reactant concentration (Cₐ ≈ 0.7-0.9·Cₐ,feed)
       - Slow reaction rate (low k·exp(-E/T))
       - Heat generation < Heat removal capacity
       - **Basin of attraction**: States with low initial temperature
       - **Characteristics**: 
         * Easy to start up (cold startup naturally goes here)
         * Safe and stable
         * Economically poor (low conversion, wasted reactant)
         * Easy to control (large stability margins)

    2. **Intermediate State (Unstable)**:
       - Moderate temperature
       - Moderate concentration
       - **Saddle point** in phase space
       - Not physically realizable (unstable equilibrium)
       - Acts as separatrix between basins of attraction
       - **Physical meaning**: Transition point where thermal generation
         rate exactly balances heat removal rate, but balance is unstable

    3. **High-Conversion State (Stable)**:
       - High temperature (T ≈ T_feed + 50-100 K)
       - Low reactant concentration (Cₐ ≈ 0.05-0.3·Cₐ,feed)
       - Fast reaction rate (high k·exp(-E/T))
       - Large heat generation balanced by cooling
       - **Basin of attraction**: States with high initial temperature
       - **Characteristics**:
         * Desired operating point (high conversion = high profit)
         * Requires good startup procedure (must cross unstable intermediate)
         * Risk of runaway if cooling fails
         * Smaller stability margins (closer to instability boundary)
         * More challenging to control (strong nonlinearity)

    **Physical Intuition for Multiple Steady States**:

    Imagine heat generation curve vs heat removal curve:
    - Heat generation: S-shaped (Arrhenius kinetics)
      * Low T: generation small (slow reaction)
      * Medium T: generation increases rapidly (exponential activation)
      * High T: generation levels off (reactant depletion)
    
    - Heat removal: Linear in T (jacket cooling)
      * Straight line: q_removal = UA·(T - T_jacket)/VρCₚ
      * Plus convective: (F/V)·(T - T_feed)

    Intersections of these curves = steady states:
    - If removal line is steep (large UA): unique high-conversion state
    - If removal line is shallow (small UA): can have 3 intersections
    - As T_jacket varies: intersections appear/disappear (bifurcations)

    **Stability of Steady States**:

    Linear stability analysis (eigenvalues of Jacobian):
    - **Stable**: Both eigenvalues have negative real parts (LHP)
      * Perturbations decay back to steady state
      * Can operate here sustainably
    - **Unstable**: At least one eigenvalue has positive real part (RHP)
      * Perturbations grow exponentially
      * Cannot operate here (physically unrealizable)

    For CSTR:
    - Low-conversion state: Typically stable
    - Intermediate state: Always unstable (saddle point)
    - High-conversion state: Stable if cooling is adequate

    **Bifurcations** (Qualitative changes in behavior):

    1. **Saddle-Node Bifurcation** (Fold):
       As T_jacket decreases (more cooling):
       - Initially: Only low-conversion state exists
       - At bifurcation point: Two new states appear (intermediate + high)
       - Further cooling: Three states coexist
       - **Hysteresis**: Different paths for heating vs cooling

    2. **Hopf Bifurcation** (Oscillations):
       At certain parameters, high-conversion state can lose stability via
       Hopf bifurcation → sustained oscillations (limit cycle)
       - Temperature and concentration oscillate periodically
       - Can occur with insufficient cooling or long residence time
       - Indicates poor controllability

    **Finding Steady States**:

    Steady states satisfy: dCₐ/dt = 0, dT/dt = 0

    This gives two coupled nonlinear algebraic equations:
    1. (F/V)·(Cₐ,feed - Cₐ) = k₀·Cₐ·exp(-E/T)
    2. (F/V)·(T_feed - T) + (-ΔH/ρCₚ)·k₀·Cₐ·exp(-E/T) = -UA/(VρCₚ)·(T_jacket - T)

    Solution methods:
    - Numerical: Newton-Raphson, fsolve with multiple initial guesses
    - Graphical: Plot dCₐ/dt and dT/dt surfaces, find intersections
    - Continuation: Track solutions as parameters vary (AUTO, MATCONT)

    See `find_steady_states()` method for implementation.

    Control Objectives:
    ------------------

    1. **Setpoint Regulation** (Most Common):
       - Maintain reactor at high-conversion steady state
       - Reject disturbances (feed composition, flow rate, ambient temperature)
       - Controllers: PID, LQR, MPC
       - Challenge: Strong nonlinearity, operate near instability

    2. **Startup Control**:
       - Transition from low-conversion to high-conversion state
       - Must cross unstable intermediate state (separatrix crossing)
       - Requires aggressive transient cooling
       - Strategies:
         * Bang-bang cooling (maximum jacket cooling)
         * Optimal control (minimize time/energy)
         * Gain scheduling (change controller as state changes)
       - Risk: Overshoot → runaway

    3. **Runaway Prevention** (Safety Critical):
       - Detect incipient runaway conditions
       - Implement emergency cooling/shutdown
       - Monitor dT/dt (temperature rate of change)
       - Constraint: T < T_max (safety limit)
       - Last resort: Emergency cooling, feed shutoff, depressurization

    4. **Economic Optimization**:
       - Maximize profit = revenue - costs
       - Revenue: Product value = price × F × conversion
       - Costs: Cooling energy, reactant waste, equipment wear
       - Often operates close to instability boundary for maximum conversion
       - Tradeoff: Higher conversion (profit) vs safety margin (risk)

    5. **Disturbance Rejection**:
       Common disturbances:
       - Feed concentration variations: Cₐ,feed(t)
       - Feed temperature changes: T_feed(t)
       - Flow rate fluctuations: F(t)
       - Ambient temperature (affects jacket): T_ambient(t)
       - Catalyst deactivation: k₀(t) decreases slowly

    State Constraints:
    -----------------

    1. **Non-negativity**: Cₐ ≥ 0
       - Concentration cannot be negative (physical)
       - Rarely active (implies complete conversion)

    2. **Concentration Bounds**: 0 ≤ Cₐ ≤ Cₐ,feed
       - Cannot exceed feed (dilution + reaction only decrease)
       - Upper bound active only if no reaction occurs

    3. **Temperature Limits**: T_min ≤ T ≤ T_max
       - **Lower limit**: Prevent solidification/freezing (≈ 280 K)
       - **Upper limit**: Safety, prevent runaway (≈ 450-500 K)
       - Material limits: polymer degradation, wall integrity
       - **Most critical constraint for safety**

    4. **Jacket Temperature Limits**: T_jacket,min ≤ T_jacket ≤ T_jacket,max
       - Chiller capacity: T_jacket,min ≈ 280 K
       - Heater capacity: T_jacket,max ≈ 400 K
       - Rate limit: |dT_jacket/dt| ≤ rate_max (jacket dynamics)

    Time Scales and Dynamics:
    -------------------------

    **Multiple Time Scales** make CSTR dynamics rich and challenging:

    1. **Fast Scale - Reaction**: t_reaction = 1/k
       - At low T (350 K): t_reaction ≈ 10-100 s
       - At high T (400 K): t_reaction ≈ 0.1-1 s
       - Exponentially dependent on temperature
       - Can be very fast at high conversion state

    2. **Medium Scale - Residence Time**: t_residence = τ = V/F
       - Typical: 10-1000 s (seconds to minutes)
       - Time for complete turnover of reactor contents
       - Natural time scale for concentration changes
       - Design parameter

    3. **Slow Scale - Thermal**: t_thermal = VρCₚ/UA
       - Typical: 100-10,000 s (minutes to hours)
       - Time constant for temperature response
       - Limited by heat transfer through jacket
       - Design parameter (limited by area, jacket design)

    **Stiffness**: When time scales differ by orders of magnitude:
    - Fast reactions with slow heat transfer → stiff system
    - Requires implicit ODE solvers (Radau, BDF)
    - Small numerical errors in fast variables → large errors in slow variables

    Integration Recommendations:
    ---------------------------

    **Solver Selection**:

    For most CSTR problems:
    - **Moderate stiffness**: RK45 (adaptive Runge-Kutta) works well
    - **High stiffness**: Use stiff solvers
      * scipy: Radau, BDF, LSODA (auto-switching)
      * Julia (DiffEqPy): Rosenbrock23, Rodas5

    **Tolerance Selection**:
    - Standard simulation: rtol=1e-6, atol=1e-8
    - High accuracy (optimization): rtol=1e-9, atol=1e-11
    - Looser tolerances may miss important dynamics

    **Event Detection**:
    For safety-critical applications, use event detection:
    - Detect T > T_max (runaway)
    - Detect dT/dt > threshold (incipient runaway)
    - Detect steady state (convergence)

    Example Usage:
    -------------
    >>> # Create CSTR with default parameters
    >>> cstr = ContinuousCSTR(F=100.0, V=100.0)
    >>> 
    >>> # Find all steady states for given jacket temperature
    >>> T_jacket_op = 350.0
    >>> steady_states = cstr.find_steady_states(T_jacket_op)
    >>> print(f"Found {len(steady_states)} steady states:")
    >>> for i, (C_A, T) in enumerate(steady_states):
    ...     print(f"  State {i+1}: Cₐ={C_A:.3f} mol/L, T={T:.1f} K")
    ...     print(f"    Conversion: {cstr.compute_conversion(C_A, 1.0)*100:.1f}%")
    >>> 
    >>> # Choose high-conversion operating point
    >>> if len(steady_states) >= 2:
    ...     # High-conversion state (highest T, lowest Cₐ)
    ...     steady_states_sorted = sorted(steady_states, key=lambda x: x[1], reverse=True)
    ...     C_A_op, T_op = steady_states_sorted[0]
    ...     x_op = np.array([C_A_op, T_op])
    ...     u_op = np.array([T_jacket_op])
    ... else:
    ...     # Use provided or default operating point
    ...     x_op = np.array([0.1, 390.0])
    ...     u_op = np.array([350.0])
    >>> 
    >>> # Verify equilibrium
    >>> dx = cstr(x_op, u_op)
    >>> print(f"\\nEquilibrium check: ||dx/dt|| = {np.linalg.norm(dx):.2e}")
    >>> 
    >>> # Linearize at operating point
    >>> A, B = cstr.linearize(x_op, u_op)
    >>> eigenvalues = np.linalg.eigvals(A)
    >>> print(f"\\nLinearized eigenvalues: {eigenvalues}")
    >>> print(f"Stable: {np.all(np.real(eigenvalues) < 0)}")
    >>> 
    >>> # Design LQR controller (emphasize temperature control)
    >>> Q = np.diag([1.0, 100.0])  # Penalize temperature error heavily
    >>> R = np.array([[1.0]])
    >>> lqr_result = cstr.control.design_lqr(A, B, Q, R, system_type='continuous')
    >>> K = lqr_result['gain']
    >>> 
    >>> # Simulate with LQR control and disturbance
    >>> def lqr_controller(x, t):
    ...     # Add feed temperature disturbance at t=50s
    ...     if t > 50:
    ...         # Disturbance increases effective T_feed by changing heat balance
    ...         # Compensate by reducing jacket temperature
    ...         disturbance_compensation = -2.0
    ...     else:
    ...         disturbance_compensation = 0.0
    ...     
    ...     u_fb = -K @ (x - x_op) + u_op
    ...     return u_fb + disturbance_compensation
    >>> 
    >>> # Perturb from equilibrium
    >>> x0 = x_op + np.array([0.05, -5.0])  # Small perturbation
    >>> 
    >>> result = cstr.simulate(
    ...     x0=x0,
    ...     controller=lqr_controller,
    ...     t_span=(0, 200),
    ...     dt=0.1,
    ...     method='Radau'  # Stiff solver
    ... )
    >>> 
    >>> # Plot results
    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    >>> 
    >>> # Concentration
    >>> axes[0].plot(result['time'], result['states'][:, 0])
    >>> axes[0].axhline(x_op[0], color='r', linestyle='--', label='Setpoint')
    >>> axes[0].set_ylabel('Cₐ [mol/L]')
    >>> axes[0].legend()
    >>> axes[0].grid(True)
    >>> 
    >>> # Temperature
    >>> axes[1].plot(result['time'], result['states'][:, 1])
    >>> axes[1].axhline(x_op[1], color='r', linestyle='--', label='Setpoint')
    >>> axes[1].set_ylabel('T [K]')
    >>> axes[1].legend()
    >>> axes[1].grid(True)
    >>> 
    >>> # Control action
    >>> axes[2].plot(result['time'], result['controls'][:, 0])
    >>> axes[2].axhline(u_op[0], color='r', linestyle='--', label='Nominal')
    >>> axes[2].set_ylabel('T_jacket [K]')
    >>> axes[2].set_xlabel('Time [s]')
    >>> axes[2].legend()
    >>> axes[2].grid(True)
    >>> 
    >>> plt.tight_layout()
    >>> plt.show()
    >>> 
    >>> # Startup simulation: low → high conversion
    >>> # Start at low-conversion steady state
    >>> if len(steady_states) >= 3:
    ...     C_A_low, T_low = steady_states_sorted[-1]  # Lowest temperature state
    ...     x_low = np.array([C_A_low, T_low])
    ... else:
    ...     x_low = np.array([0.9, 360.0])
    >>> 
    >>> def startup_controller(x, t):
    ...     '''Aggressive cooling for startup, then switch to regulator'''
    ...     if t < 100:
    ...         # Phase 1: Aggressive cooling to jump to high-conversion state
    ...         return np.array([330.0])  # Cold jacket
    ...     else:
    ...         # Phase 2: LQR regulation around high-conversion setpoint
    ...         return lqr_controller(x, t)
    >>> 
    >>> result_startup = cstr.simulate(
    ...     x0=x_low,
    ...     controller=startup_controller,
    ...     t_span=(0, 300),
    ...     dt=0.1,
    ...     method='Radau'
    ... )
    >>> 
    >>> # Check if startup succeeded
    >>> final_state = result_startup['states'][-1, :]
    >>> distance = np.linalg.norm(final_state - x_op)
    >>> print(f"\\nStartup result:")
    >>> print(f"  Final state: Cₐ={final_state[0]:.3f}, T={final_state[1]:.1f}")
    >>> print(f"  Distance to target: {distance:.3f}")
    >>> print(f"  Success: {distance < 5.0}")
    >>> 
    >>> # Phase portrait (requires multiple simulations)
    >>> # Shows basins of attraction for multiple steady states
    >>> 
    >>> # Bifurcation diagram (vary T_jacket)
    >>> T_jacket_range = np.linspace(320, 360, 20)
    >>> bifurcation_data = {'T_jacket': [], 'C_A': [], 'T': [], 'stable': []}
    >>> 
    >>> for Tj in T_jacket_range:
    ...     states = cstr.find_steady_states(Tj)
    ...     for C_A, T in states:
    ...         # Check stability
    ...         A_local, _ = cstr.linearize(np.array([C_A, T]), np.array([Tj]))
    ...         eigs = np.linalg.eigvals(A_local)
    ...         is_stable = np.all(np.real(eigs) < 0)
    ...         
    ...         bifurcation_data['T_jacket'].append(Tj)
    ...         bifurcation_data['C_A'].append(C_A)
    ...         bifurcation_data['T'].append(T)
    ...         bifurcation_data['stable'].append(is_stable)
    >>> 
    >>> # Plot bifurcation diagram
    >>> plt.figure(figsize=(10, 6))
    >>> stable = np.array(bifurcation_data['stable'])
    >>> plt.plot(
    ...     np.array(bifurcation_data['T_jacket'])[stable],
    ...     np.array(bifurcation_data['T'])[stable],
    ...     'b-', linewidth=2, label='Stable'
    ... )
    >>> plt.plot(
    ...     np.array(bifurcation_data['T_jacket'])[~stable],
    ...     np.array(bifurcation_data['T'])[~stable],
    ...     'r--', linewidth=2, label='Unstable'
    ... )
    >>> plt.xlabel('Jacket Temperature [K]')
    >>> plt.ylabel('Reactor Temperature [K]')
    >>> plt.title('CSTR Bifurcation Diagram')
    >>> plt.legend()
    >>> plt.grid(True)
    >>> plt.show()

    Physical Insights:
    -----------------

    **Why Multiple Steady States Occur**:

    The interplay between heat generation (nonlinear, S-shaped) and heat
    removal (linear) creates the possibility of multiple intersections:

    1. **Low T region**: 
       - Reaction slow (small exp(-E/T))
       - Generation curve starts flat
       - Removal line dominates → low-conversion stable

    2. **Medium T region**:
       - Reaction accelerates rapidly (exponential growth)
       - Generation curve steepens dramatically
       - Can intersect removal line 3 times

    3. **High T region**:
       - Reaction very fast but Cₐ depleted
       - Generation curve plateaus (limited by Cₐ)
       - Removal continues linearly → high-conversion stable

    **Thermal Runaway Mechanism**:

    Positive feedback loop if cooling insufficient:
    1. Small T increase (disturbance or control action)
    2. Reaction rate jumps (exponential Arrhenius)
    3. More heat generated (exothermic)
    4. Temperature rises further
    5. Loop continues → runaway!

    Stabilizing mechanisms:
    - Reactant depletion (limits generation at high T)
    - Jacket cooling (removes heat)
    - Feed cooling (convective heat removal)

    **Industrial Significance**:

    CSTRs are ubiquitous in chemical industry:
    - Polymerization reactors
    - Pharmaceutical synthesis
    - Wastewater treatment (biological reactors)
    - Fermentation processes

    Multiple steady states have practical implications:
    - **Startup**: Complex procedure to reach desired state
    - **Control**: Must prevent unintended switching
    - **Safety**: Runaway risk at high-conversion state
    - **Economics**: Higher conversion → more profit but harder control

    **Control Challenges**:

    1. **Nonlinearity**: 
       - Linear controllers (PID) may perform poorly
       - Gain scheduling or nonlinear control needed
       - Operating point dependent behavior

    2. **Instability risk**:
       - High-conversion state often close to instability
       - Small disturbances can cause large excursions
       - Need fast, aggressive control action

    3. **Constraints**:
       - Temperature limits (safety)
       - Jacket temperature limits (physical)
       - Actuator saturation degrades performance

    4. **Multiple states**:
       - System can "jump" between states
       - Hysteresis complicates control
       - Need to prevent unintended transitions

    **Design Considerations**:

    For industrial CSTR design:
    - **Safety first**: Adequate cooling capacity (large UA)
    - **Residence time**: Balance conversion vs stability (choose V/F)
    - **Operating point**: High conversion but safe margin from instability
    - **Instrumentation**: Fast, reliable temperature measurement
    - **Emergency systems**: Backup cooling, feed shutoff, pressure relief

    Comparison with Other Reactors:
    ------------------------------

    **CSTR vs Batch Reactor**:
    - CSTR: Continuous operation, steady state, higher throughput
    - Batch: Transient operation, finite time, better for small volumes

    **CSTR vs Plug Flow Reactor (PFR)**:
    - CSTR: Back-mixed, uniform concentration/temperature
    - PFR: No back-mixing, concentration/temperature gradients
    - PFR: Generally more efficient but harder to control

    **CSTR vs Semi-Batch**:
    - CSTR: Continuous in/out
    - Semi-batch: Batch with continuous feed, better heat management

    See Also:
    --------
    DiscreteCSTR : Discrete-time version for digital control
    ContinuousBatchReactor : Batch operation instead of continuous
    DiscreteBatchReactor : Discrete batch reactor

    Notes:
    -----
    **Extensions**:
    More complex CSTR models can include:
    - Multiple reactions (A → B → C, parallel reactions)
    - Non-ideal mixing (RTD, compartment models)
    - Jacket dynamics (first-order lag in cooling)
    - Catalyst deactivation (slow time scale)
    - pH effects (additional state)
    - Gas-liquid reactions (mass transfer limitations)
    """

    def define_system(
        self,
        F_val: float = 100.0,  # Flow rate [L/s]
        V_val: float = 100.0,  # Volume [L]
        C_A_feed_val: float = 1.0,  # Feed concentration [mol/L]
        T_feed_val: float = 350.0,  # Feed temperature [K]
        k0_val: float = 7.2e10,  # Pre-exponential [1/s]
        E_val: float = 8750.0,  # Activation energy [K]
        delta_H_val: float = -5e4,  # Heat of reaction [J/mol]
        rho_val: float = 1000.0,  # Density [kg/L]
        Cp_val: float = 0.239,  # Heat capacity [J/(kg*K)]
        UA_val: float = 5e4,  # Heat transfer coef [J/(s*K)]
        x_ss: Optional[np.ndarray] = None,
        u_ss: Optional[np.ndarray] = None,
    ):
        """
        Define continuous-time CSTR dynamics.

        Parameters
        ----------
        F_val : float
            Volumetric flow rate [L/s]
        V_val : float
            Reactor volume [L]
        C_A_feed_val : float
            Feed concentration [mol/L]
        T_feed_val : float
            Feed temperature [K]
        k0_val : float
            Pre-exponential factor [1/s]
        E_val : float
            Activation energy [K] (dimensionless Eₐ/R)
        delta_H_val : float
            Heat of reaction [J/mol] (negative = exothermic)
        rho_val : float
            Density [kg/L]
        Cp_val : float
            Specific heat capacity [J/(kg·K)]
        UA_val : float
            Overall heat transfer coefficient × area [J/(s·K)]
        x_ss : Optional[np.ndarray]
            Steady-state [Cₐ, T] for equilibrium setup
        u_ss : Optional[np.ndarray]
            Steady-state [T_jacket] for equilibrium setup
        """
        self.x_ss = x_ss
        self.u_ss = u_ss

        # State and control variables
        C_A, T = sp.symbols("C_A T", real=True, positive=True)
        T_jacket = sp.symbols("T_jacket", real=True, positive=True)

        # Parameters
        F, V, C_A_feed, T_feed = sp.symbols("F V C_A_feed T_feed", real=True, positive=True)
        k0, E, delta_H, rho, Cp, UA = sp.symbols(
            "k0 E delta_H rho Cp UA", real=True, positive=True
        )

        self.parameters = {
            F: F_val,
            V: V_val,
            C_A_feed: C_A_feed_val,
            T_feed: T_feed_val,
            k0: k0_val,
            E: E_val,
            delta_H: delta_H_val,
            rho: rho_val,
            Cp: Cp_val,
            UA: UA_val,
        }

        self.state_vars = [C_A, T]
        self.control_vars = [T_jacket]
        self.output_vars = []
        self.order = 1

        # Reaction rate (Arrhenius kinetics)
        r = k0 * C_A * sp.exp(-E / T)

        # Continuous-time dynamics
        # Material balance: dC_A/dt
        dC_A_dt = (F / V) * (C_A_feed - C_A) - r

        # Energy balance: dT/dt
        dT_dt = (
            (F / V) * (T_feed - T)
            + ((-delta_H) / (rho * Cp)) * r
            + (UA / (V * rho * Cp)) * (T_jacket - T)
        )

        self._f_sym = sp.Matrix([dC_A_dt, dT_dt])
        self._h_sym = sp.Matrix([C_A, T])

    def setup_equilibria(self):
        """
        Set up steady-state equilibrium if provided.

        Notes
        -----
        CSTR can have multiple steady states! Only add user-provided
        equilibrium. Finding all equilibria requires solving nonlinear
        algebraic equations (see find_steady_states() method).
        """
        if self.x_ss is not None and self.u_ss is not None:
            self.add_equilibrium(
                "steady_state",
                x_eq=self.x_ss,
                u_eq=self.u_ss,
                verify=True,
                stability="unknown",
                notes="User-provided steady state - CSTR may have multiple equilibria",
            )
            self.set_default_equilibrium("steady_state")

    def compute_conversion(self, C_A: float, C_A_feed: float) -> float:
        """
        Compute fractional conversion of reactant A.

        Parameters
        ----------
        C_A : float
            Current reactor concentration [mol/L]
        C_A_feed : float
            Feed concentration [mol/L]

        Returns
        -------
        float
            Conversion fraction X_A = (C_A_feed - C_A) / C_A_feed

        Examples
        --------
        >>> cstr = ContinuousCSTR()
        >>> X = cstr.compute_conversion(C_A=0.1, C_A_feed=1.0)
        >>> print(f"Conversion: {X*100:.1f}%")
        Conversion: 90.0%

        Notes
        -----
        High conversion (X > 0.9) typically corresponds to high-temperature
        steady state with fast kinetics and strong exothermic heat generation.
        """
        return (C_A_feed - C_A) / C_A_feed

    def compute_residence_time(self) -> float:
        """
        Compute residence time τ = V/F.

        Returns
        -------
        float
            Residence time [s]

        Examples
        --------
        >>> cstr = ContinuousCSTR(F=100.0, V=100.0)
        >>> tau = cstr.compute_residence_time()
        >>> print(f"Residence time: {tau} s")
        Residence time: 1.0 s

        Notes
        -----
        Residence time is the average time a molecule spends in the reactor.
        - Longer τ (smaller F): More conversion, less stable
        - Shorter τ (larger F): Less conversion, more stable
        """
        F = self.parameters[sp.symbols("F")]
        V = self.parameters[sp.symbols("V")]
        return V / F

    def compute_damkohler_number(self, T: float) -> float:
        """
        Compute Damköhler number Da = k·τ (reaction rate × residence time).

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        float
            Damköhler number [dimensionless]

        Notes
        -----
        Damköhler number measures reaction rate relative to flow rate:
        - Da << 1: Reaction slow, flow dominates, low conversion
        - Da >> 1: Reaction fast, kinetics dominate, high conversion
        - Da ≈ 1: Balanced, optimal efficiency

        Examples
        --------
        >>> cstr = ContinuousCSTR()
        >>> Da_low = cstr.compute_damkohler_number(T=350.0)
        >>> Da_high = cstr.compute_damkohler_number(T=400.0)
        >>> print(f"Da(350K) = {Da_low:.2f}")
        >>> print(f"Da(400K) = {Da_high:.2f}")
        """
        k0 = self.parameters[sp.symbols("k0")]
        E = self.parameters[sp.symbols("E")]
        tau = self.compute_residence_time()

        k = k0 * np.exp(-E / T)
        return k * tau

    def find_steady_states(
        self,
        T_jacket: float,
        T_range: tuple = (300.0, 500.0),
        n_points: int = 100,
    ) -> list:
        """
        Find all steady states for a given jacket temperature.

        Uses multiple initial guesses across temperature range to find
        all solutions to the steady-state equations.

        Parameters
        ----------
        T_jacket : float
            Jacket temperature [K]
        T_range : tuple
            Temperature range to search (T_min, T_max) [K]
        n_points : int
            Number of initial guesses for root finding

        Returns
        -------
        list
            List of (C_A, T) steady state tuples

        Examples
        --------
        >>> cstr = ContinuousCSTR()
        >>> steady_states = cstr.find_steady_states(T_jacket=350.0)
        >>> print(f"Found {len(steady_states)} steady states")
        >>> for i, (C_A, T) in enumerate(steady_states):
        ...     X = cstr.compute_conversion(C_A, 1.0)
        ...     print(f"  State {i+1}: Cₐ={C_A:.3f}, T={T:.1f}, X={X*100:.1f}%")

        Notes
        -----
        This method finds steady states by solving:
            dCₐ/dt = 0  (material balance)
            dT/dt = 0   (energy balance)

        For CSTR, there can be 1, 2, or 3 steady states depending on
        parameters. This method attempts to find all of them by using
        many different initial guesses across the temperature range.

        For production code, consider:
        - scipy.optimize.fsolve with multiple guesses
        - Continuation methods (AUTO, MATCONT)
        - Homotopy methods for guaranteed finding of all solutions
        """
        from scipy.optimize import fsolve

        # Extract parameters
        F = self.parameters[sp.symbols("F")]
        V = self.parameters[sp.symbols("V")]
        C_A_feed = self.parameters[sp.symbols("C_A_feed")]
        T_feed = self.parameters[sp.symbols("T_feed")]
        k0 = self.parameters[sp.symbols("k0")]
        E = self.parameters[sp.symbols("E")]
        delta_H = self.parameters[sp.symbols("delta_H")]
        rho = self.parameters[sp.symbols("rho")]
        Cp = self.parameters[sp.symbols("Cp")]
        UA = self.parameters[sp.symbols("UA")]

        def steady_state_equations(state):
            """Steady state conditions: dC_A/dt = 0, dT/dt = 0"""
            C_A, T = state

            # Avoid numerical issues
            if C_A < 0 or T < 250:
                return [1e10, 1e10]

            r = k0 * C_A * np.exp(-E / T)

            # Material balance
            dC_A_dt = (F / V) * (C_A_feed - C_A) - r

            # Energy balance
            dT_dt = (
                (F / V) * (T_feed - T)
                + ((-delta_H) / (rho * Cp)) * r
                + (UA / (V * rho * Cp)) * (T_jacket - T)
            )

            return [dC_A_dt, dT_dt]

        # Try multiple initial guesses across temperature range
        steady_states = []
        T_guesses = np.linspace(T_range[0], T_range[1], n_points)

        for T_guess in T_guesses:
            # Estimate C_A from material balance at this T
            r_guess = k0 * C_A_feed * np.exp(-E / T_guess)
            C_A_guess = C_A_feed / (1 + (V / F) * r_guess / C_A_feed)
            C_A_guess = np.clip(C_A_guess, 0.0, C_A_feed)

            try:
                solution, info, ier, msg = fsolve(
                    steady_state_equations,
                    [C_A_guess, T_guess],
                    full_output=True,
                )

                if ier == 1:  # Solution converged
                    C_A_sol, T_sol = solution

                    # Check if solution is physical and unique
                    if (
                        0 <= C_A_sol <= C_A_feed
                        and T_range[0] <= T_sol <= T_range[1]
                        and not any(
                            np.allclose([C_A_sol, T_sol], ss, rtol=1e-3) for ss in steady_states
                        )
                    ):
                        steady_states.append((C_A_sol, T_sol))

            except Exception:
                continue

        # Sort by temperature (low to high)
        steady_states.sort(key=lambda x: x[1])

        return steady_states

    # def print_equations(self, simplify: bool = True):
    #     """
    #     Print symbolic equations using continuous-time notation.

    #     Parameters
    #     ----------
    #     simplify : bool
    #         If True, simplify expressions before printing
    #     """
    #     print("=" * 70)
    #     print(f"{self.__class__.__name__} (Continuous-Time)")
    #     print("=" * 70)
    #     print("Continuous Stirred-Tank Reactor with Cooling Jacket")
    #     print("\nReaction: A → B (exothermic)")
    #     print(f"\nState: x = [Cₐ, T]")
    #     print(f"Control: u = [T_jacket]")
    #     print(f"Dimensions: nx={self.nx}, nu={self.nu}")

    #     # Calculate and display characteristic parameters
    #     tau = self.compute_residence_time()
    #     print(f"\nCharacteristic Parameters:")
    #     print(f"  Residence time τ = V/F = {tau:.2f} s")
    #     print(f"  Damköhler number Da(350K) = {self.compute_damkohler_number(350.0):.2f}")
    #     print(f"  Damköhler number Da(400K) = {self.compute_damkohler_number(400.0):.2f}")

    #     print("\nDynamics: dx/dt = f(x, u)")
    #     for var, expr in zip(self.state_vars, self._f_sym):
    #         expr_sub = self.substitute_parameters(expr)
    #         if simplify:
    #             expr_sub = sp.simplify(expr_sub)
    #         print(f"  d{var}/dt = {expr_sub}")

    #     print("\nPhysical Interpretation:")
    #     print("  - Cₐ: Reactor concentration [mol/L]")
    #     print("  - T: Reactor temperature [K]")
    #     print("  - T_jacket: Cooling jacket temperature [K]")

    #     print("\nTypical Operating Range:")
    #     C_A_feed = self.parameters[sp.symbols("C_A_feed")]
    #     print(f"  - Cₐ: 0 - {C_A_feed} mol/L")
    #     print("  - T: 350 - 450 K")
    #     print("  - T_jacket: 280 - 360 K")

    #     print("\n⚠️  IMPORTANT - Multiple Steady States:")
    #     print("  CSTR can have 1, 2, or 3 steady states!")
    #     print("  Use find_steady_states() to locate all equilibria.")
    #     print("  Linearization and stability depend on operating point.")

    #     print("=" * 70)