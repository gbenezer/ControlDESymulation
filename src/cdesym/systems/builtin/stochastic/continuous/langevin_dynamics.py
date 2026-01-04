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
Langevin Dynamics - Classical Statistical Mechanics of Brownian Motion
=======================================================================

This module provides the Langevin equation, the fundamental model connecting
microscopic random motion to macroscopic diffusion. Langevin dynamics serve as:

- The foundation of statistical mechanics of Brownian motion (1908 landmark)
- The bridge between Newton's laws and stochastic differential equations
- The canonical model for molecular dynamics with thermal fluctuations
- The theoretical basis for the fluctuation-dissipation theorem
- A fundamental tool in computational statistical physics and molecular simulation

The Langevin equation represents the most important stochastic differential
equation in physics, connecting:
- Classical mechanics (Newton's F = ma)
- Statistical thermodynamics (temperature, fluctuation-dissipation)
- Stochastic processes (Ornstein-Uhlenbeck process)
- Computational physics (molecular dynamics, MCMC sampling)

Historical Context
------------------

**Einstein vs. Langevin (1905-1908):**

Two approaches to Brownian motion:

**Einstein (1905):**
- Studied particle POSITION: x(t)
- Derived diffusion equation: ∂p/∂t = D·∂²p/∂x²
- Diffusion coefficient: D = k_B·T/γ (Einstein relation)
- Macroscopic, statistical approach

**Langevin (1908):**
- Studied particle VELOCITY: v(t)
- Derived stochastic equation: m·dv = -γ·v·dt + √(2γ·k_B·T)·dW
- Microscopic, force-based approach
- More intuitive physically

**Breakthrough:**
Langevin connected Newton's law to thermal fluctuations explicitly.

**Key Insight:**
Friction (dissipation) and random force (fluctuation) are two sides
of same coin - both arise from molecular collisions.

**Legacy:**
- Founded modern statistical mechanics
- Established stochastic differential equations in physics
- Proved existence of atoms/molecules (via quantitative predictions)
- Enabled molecular dynamics simulations

Mathematical Formulation
------------------------

**Langevin Equation (Original Form):**

Newton's second law with friction and random forcing:
    m·dv = -γ·v·dt + F_external·dt + η·dt

where:
    - m: Particle mass [kg]
    - v: Velocity [m/s]
    - γ: Friction coefficient [kg/s]
    - F_external: Applied force (control, gravity, etc.)
    - η(t): Random force (white noise, thermal fluctuations)

**Stochastic Differential Equation Form:**

In Itô SDE notation:
    dv = -(γ/m)·v·dt + (F/m)·dt + √(2·k_B·T/m)·dW

where:
    - γ/m: Friction rate [1/s]
    - k_B: Boltzmann constant = 1.380649×10⁻²³ J/K
    - T: Absolute temperature [K]
    - W(t): Standard Wiener process

**This is Ornstein-Uhlenbeck Process!**

Velocity follows OU:
    dv = -α·v·dt + σ·dW

with:
    - α = γ/m: Friction rate
    - σ = √(2·k_B·T/m): Noise intensity

**Full 1D Langevin (Position + Velocity):**

State-space form:
    dx = v·dt
    dv = -(γ/m)·v·dt + (F/m)·dt + √(2·k_B·T/m)·dW

Two coupled SDEs: Position integrates velocity (deterministic),
velocity follows OU process (stochastic).

**Multidimensional (3D Space):**

    dx_i = v_i·dt
    dv_i = -(γ/m)·v_i·dt + (F_i/m)·dt + √(2·k_B·T/m)·dW_i

for i ∈ {x, y, z} with independent noise sources.

Fluctuation-Dissipation Theorem
--------------------------------

**Fundamental Principle of Statistical Mechanics:**

The same molecular collisions causing friction (dissipation) also cause
random forcing (fluctuation). These are NOT independent!

**Mathematical Statement:**

For Langevin equation with friction γ and noise intensity σ:
    σ² = 2·γ·k_B·T/m

This relates:
- Dissipation: γ (energy removal rate)
- Fluctuation: σ (energy injection rate)
- Temperature: T (thermal equilibrium)

**Physical Meaning:**

At thermal equilibrium:
- Friction removes kinetic energy: ⟨-γ·v²⟩
- Noise adds kinetic energy: ⟨σ²⟩
- Balance: Energy in = Energy out
- Result: Maxwell-Boltzmann distribution

**Consequences:**

1. **Cannot choose γ and σ independently:**
   If friction is large (large γ), noise must be large too (large σ).
   Both determined by temperature T and molecular properties.

2. **Equipartition theorem:**
   At equilibrium: ⟨(1/2)·m·v²⟩ = (1/2)·k_B·T

   Kinetic energy per degree of freedom = k_B·T/2

3. **Validation:**
   Check simulated velocity variance matches thermal expectation:
   Var[v] = k_B·T/m

**Einstein Relation:**

Diffusion coefficient D relates to friction:
    D = k_B·T/γ

This connects:
- Microscopic: Friction γ from molecular collisions
- Macroscopic: Diffusion D from spreading

Key Properties
--------------

**1. Two Time Scales:**

Fast scale: Velocity relaxation
    τ_v = m/γ (momentum relaxation time)

Slow scale: Position diffusion
    τ_x → ∞ (diffusive, no natural scale)

**Separation:**
For m → 0 (massless limit): τ_v → 0
    → Velocity instantly equilibrates
    → Reduces to: dx = (F/γ)·dt + √(2·k_B·T/γ)·dW

This is **overdamped Langevin equation** (position-only).

**2. Stationary Velocity Distribution:**

Maxwell-Boltzmann:
    p(v) = √(m/2πk_B·T)·exp(-m·v²/2k_B·T)

Gaussian with variance k_B·T/m (equipartition).

**3. Position Distribution:**

Non-stationary - diffuses indefinitely:
    x(t) ~ N(x₀ + (F/γ)·t, 2·D·t)

where D = k_B·T/γ is diffusion coefficient.

**4. Velocity Autocorrelation:**

    ⟨v(0)·v(t)⟩ = (k_B·T/m)·exp(-γ·t/m)

Exponential decay with time constant m/γ.

**5. Mean Square Displacement:**

Einstein's relation:
    ⟨(x(t) - x(0))²⟩ = 2·D·t

Linear growth (diffusive scaling).

**6. Connection to Thermodynamics:**

First Law: Energy conservation (on average)
Second Law: Entropy increases (irreversible)
Fluctuation-dissipation: Connects dynamics to equilibrium

Overdamped vs Underdamped
--------------------------

**Underdamped Langevin (Full - Implemented):**
    dx = v·dt
    dv = -(γ/m)·v·dt + (F/m)·dt + √(2·k_B·T/m)·dW

Two variables: position AND velocity.
Inertia matters: m·dv/dt term.

**Regime:** Light particles, low viscosity
- Examples: Electrons in metal, atoms in gas
- Time scale: m/γ << observation time

**Overdamped Langevin (Simplified):**
    dx = (F/γ)·dt + √(2·k_B·T/γ)·dW

One variable: position only (velocity instantaneously equilibrated).
No inertia: m → 0 limit.

**Regime:** Heavy particles, high viscosity
- Examples: Colloids in water, polymers in solution, proteins
- Time scale: m/γ → 0 (instantaneous velocity relaxation)

**When to Use Each:**

Underdamped:
- Fast dynamics (picoseconds to nanoseconds)
- Light particles
- Need velocity information

Overdamped:
- Slow dynamics (microseconds to seconds)
- Heavy particles in viscous media
- Position primary variable

**This Implementation:** Underdamped (full position + velocity).

Applications
------------

**1. Molecular Dynamics:**

**Canonical Ensemble (NVT):**
Constant number, volume, temperature.
Langevin thermostat maintains T:
    - Friction removes excess energy
    - Noise adds thermal energy
    - Equilibrium: Boltzmann distribution

**Protein Folding:**
Molecular dynamics of proteins in water:
    - Heavy atoms: Overdamped
    - Hydrogen atoms: Underdamped
    - Solvent: Implicit (friction + noise)

**2. Computational Physics:**

**Sampling Boltzmann Distribution:**

For potential U(x), equilibrium distribution:
    p(x) ∝ exp(-U(x)/k_B·T)

Langevin dynamics with F = -∇U samples this distribution:
    dx = v·dt
    dv = -(γ/m)·v·dt - (1/m)·∇U·dt + √(2·k_B·T/m)·dW

After long time: Positions distributed as Boltzmann.

**Applications:**
- Statistical mechanics calculations
- Partition function estimation
- Free energy computations

**3. Stochastic Optimization:**

**Langevin MCMC:**

For sampling posterior p(θ|data) ∝ exp(-E(θ)):
    dθ = -∇E·dt + √(2·T)·dW

Explores probability landscape with thermal noise.

**Simulated Annealing:**
Gradually reduce T → 0:
- High T: Explore widely (escape local minima)
- Low T: Exploit (converge to global minimum)

**4. Colloidal Dynamics:**

**Optical Tweezers:**
Trapped colloidal particle:
    - Potential: Harmonic trap U = (1/2)·k·x²
    - Friction: Stokes drag γ = 6πηr
    - Thermal noise: Room temperature

Measure: Stiffness k, friction γ, temperature T.

**5. Biophysics:**

**Single-Molecule Experiments:**
- DNA stretching
- Protein unfolding (AFM)
- Molecular motors

**6. Machine Learning:**

**Stochastic Gradient Langevin Dynamics (SGLD):**
    dθ = -∇L(θ)·dt + √(2·η)·dW

where L is loss function.

Adds noise to SGD → explores loss landscape → Bayesian inference.

Physical Parameters and Scaling
--------------------------------

**Friction Coefficient γ:**

For sphere of radius r in fluid of viscosity η:
    γ = 6πηr (Stokes' law)

**Examples:**
- Water (η ≈ 10⁻³ Pa·s), r = 1 μm: γ ≈ 2×10⁻⁸ kg/s
- Air (η ≈ 2×10⁻⁵ Pa·s), r = 1 μm: γ ≈ 4×10⁻¹⁰ kg/s

**Time Scales:**

Momentum relaxation: τ_v = m/γ

**Examples:**
- Electron in metal: m = 10⁻³⁰ kg, γ ≈ 10⁻²⁰ → τ_v ~ 10⁻¹⁰ s
- Protein in water: m = 10⁻²⁰ kg, γ ≈ 10⁻⁸ → τ_v ~ 10⁻¹² s
- Colloidal particle: m = 10⁻¹⁵ kg, γ ≈ 10⁻⁸ → τ_v ~ 10⁻⁷ s
- Pollen grain: m = 10⁻¹² kg, γ ≈ 10⁻⁸ → τ_v ~ 10⁻⁴ s

**Diffusion Coefficient:**

Einstein-Stokes relation:
    D = k_B·T/γ = k_B·T/(6πηr)

**Examples (room temperature, T = 300 K):**
- Protein in water: D ~ 10⁻¹¹ m²/s
- Colloidal particle: D ~ 10⁻¹³ m²/s

**Thermal Energy:**
    k_B·T ≈ 4.1×10⁻²¹ J at 300 K

Typical kinetic energy scale.

Numerical Integration
---------------------

**Challenges:**

1. **Multiple Time Scales:**
   - Fast: Velocity relaxation (τ_v = m/γ)
   - Slow: Position diffusion
   - Stiffness if m << γ

2. **Overdamped Limit:**
   - For m → 0: Velocity equation stiff
   - Need small dt or implicit methods
   - Or use overdamped approximation directly

**Methods:**

**1. Euler-Maruyama (Standard):**
    x[k+1] = x[k] + v[k]·Δt
    v[k+1] = v[k] - (γ/m)·v[k]·Δt + (F/m)·Δt + √(2·k_B·T/m)·√Δt·Z[k]

Requires: Δt << m/γ for stability.

**2. BBK (Brünger-Brooks-Karplus):**
Position Verlet variant for Langevin:
    - Better energy conservation
    - Standard in molecular dynamics (LAMMPS, GROMACS)

**3. Exact for Harmonic Potential:**
If F = -k·x (harmonic), exact solution exists.
Can sample directly from Gaussian.

**4. Overdamped Limit:**
If m/γ << Δt (heavy particle or slow sampling):
    dx = (F/γ)·dt + √(2·k_B·T/γ)·dW

Eliminates velocity, single SDE.

**Recommended:**
- Underdamped (light particles): Euler-Maruyama, dt ~ 0.1·(m/γ)
- Overdamped (heavy particles): Use simplified equation

Applications to Sampling
-------------------------

**Langevin Monte Carlo:**

Sample from distribution p(x) ∝ exp(-U(x)/k_B·T):

    dx = v·dt
    dv = -(γ/m)·v·dt - (1/m)·∇U·dt + √(2·k_B·T/m)·dW

After transient, positions distributed as p(x).

**Advantages:**
- Efficient exploration (momentum)
- Gradient information (∇U guides search)
- Thermal noise (escapes local minima)

**Applications:**
- Bayesian inference (posterior sampling)
- Machine learning (explore loss landscape)
- Statistical mechanics (compute partition functions)

**Comparison with Metropolis-Hastings:**
- MH: Discrete jumps, memoryless
- Langevin: Continuous motion, momentum
- Langevin often more efficient (fewer rejections)

**Stochastic Gradient Langevin Dynamics (SGLD):**

For machine learning with loss L(θ):
    dθ = -∇L(θ)·dt + √(2·η)·dW

Mini-batch gradient ≈ ∇L + noise:
- Noise from mini-batching serves as thermal fluctuations
- Annealing: Reduce learning rate η → 0
- Converges to Bayesian posterior

Fluctuation-Dissipation Applications
-------------------------------------

**Measuring Temperature:**

From trajectory, estimate:
- Friction: γ from velocity autocorrelation decay
- Temperature: T from velocity variance
- Verify: σ² = 2·γ·k_B·T/m

**Calibrating Optical Tweezers:**

Trapped particle:
- Measure position fluctuations
- Fit to Langevin with U = (1/2)·k·x²
- Extract: k (trap stiffness), γ, T

**Rheology:**

Microrheology using Brownian probe:
- Particle in complex fluid
- Generalized Langevin with memory
- Extract: Viscoelastic moduli from fluctuations

Common Pitfalls
---------------

1. **Wrong Noise Intensity:**
   - Must use σ = √(2·γ·k_B·T/m), not arbitrary
   - Violates fluctuation-dissipation theorem
   - Non-equilibrium steady state instead of Boltzmann

2. **Overdamped When Shouldn't:**
   - Using dx = (F/γ)·dt + noise when m/γ not small
   - Loses inertial effects
   - Check: Is m/γ << observation time?

3. **Ignoring Mass:**
   - Setting m = 1 without dimensional analysis
   - Breaks physical units
   - Einstein relation D = k_B·T/γ becomes wrong

4. **Temperature Units:**
   - T in Kelvin, k_B in J/K
   - Energy in Joules, not eV or other units
   - Dimensional consistency critical

5. **Equilibration Time:**
   - Need time >> m/γ to reach thermal equilibrium
   - Initial velocity distribution may not be Maxwellian
   - Discard initial transient

6. **Integrator Choice:**
   - Standard SDE integrators may not preserve detailed balance
   - Use specialized integrators (BBK, BAOAB) for sampling

Historical Significance
-----------------------

**Scientific Revolution:**

Langevin equation (1908) was revolutionary because:

1. **Proved Atomic Theory:**
   - Quantitative predictions matched experiments
   - Perrin's experiments (1909) confirmed
   - Nobel Prize 1926 to Perrin

2. **Unified Frameworks:**
   - Newton's laws (deterministic)
   - Thermodynamics (statistical)
   - Stochastic processes (random)

3. **Founded Computational Physics:**
   - Molecular dynamics (MD) simulations
   - Monte Carlo methods
   - Modern computational chemistry/biology

4. **Established SDE in Physics:**
   - Before: Differential equations deterministic
   - After: Stochastic DEs fundamental
   - Enabled quantum mechanics developments

**Modern Legacy:**

- Every MD simulation uses Langevin (or variants)
- Foundation of non-equilibrium statistical mechanics
- Basis for understanding fluctuations in small systems
- Template for stochastic modeling in all sciences

"""


import numpy as np
import sympy as sp

from cdesym.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem


class LangevinDynamics(ContinuousStochasticSystem):
    """
    Langevin dynamics - particle in viscous fluid with thermal noise.

    The fundamental equation of statistical mechanics, combining Newton's law
    with friction and thermal fluctuations. Provides the microscopic foundation
    for Brownian motion and the bridge between mechanics and thermodynamics.

    Stochastic Differential Equations
    ----------------------------------
    Full Langevin equation (position + velocity):

        dx = v·dt
        dv = -(γ/m)·v·dt + (F/m)·dt + √(2·k_B·T/m)·dW

    where:
        - x: Position [m]
        - v: Velocity [m/s]
        - m: Particle mass [kg]
        - γ: Friction coefficient [kg/s]
        - F: External force (control, potential gradient) [N]
        - k_B: Boltzmann constant = 1.380649×10⁻²³ J/K
        - T: Temperature [K]
        - W: Standard Wiener process

    **This is OU Process for Velocity:**
    Velocity component is Ornstein-Uhlenbeck process.

    Physical Interpretation
    -----------------------
    **Newton's Law with Fluctuations:**

    m·dv/dt = F_total
            = F_external - γ·v + F_random

    Components:
    1. F_external: Applied force (gravity, fields, control)
    2. -γ·v: Friction (Stokes drag, dissipation)
    3. F_random: Thermal fluctuations (molecular collisions)

    **Friction:**
    For sphere in fluid (Stokes' law):
        γ = 6π·η·r

    where η is viscosity, r is radius.

    **Thermal Noise:**
    Fluctuation-dissipation theorem:
        ⟨F_random(t)·F_random(s)⟩ = 2·γ·k_B·T·δ(t-s)

    White noise with intensity √(2·γ·k_B·T).

    **Temperature:**
    Sets equilibrium energy scale:
        ⟨(1/2)·m·v²⟩ = (1/2)·k_B·T (equipartition)

    Key Features
    ------------
    **Fluctuation-Dissipation:**
    Friction γ and noise intensity related by temperature.
    Cannot choose independently!

    **Thermal Equilibrium:**
    Velocity distribution: Maxwell-Boltzmann
        p(v) ∝ exp(-m·v²/2k_B·T)

    **Two Time Scales:**
    - Fast: Velocity relaxation τ_v = m/γ
    - Slow: Position diffusion (no natural scale)

    **Markov Property:**
    Future depends only on current (x, v), not history.

    **Ergodicity:**
    Long trajectory → Boltzmann distribution.

    Mathematical Properties
    -----------------------
    **Velocity (OU Process):**

    For F = 0 (free particle):
    Mean: E[v(t)] = v₀·exp(-γ·t/m)
    Variance: Var[v(t)] = (k_B·T/m)·(1 - exp(-2γ·t/m))

    Asymptotic: Var[v(∞)] = k_B·T/m (equipartition)

    **Position (Diffusion):**

    Mean: E[x(t)] = x₀ + (m/γ)·v₀·(1 - exp(-γ·t/m))
    Variance: Var[x(t)] = 2·D·t - 2·D·(m/γ)·(1 - exp(-γ·t/m))

    Long time: Var[x] ≈ 2·D·t (Einstein diffusion)

    **Diffusion Coefficient:**
        D = k_B·T/γ (Einstein relation)

    Physical Interpretation
    -----------------------
    **Mass m:**
    - Particle inertia
    - Heavier → slower velocity changes
    - Time scale: m/γ

    **Friction γ:**
    - Energy dissipation rate
    - Larger → faster equilibration
    - From Stokes: γ = 6π·η·r

    **Temperature T:**
    - Thermal energy reservoir
    - Higher T → larger fluctuations
    - Sets equilibrium distribution

    **Boltzmann k_B:**
    - Fundamental constant
    - Connects microscopic and macroscopic
    - Value: 1.380649×10⁻²³ J/K

    State Space
    -----------
    State: X = [x, v] ∈ ℝ²
        - x: Position (unbounded, diffuses)
        - v: Velocity (Boltzmann distributed at equilibrium)

    Control: u = F/m ∈ ℝ (optional)
        - Acceleration input
        - Can be potential gradient: -∇U/m

    Noise: w ∈ ℝ
        - Single Wiener process
        - Enters velocity equation (thermal fluctuations)

    Parameters
    ----------
    m : float, default=1e-15
        Particle mass [kg]
        - Colloidal particle: ~10⁻¹⁵ kg
        - Protein: ~10⁻²⁰ kg
        - Affects time scale m/γ

    gamma : float, default=1e-8
        Friction coefficient [kg/s]
        - From Stokes: 6π·η·r
        - Water, 1μm sphere: ~10⁻⁸ kg/s

    T : float, default=300.0
        Temperature [K]
        - Room temperature: 300 K
        - Physiological: 310 K
        - Sets thermal energy k_B·T

    k_B : float, default=1.380649e-23
        Boltzmann constant [J/K]
        - Fundamental constant (fixed)

    Stochastic Properties
    ---------------------
    - System Type: LINEAR (no external force)
    - Noise Type: ADDITIVE (constant)
    - SDE Type: Itô
    - Noise Dimension: nw = 1
    - Stationary: Velocity (yes), Position (no)
    - Ergodic: Yes (for velocity distribution)
    - Physical: Yes (fluctuation-dissipation satisfied)

    Applications
    ------------
    **1. Molecular Dynamics:**
    - Thermostat (constant T simulations)
    - Protein folding simulations
    - Material properties

    **2. Bayesian Sampling:**
    - Langevin MCMC
    - Posterior sampling
    - Optimization with noise

    **3. Colloidal Physics:**
    - Optical tweezers experiments
    - Rheology measurements
    - Single-molecule biophysics

    **4. Machine Learning:**
    - Stochastic gradient Langevin dynamics
    - Bayesian neural networks
    - Exploration in RL

    **5. Statistical Mechanics:**
    - Equilibrium sampling
    - Free energy calculations
    - Rare event simulation

    Numerical Integration
    ---------------------
    **Time Step Selection:**

    Must resolve velocity relaxation:
        Δt < m/γ (typically 0.1·m/γ)

    For colloidal particle (m/γ ~ 10⁻⁷ s):
        Δt ~ 10⁻⁸ s

    **Overdamped Approximation:**

    If observation time >> m/γ, use:
        dx = (F/γ)·dt + √(2·D)·dW

    Much larger Δt possible (no velocity constraint).

    Validation
    ----------
    **Physical Checks:**

    1. **Equipartition:**
       Var[v] = k_B·T/m (after equilibration)

    2. **Einstein Relation:**
       D_measured = Var[x]/2t should equal k_B·T/γ

    3. **Maxwell-Boltzmann:**
       Velocity histogram should be Gaussian with variance k_B·T/m

    4. **Autocorrelation:**
       ⟨v(0)·v(t)⟩ = (k_B·T/m)·exp(-γ·t/m)

    Comparison with Other Models
    -----------------------------
    **vs. Brownian Motion:**
    - BM: Position only, no velocity
    - Langevin: Position + velocity (more complete)

    **vs. Ornstein-Uhlenbeck:**
    - OU: Velocity equation only
    - Langevin: Position + velocity

    **vs. Stochastic Double Integrator:**
    - DI: No friction (conservative)
    - Langevin: With friction (dissipative, equilibrium)

    Limitations
    -----------
    - Linear (no potential in base class)
    - 1D space (extend to 3D for realism)
    - Constant γ, T (no hydrodynamic interactions)
    - Additive noise (no multiplicative)
    - Markov (memoryless, no viscoelasticity)

    Extensions
    ----------
    - Potential: Add U(x) via control F = -∇U
    - 3D: Three spatial dimensions
    - Generalized Langevin: Memory kernel (viscoelasticity)
    - Active particles: Self-propulsion
    - Hydrodynamic interactions: Many-body γ_ij

    See Also
    --------
    OrnsteinUhlenbeck : Velocity-only version
    StochasticDoubleIntegrator : No friction version
    """

    def define_system(
        self,
        m: float = 1e-15,
        gamma: float = 1e-8,
        T: float = 300.0,
        k_B: float = 1.380649e-23,
    ):
        """
        Define Langevin dynamics (position + velocity with thermal noise).

        Parameters
        ----------
        m : float, default=1e-15
            Particle mass [kg]
            - Colloidal particle: ~10⁻¹⁵ kg
            - Protein: ~10⁻²⁰ kg
            - Determines time scale m/γ

        gamma : float, default=1e-8
            Friction coefficient [kg/s]
            - From Stokes: γ = 6π·η·r
            - Water, 1μm sphere: ~10⁻⁸ kg/s
            - Controls relaxation rate

        T : float, default=300.0
            Temperature [K]
            - Room temperature: 300 K
            - Physiological: 310 K
            - Sets thermal energy k_B·T

        k_B : float, default=1.380649e-23
            Boltzmann constant [J/K]
            - Physical constant (don't change)

        Raises
        ------
        ValueError
            If m, gamma, T, or k_B are non-positive

        Notes
        -----
        **Fluctuation-Dissipation:**

        Noise intensity MUST be:
            σ = √(2·γ·k_B·T/m)

        This is not a free parameter! Determined by γ, T.

        **Time Scales:**

        Velocity relaxation: τ_v = m/γ
        - Fast equilibration of velocity
        - After ~5·τ_v: Velocity is Maxwellian

        Position diffusion: No characteristic time
        - Variance grows linearly: 2·D·t
        - Diffusion coefficient: D = k_B·T/γ

        **Regime Classification:**

        Underdamped: m/γ >> observation time
        - Ballistic short-time: ⟨x²⟩ ~ t² (free flight)
        - Diffusive long-time: ⟨x²⟩ ~ t (Brownian)

        Overdamped: m/γ << observation time
        - Pure diffusion: ⟨x²⟩ ~ t always
        - Can use simplified equation (position only)

        **Physical Examples:**

        Colloidal particle (1 μm, water, 300 K):
        - m ~ 10⁻¹⁵ kg
        - γ ~ 10⁻⁸ kg/s
        - τ_v ~ 10⁻⁷ s (100 ns)
        - D ~ 4×10⁻¹³ m²/s

        Protein in water:
        - m ~ 10⁻²⁰ kg
        - γ ~ 10⁻¹¹ kg/s
        - τ_v ~ 10⁻⁹ s (1 ns)
        - D ~ 10⁻¹¹ m²/s

        **Validation:**

        Check equilibrium properties:
        1. Velocity variance: k_B·T/m
        2. Diffusion coefficient: k_B·T/γ
        3. Velocity autocorrelation: exp(-γ·t/m)
        """
        if m <= 0:
            raise ValueError(f"mass must be positive, got {m}")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if T <= 0:
            raise ValueError(f"temperature must be positive, got {T}")
        if k_B <= 0:
            raise ValueError(f"Boltzmann constant must be positive, got {k_B}")

        # Store physical parameters
        self.m = m
        self.gamma = gamma
        self.T = T
        self.k_B = k_B

        # Compute derived quantities
        self.tau_v = m / gamma  # Velocity relaxation time
        self.D = k_B * T / gamma  # Diffusion coefficient
        self.sigma_v = np.sqrt(2 * gamma * k_B * T / m)  # Noise intensity (derived!)

        # State variables
        x, v = sp.symbols("x v", real=True)
        u = sp.symbols("u", real=True)  # External force per unit mass

        # Parameters (store as symbols for compatibility)
        m_sym = sp.symbols("m", positive=True)
        gamma_sym = sp.symbols("gamma", positive=True)
        T_sym = sp.symbols("T", positive=True)
        k_B_sym = sp.symbols("k_B", positive=True)

        self.state_vars = [x, v]
        self.control_vars = [u]

        # DRIFT (Deterministic dynamics)
        # dx/dt = v
        # dv/dt = -(γ/m)·v + u (u is force per unit mass, e.g., F/m or -∇U/m)
        self._f_sym = sp.Matrix(
            [
                v,
                -(gamma_sym / m_sym) * v + u,
            ],
        )

        self.parameters = {
            m_sym: m,
            gamma_sym: gamma,
            T_sym: T,
            k_B_sym: k_B,
        }
        self.order = 1

        # DIFFUSION (Thermal noise via fluctuation-dissipation)
        # Noise on velocity only (thermal fluctuations are forces)
        # σ = √(2·γ·k_B·T/m) - DETERMINED by fluctuation-dissipation!
        sigma_thermal = sp.sqrt(2 * gamma_sym * k_B_sym * T_sym / m_sym)

        self.diffusion_expr = sp.Matrix(
            [
                [0],
                [sigma_thermal],
            ],
        )

        # Itô SDE
        self.sde_type = "ito"

        # Output: Position (typical in experiments)
        self._h_sym = sp.Matrix([x])

    def setup_equilibria(self):
        """
        Set up equilibrium points.

        For free Langevin (u=0), equilibrium is origin with thermal velocity.
        """
        self.add_equilibrium(
            "thermal_equilibrium",
            x_eq=np.array([0.0, 0.0]),
            u_eq=np.array([0.0]),
            verify=True,
            stability="marginally_stable",
            notes="Thermal equilibrium. Position drifts (non-stationary), "
            "velocity is Maxwell-Boltzmann distributed. Fluctuation-dissipation satisfied.",
        )
        self.set_default_equilibrium("thermal_equilibrium")

    def get_thermal_velocity_std(self) -> float:
        """
        Get thermal velocity standard deviation from equipartition.

        σ_v = √(k_B·T/m)

        Returns
        -------
        float
            Thermal velocity std [m/s]

        Notes
        -----
        From equipartition: ⟨(1/2)·m·v²⟩ = (1/2)·k_B·T

        Examples
        --------
        >>> langevin = LangevinDynamics(m=1e-15, T=300)
        >>> sigma_v = langevin.get_thermal_velocity_std()
        >>> print(f"Thermal velocity: {sigma_v:.3e} m/s")
        """
        return np.sqrt(self.k_B * self.T / self.m)

    def get_diffusion_coefficient(self) -> float:
        """
        Get diffusion coefficient D = k_B·T/γ (Einstein relation).

        Returns
        -------
        float
            Diffusion coefficient [m²/s]

        Examples
        --------
        >>> langevin = LangevinDynamics(gamma=1e-8, T=300)
        >>> D = langevin.get_diffusion_coefficient()
        >>> print(f"Diffusion: {D:.3e} m²/s")
        """
        return self.D

    def get_relaxation_time(self) -> float:
        """
        Get velocity relaxation time τ_v = m/γ.

        Returns
        -------
        float
            Relaxation time [s]

        Examples
        --------
        >>> langevin = LangevinDynamics(m=1e-15, gamma=1e-8)
        >>> tau = langevin.get_relaxation_time()
        >>> print(f"Relaxation: {tau:.3e} s")
        """
        return self.tau_v

    def check_fluctuation_dissipation(self) -> dict:
        """
        Verify fluctuation-dissipation theorem is satisfied.

        Returns
        -------
        dict
            Expected vs implemented noise intensity

        Examples
        --------
        >>> langevin = LangevinDynamics()
        >>> check = langevin.check_fluctuation_dissipation()
        >>> print(f"FDT satisfied: {check}")
        """
        # Expected from FDT
        sigma_expected = np.sqrt(2 * self.gamma * self.k_B * self.T / self.m)

        # Implemented
        sigma_implemented = self.sigma_v

        return {
            "expected": sigma_expected,
            "implemented": sigma_implemented,
            "ratio": sigma_implemented / sigma_expected,
            "FDT_satisfied": np.isclose(sigma_expected, sigma_implemented, rtol=1e-6),
        }


# Convenience functions
def create_colloidal_particle(
    radius: float = 1e-6,
    density: float = 1000.0,
    fluid_viscosity: float = 1e-3,
    temperature: float = 300.0,
) -> LangevinDynamics:
    """
    Create Langevin model for spherical colloidal particle.

    Automatically computes mass (from density) and friction (Stokes' law).

    Parameters
    ----------
    radius : float, default=1e-6
        Particle radius [m] (1 μm default)
    density : float, default=1000.0
        Particle density [kg/m³]
    fluid_viscosity : float, default=1e-3
        Fluid viscosity [Pa·s] (water at 20°C)
    temperature : float, default=300.0
        Temperature [K]

    Returns
    -------
    LangevinDynamics

    Examples
    --------
    >>> # 1 μm polystyrene bead in water
    >>> bead = create_colloidal_particle(
    ...     radius=1e-6,
    ...     density=1050,  # Polystyrene
    ...     fluid_viscosity=1e-3,  # Water
    ...     temperature=300
    ... )
    """
    # Mass from volume and density
    volume = (4 / 3) * np.pi * radius**3
    mass = volume * density

    # Friction from Stokes' law
    gamma = 6 * np.pi * fluid_viscosity * radius

    return LangevinDynamics(m=mass, gamma=gamma, T=temperature)


def create_optical_tweezer_system(
    particle_radius: float = 500e-9,
    trap_stiffness: float = 1e-6,
    temperature: float = 300.0,
) -> tuple:
    """
    Create Langevin system for optical tweezers experiment.

    Returns both Langevin dynamics and trap potential info.

    Parameters
    ----------
    particle_radius : float, default=500e-9
        Particle radius [m] (500 nm default)
    trap_stiffness : float, default=1e-6
        Optical trap stiffness [N/m]
    temperature : float, default=300.0
        Temperature [K]

    Returns
    -------
    tuple
        (LangevinDynamics, trap_info_dict)

    Notes
    -----
    In optical tweezers:
    - Potential: U(x) = (1/2)·k·x²
    - Force: F(x) = -k·x
    - Equilibrium: Gaussian around trap center

    Examples
    --------
    >>> # Typical optical tweezers setup
    >>> system, trap = create_optical_tweezer_system(
    ...     particle_radius=500e-9,
    ...     trap_stiffness=1e-6
    ... )
    >>> print(f"Corner frequency: {trap['corner_frequency']:.1f} Hz")
    """
    # Create particle
    particle = create_colloidal_particle(
        radius=particle_radius,
        density=1050,  # Polystyrene typical
        fluid_viscosity=1e-3,
        temperature=temperature,
    )

    # Trap characteristics
    gamma = particle.gamma
    k_B = particle.k_B

    # Corner frequency: f_c = k/(2π·γ)
    corner_freq = trap_stiffness / (2 * np.pi * gamma)

    # Expected position variance in trap: k_B·T/k
    position_var = k_B * temperature / trap_stiffness

    trap_info = {
        "stiffness": trap_stiffness,
        "corner_frequency": corner_freq,
        "position_std": np.sqrt(position_var),
        "control_function": lambda x: -trap_stiffness * x[0] / particle.m,
    }

    return particle, trap_info


def create_sgld_sampler(
    learning_rate: float = 0.01,
    temperature: float = 1.0,
) -> LangevinDynamics:
    """
    Create Stochastic Gradient Langevin Dynamics sampler for ML.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size (analogous to 1/γ)
    temperature : float, default=1.0
        Exploration temperature

    Returns
    -------
    LangevinDynamics

    Notes
    -----
    SGLD for Bayesian inference:
        dθ = -∇E(θ)·dt + √(2·η)·dW

    Maps to Langevin with:
    - Position → parameters θ
    - Force → -∇E (gradient of energy/loss)
    - Temperature → exploration level

    Examples
    --------
    >>> # Bayesian neural network training
    >>> sampler = create_sgld_sampler(
    ...     learning_rate=0.01,
    ...     temperature=1.0
    ... )
    """
    # Map ML parameters to physical parameters
    # For overdamped limit: set m very small
    m_eff = 1e-10  # Nearly massless
    gamma_eff = 1.0 / learning_rate
    k_B_eff = 1.0  # Normalized

    return LangevinDynamics(
        m=m_eff,
        gamma=gamma_eff,
        T=temperature,
        k_B=k_B_eff,
    )
