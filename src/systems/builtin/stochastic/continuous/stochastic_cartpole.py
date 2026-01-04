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
Stochastic Inverted Pendulum on Cart - Underactuated System with Noise
=======================================================================

This module provides the stochastic cart-pole (inverted pendulum on cart),
combining the challenging nonlinear underactuated dynamics with process noise
and disturbances. The stochastic cart-pole serves as:

- The benchmark for robust control of underactuated systems under uncertainty
- A test case for nonlinear stochastic control (swing-up and stabilization with noise)
- An illustration of noise-induced falling from unstable equilibria
- A model for balance control under random perturbations (humanoid robots, Segways)
- A challenge problem for robust reinforcement learning algorithms

The stochastic cart-pole extends the deterministic benchmark by adding realistic
disturbances that every physical balancing system experiences:
- Ground vibrations and irregularities
- Wind gusts and air resistance fluctuations
- Sensor noise and actuator uncertainty
- Model mismatch and unmodeled dynamics

This creates a much more realistic and challenging control problem where:
- Perfect stabilization is impossible (noise prevents exact balance)
- Control must be robust to disturbances (not just nominal performance)
- Falling becomes a probabilistic event (mean time to failure analysis)
- State estimation is critical (noisy measurements of unstable system)

Physical Context
----------------

**Balancing Under Uncertainty:**

Every real balancing system experiences random disturbances:

1. **Segway/Self-Balancing Vehicles:**
   - Ground irregularities (bumps, slopes)
   - Wind gusts
   - Rider movement uncertainties
   - Wheel slip variations

2. **Humanoid Robots:**
   - Joint backlash and compliance
   - Ground contact uncertainty
   - External pushes and contacts
   - Sensor noise (IMU drift, encoders)

3. **Rocket/Missile Stabilization:**
   - Atmospheric turbulence
   - Thrust variations
   - Aerodynamic fluctuations
   - Structural vibrations

4. **Inverted Pendulum Labs:**
   - Track friction variations
   - Actuator noise (motor ripple)
   - Encoder quantization
   - Electromagnetic interference

**Why Stochastic Model is Essential:**

Deterministic cart-pole: Assumes perfect knowledge and no disturbances.
- Can stabilize indefinitely
- Perfect tracking possible
- Unrealistic for real systems

Stochastic cart-pole: Realistic model with disturbances.
- Stabilization is probabilistic (minimize falling probability)
- Perfect balance impossible (noise prevents exact equilibrium)
- Must design for robustness, not just nominal performance
- Falling is rare event (need statistical analysis)

**Key Challenges:**

1. **Unstable Equilibrium + Noise:**
   - Small disturbance → exponential growth (if not controlled)
   - Noise continuously perturbs system
   - Controller must act fast and accurately

2. **Underactuation:**
   - 4 states, 1 control
   - Cannot independently control all modes
   - Noise affects uncontrolled modes directly

3. **Nonlinearity:**
   - sin(θ), cos(θ) terms
   - Linearization valid only near upright
   - Noise can push outside linear regime

4. **State Estimation:**
   - Noisy measurements of unstable system
   - Extended Kalman Filter challenging
   - Estimation errors can cause falling

Mathematical Formulation
------------------------

**Stochastic Differential Equations:**

The cart-pole with random disturbances:

    dx = ẋ·dt
    dẋ = F_cart·dt + σ_x·dW_x
    dθ = θ̇·dt  
    dθ̇ = F_pole·dt + σ_θ·dW_θ

where:
    - x: Cart position [m]
    - ẋ: Cart velocity [m/s]
    - θ: Pole angle from upward vertical [rad]
    - θ̇: Pole angular velocity [rad/s]
    - F_cart, F_pole: Nonlinear coupling (same as deterministic)
    - σ_x: Horizontal disturbance intensity [m/(s²·√s)]
    - σ_θ: Angular disturbance intensity [rad/(s²·√s)]
    - W_x, W_θ: Independent Wiener processes

**Deterministic Part (from Lagrangian):**

Cart equation:
    (M + m)·ẍ + m·L·θ̈·cos(θ) - m·L·θ̇²·sin(θ) = F

Pole equation:
    m·L²·θ̈ + m·L·ẍ·cos(θ) - m·g·L·sin(θ) = 0

Solving for accelerations:
    
    θ̈ = [g·sin(θ) - cos(θ)·(F + m·L·θ̇²·sin(θ))/(M+m)] / 
         [L·(4/3 - m·cos²(θ)/(M+m))]

    ẍ = [F + m·L·(θ̇²·sin(θ) - θ̈·cos(θ))] / (M+m)

**Stochastic Part:**

Noise can enter in different ways:

**Option 1: Force Disturbances (Implemented)**
Noise on cart and pole equations:
    F_total = F_control + w_cart(t)
    τ_total = w_pole(t)

This models:
- Cart: Ground irregularities, wind on cart
- Pole: Direct wind on pole, measurement noise

**Option 2: Velocity Noise**
Noise on velocity derivatives directly:
    dẋ = (...)·dt + σ_x·dW_x
    dθ̇ = (...)·dt + σ_θ·dW_θ

Simpler but less physically motivated.

**This Implementation:** Option 2 (additive velocity noise).

Noise-Induced Falling
----------------------

**Critical Phenomenon:**

At upright equilibrium (θ=0, all velocities=0), the system is unstable.
Without control, exponentially diverges.

With control, noise creates falling events:
- Most time: Small fluctuations around upright
- Rare events: Large noise realization → angle exceeds threshold → falling

**Mean Time to Failure (MTF):**

Expected time until |θ| > θ_max (failure criterion):
    MTF ≈ exp(θ_max²·κ/σ_θ²)

where κ is related to control gain.

**Exponential Dependence:**
- Small noise: MTF ~ years (very reliable)
- Medium noise: MTF ~ hours
- Large noise: MTF ~ seconds (unusable)

**Implications for Design:**

1. **Noise Tolerance:** 
   - Larger θ_max → more forgiving (but less precise)
   - Stronger control → larger κ → longer MTF

2. **Reliability Requirement:**
   - Segway: Want MTF > 1000 hours
   - Robot demo: Want MTF > 10 minutes
   - Determines acceptable noise levels

3. **Sensor Quality:**
   - Better sensors → smaller σ_θ → longer MTF
   - Cost-reliability tradeoff

Robust Control Challenges
--------------------------

**Why Harder Than Deterministic:**

1. **Perfect Stabilization Impossible:**
   - Noise prevents exact θ = 0
   - Best: Keep |θ| small with high probability
   - Probabilistic specification: P(|θ| < 0.1 rad) > 0.95

2. **State Estimation:**
   - Measurements noisy (θ measured with error)
   - Kalman filter on unstable system challenging
   - Estimation errors compound instability

3. **Worst-Case vs Average:**
   - LQG minimizes expected cost (average)
   - May ignore rare large disturbances (tail risk)
   - Robust control focuses on worst-case

4. **Model Uncertainty:**
   - Don't know exact M, m, L, g
   - Noise models this uncertainty
   - Need robust design (H∞, μ-synthesis)

**Control Approaches:**

**1. LQG (Linear-Quadratic-Gaussian):**
- Linearize around upright
- Design LQR + Kalman filter
- Optimal for Gaussian noise
- Good for small deviations

**Limitations:**
- Only valid near upright (linear regime)
- Ignores constraints |θ| < θ_max
- Can violate safety in large noise

**2. Robust LQG (H∞ / μ-synthesis):**
- Design for worst-case disturbances
- Guarantees performance under bounded noise
- More conservative (larger control effort)

**3. Stochastic MPC:**
- Explicit constraints: P(|θ| < θ_max) ≥ 0.99
- Nonlinear model (valid for large angles)
- Chance constraints (probabilistic safety)
- Computational cost

**4. Risk-Sensitive Control:**
- Minimize: E[cost] + λ·Var[cost]
- Penalizes variance (risk aversion)
- Avoids tail events

**5. Learning-Based:**
- Robust RL (domain randomization)
- Learn policy over noise distribution
- Simulation-to-real transfer

State Estimation Under Uncertainty
-----------------------------------

**Measurement Models:**

**Full State (Ideal):**
    y = [x, ẋ, θ, θ̇]ᵀ + v

where v ~ N(0, R) is measurement noise.

**Partial (Realistic):**
    y = [x, θ]ᵀ + v

Measure position and angle, infer velocities.

**Extended Kalman Filter (EKF):**

Linearize nonlinear dynamics around current estimate:
- Prediction: X̂[k|k-1] from nonlinear f
- Jacobian: F = ∂f/∂X (linearization)
- Update: Standard Kalman with linearized H

**Challenges:**
- Linearization poor near horizontal (θ ≈ π/2)
- Unstable dynamics amplify estimation errors
- Can diverge if noise large

**Unscented Kalman Filter (UKF):**
- Sigma points capture nonlinearity
- No explicit Jacobians
- Better for large angles

**Particle Filter:**
- For swing-up with noise (highly nonlinear)
- Multimodal distributions
- Expensive but robust

Applications
------------

**1. Robotics:**

**Balance Control:**
- Segway, hoverboards (2-wheel balancing)
- Humanoid robots (inverted pendulum model of CoM)
- Ballbots (3D balancing)

**Disturbances:**
- Terrain variations
- External pushes
- Payload changes

**2. Reinforcement Learning:**

**Robust RL Benchmark:**
- OpenAI Gym with process noise
- Domain randomization (varying physics)
- Sim-to-real transfer

**Metrics:**
- Success rate (fraction of episodes without falling)
- Mean time to failure
- Cumulative reward under noise

**3. Control Education:**

**Laboratory Experiments:**
- Real inverted pendulum with actual noise
- Validates theory vs practice
- Demonstrates limitations of deterministic design

**4. Safety-Critical Systems:**

**Reliability Analysis:**
- Probability of falling vs time
- Mean time between failures
- Safety verification (formal methods)

Numerical Integration
---------------------

**Challenges:**

1. **Stiffness:**
   - Unstable equilibrium (one eigenvalue positive)
   - Fast stabilizing control
   - Multiple time scales

2. **Nonlinearity:**
   - Trigonometric terms
   - Coupling between states
   - Requires careful integration

3. **Constraints:**
   - Physical: |x| < x_max (track length)
   - Safety: |θ| < θ_max (falling criterion)
   - May need event detection

**Recommended Methods:**

1. **Euler-Maruyama:**
   - Simple, works for small dt
   - dt = 0.001-0.01 s typical
   - Check convergence

2. **Stochastic Runge-Kutta:**
   - Higher order
   - Better accuracy
   - More computation

3. **Implicit Methods:**
   - For very stiff cases
   - Stabilizes unstable dynamics
   - Rarely needed for cart-pole

**Event Detection:**

Detect falling: |θ| > θ_max
- Terminate simulation
- Record failure time
- Compute statistics

Monte Carlo Analysis
---------------------

**Reliability Assessment:**

Run N = 100-1,000 simulations:

1. **Success Rate:**
   Fraction maintaining |θ| < θ_max over time T

2. **Mean Time to Failure:**
   Average time until first |θ| > θ_max

3. **Histogram:**
   Distribution of failure times

4. **Worst-Case:**
   Longest excursion: max_k |θ[k]|

**Rare Event Simulation:**

For highly reliable controllers (MTF > hours):
- Importance sampling (bias toward large noise)
- Splitting methods
- Large deviations estimate

Comparison with Deterministic
------------------------------

**Deterministic Cart-Pole:**
- Perfect stabilization possible
- Infinite uptime with LQR
- Swing-up deterministic

**Stochastic Cart-Pole:**
- Only probabilistic stabilization
- Finite mean time to failure
- Swing-up success rate < 100%

**Critical Differences:**

1. **Stabilization:**
   - Det: θ → 0 exactly
   - Stoch: θ fluctuates, |θ| < ε with probability p

2. **Falling:**
   - Det: Only from large initial conditions
   - Stoch: Can fall from noise accumulation (rare but possible)

3. **Control Design:**
   - Det: Minimize tracking error
   - Stoch: Maximize reliability, minimize failure probability

Common Pitfalls
---------------

1. **Underestimating Noise Impact:**
   - Small noise can still cause falling (exponential instability)
   - Need Monte Carlo to assess reliability

2. **Using Only LQG:**
   - Works near upright only
   - Ignores constraints
   - Need robust methods for safety

3. **Ignoring Estimation:**
   - Perfect state knowledge unrealistic
   - Must include sensor noise
   - EKF can diverge on unstable system

4. **Wrong Noise Model:**
   - Additive vs multiplicative
   - Which states get noise (x vs ẋ)
   - Correlated vs independent

5. **Single Trajectory:**
   - One simulation not representative
   - Need ensemble for statistics
   - Rare events require many runs

6. **Deterministic Intuition:**
   - Noise changes optimal strategy
   - May need to back away from aggressive control
   - Robustness-performance tradeoff

"""

import numpy as np
import sympy as sp
from typing import Optional

from src.systems.base.core.continuous_stochastic_system import ContinuousStochasticSystem


class StochasticCartPole(ContinuousStochasticSystem):
    """
    Stochastic inverted pendulum on cart with random disturbances.

    Combines the challenging underactuated nonlinear dynamics of the cart-pole
    with process noise, creating a realistic benchmark for robust control,
    state estimation, and reliability analysis of balancing systems.

    Stochastic Differential Equations
    ----------------------------------
    State-space form with noise on velocities:

        dx = ẋ·dt
        dẋ = [(F + m·L·θ̇²·sin(θ) - m·L·θ̈·cos(θ))/(M+m)]·dt + σ_x·dW_x
        dθ = θ̇·dt
        dθ̇ = [(g·sin(θ) - cos(θ)·(F + m·L·θ̇²·sin(θ))/(M+m)) / 
              (L·(4/3 - m·cos²(θ)/(M+m)))]·dt + σ_θ·dW_θ

    where:
        - x: Cart position [m]
        - ẋ: Cart velocity [m/s]
        - θ: Pole angle from upward vertical [rad]
        - θ̇: Pole angular velocity [rad/s]
        - F: Applied horizontal force (control) [N]
        - σ_x: Cart disturbance intensity [m/(s²·√s)]
        - σ_θ: Pole disturbance intensity [rad/(s²·√s)]
        - W_x, W_θ: Independent Wiener processes

    **Deterministic Part:**
    Same coupled nonlinear dynamics as deterministic cart-pole.
    See CartPole or DiscreteCartPole for detailed derivation.

    **Stochastic Part:**
    Additive noise on velocity derivatives models:
    - Cart: Ground vibrations, wind on cart body
    - Pole: Wind on pole, measurement/actuation noise

    Physical Interpretation
    -----------------------
    **Cart Noise (σ_x):**

    Sources:
    - Ground irregularities (bumps, friction variations)
    - Wind gusts on cart
    - Motor noise (torque ripple)
    - Track vibrations

    Effect:
    - Horizontal disturbances
    - Couples to pole via inertial forces
    - Less critical than pole noise

    **Pole Noise (σ_θ):**

    Sources:
    - Wind on pole (large moment arm)
    - Joint friction variations
    - Measurement noise (affects observer)
    - Structural flexibility

    Effect:
    - Directly affects unstable mode
    - Most critical for stability
    - Primary cause of falling

    **Why Pole Noise Dominates:**

    Upright equilibrium is unstable in θ:
    - Eigenvalue: λ ≈ +√(g/L) > 0 (exponentially unstable)
    - Small θ perturbation grows exponentially
    - Noise on θ̇ integrates to θ → falling

    Cart position (x) is marginally stable:
    - Can drift but doesn't diverge
    - Less critical for balance

    Key Features
    ------------
    **Underactuated + Noise:**
    - 4 states, 1 control, 2 noise sources
    - Cannot independently control all modes
    - Noise affects modes directly

    **Unstable + Noise:**
    - Upright equilibrium exponentially unstable
    - Noise continuously perturbs
    - Requires fast, accurate control

    **Nonlinear + Noise:**
    - Coupling via sin(θ), cos(θ)
    - Small angle approximation poor under noise
    - Need nonlinear control/estimation

    **Bistable:**
    - Upright (unstable) vs downward (stable)
    - Noise can cause transitions
    - Swing-up under noise challenging

    State Space
    -----------
    State: X = [x, ẋ, θ, θ̇]
        - x ∈ ℝ: Cart position (track limits in practice)
        - ẋ ∈ ℝ: Cart velocity
        - θ ∈ ℝ: Pole angle (|θ| < π/4 for balance typically)
        - θ̇ ∈ ℝ: Pole angular velocity

    Control: u = F ∈ ℝ
        - Horizontal force on cart [N]
        - Bounded: |F| ≤ F_max typically

    Noise: w = [w_x, w_θ] ∈ ℝ²
        - Independent Wiener processes
        - Enter velocity equations

    Parameters
    ----------
    M : float, default=1.0
        Cart mass [kg]

    m : float, default=0.1
        Pole mass [kg]

    L : float, default=0.5
        Pole half-length [m]

    g : float, default=9.81
        Gravity [m/s²]

    b_cart : float, default=0.1
        Cart friction [N·s/m]

    b_pole : float, default=0.0
        Pole friction [N·m·s/rad]

    sigma_x : float, default=0.1
        Cart noise intensity [m/(s²·√s)]
        - Typical: 0.01-1.0
        - Less critical than σ_θ

    sigma_theta : float, default=0.05
        Pole noise intensity [rad/(s²·√s)]
        - Typical: 0.01-0.5
        - Most critical parameter
        - Determines mean time to failure

    Stochastic Properties
    ---------------------
    - System Type: NONLINEAR
    - Noise Type: ADDITIVE (on velocities)
    - SDE Type: Itô
    - Noise Dimension: nw = 2
    - Stationary: No (open-loop)
    - Bistable: Yes (upright vs downward)
    - Unstable: Yes (upright equilibrium)

    Applications
    ------------
    **1. Robust Control:**
    - LQG near upright (linearized)
    - Robust H∞ control
    - Stochastic MPC with chance constraints
    - Risk-sensitive control

    **2. State Estimation:**
    - EKF for balance (linearized)
    - UKF for swing-up (nonlinear)
    - Particle filter for multimodal

    **3. Reinforcement Learning:**
    - Robust RL with noise
    - Domain randomization
    - Sim-to-real transfer

    **4. Reliability Analysis:**
    - Mean time to failure
    - Probability of falling
    - Safety verification

    **5. Sensor Fusion:**
    - IMU + encoders
    - Kalman filter for fusion
    - Complementary filtering

    Numerical Integration
    ---------------------
    **Recommended:**
    - Euler-Maruyama: dt = 0.001-0.01 s
    - Check falling: |θ| > threshold
    - Monitor constraints

    **Event Detection:**
    Terminate when falling:
    - |θ| > π/4 (45°)
    - |x| > x_max (track end)

    Monte Carlo Guidelines
    -----------------------
    **Reliability Assessment:**
    - N = 100-1,000 runs
    - Record failure times
    - Compute success rate, MTF

    **Statistics:**
    - Mean trajectory ± 2σ bands
    - Histogram of max |θ|
    - Distribution of failure times

    Comparison with Deterministic
    ------------------------------
    **Deterministic:**
    - Perfect upright stabilization
    - Deterministic swing-up
    - Infinite uptime with LQR

    **Stochastic:**
    - Probabilistic stabilization
    - Swing-up success rate
    - Finite mean time to failure

    Limitations
    -----------
    - Additive noise only (not multiplicative)
    - Independent noise sources
    - No actuator noise (only process)
    - Rigid body (no flexibility)

    Extensions
    ----------
    - Multiplicative noise: σ(X)
    - Actuator noise: u → u + noise
    - Measurement delays
    - Flexible pole
    - 3D cart-pole
    
    See Also
    --------
    CartPole : Deterministic version
    DiscreteCartPole : Discrete-time deterministic
    StochasticPendulum : Simpler (no cart)
    """

    def define_system(
        self,
        M: float = 1.0,
        m: float = 0.1,
        L: float = 0.5,
        g: float = 9.81,
        b_cart: float = 0.1,
        b_pole: float = 0.0,
        sigma_x: float = 0.1,
        sigma_theta: float = 0.05,
    ):
        """
        Define stochastic cart-pole dynamics.

        Parameters
        ----------
        M : float, default=1.0
            Cart mass [kg]

        m : float, default=0.1
            Pole mass [kg]

        L : float, default=0.5
            Pole half-length [m]

        g : float, default=9.81
            Gravity [m/s²]

        b_cart : float, default=0.1
            Cart friction [N·s/m]

        b_pole : float, default=0.0
            Pole friction [N·m·s/rad]

        sigma_x : float, default=0.1
            Cart disturbance intensity [m/(s²·√s)]
            - Horizontal disturbances
            - Typical: 0.01-1.0
            - Less critical than σ_θ

        sigma_theta : float, default=0.05
            Pole disturbance intensity [rad/(s²·√s)]
            - Angular disturbances
            - Typical: 0.01-0.5
            - MOST CRITICAL for stability
            - Determines mean time to failure

        Notes
        -----
        **Noise Intensity Guidelines:**

        Cart noise (σ_x):
        - Small (0.01): Lab conditions, smooth track
        - Medium (0.1): Typical industrial
        - Large (1.0): Rough terrain, high wind

        Pole noise (σ_θ):
        - Small (0.01): Precise control, low wind
        - Medium (0.05): Typical outdoor
        - Large (0.2): High wind, poor sensors

        **Critical: σ_θ Determines Reliability**

        Exponential dependence:
            MTF ~ exp(θ_max²/σ_θ²)

        Example (θ_max = 0.2 rad ≈ 11.5°):
        - σ_θ = 0.01: MTF ~ exp(400) ~ infinite
        - σ_θ = 0.05: MTF ~ exp(16) ~ 9 million seconds ~ 3 months
        - σ_θ = 0.1: MTF ~ exp(4) ~ 55 seconds
        - σ_θ = 0.2: MTF ~ exp(1) ~ 3 seconds

        Small increase in noise → huge decrease in reliability!

        **Noise Correlation:**

        Currently: Independent noise on x and θ.

        In reality: May be correlated (common wind gust affects both).
        Extension: Use 2×2 covariance matrix.

        **Physical Validation:**

        Check noise levels are reasonable:
        1. Simulate deterministic + stochastic
        2. Compare trajectories (should be similar but not identical)
        3. Success rate should be 50-95% (not 0% or 100%)

        **Design Tradeoff:**

        Smaller noise:
        - More reliable (longer MTF)
        - Easier control problem
        - Less realistic

        Larger noise:
        - Less reliable (shorter MTF)
        - Harder control (robustness needed)
        - More realistic

        Choose based on application requirements.
        """
        # Store parameters
        self.M = M
        self.m = m
        self.L = L
        self.g = g
        self.b_cart = b_cart
        self.b_pole = b_pole

        # State variables
        x, x_dot, theta, theta_dot = sp.symbols('x x_dot theta theta_dot', real=True)
        F = sp.symbols('F', real=True)

        # Parameters
        M_sym, m_sym, L_sym, g_sym = sp.symbols('M m L g', positive=True)
        b_cart_sym, b_pole_sym = sp.symbols('b_cart b_pole', nonnegative=True)
        sigma_x_sym = sp.symbols('sigma_x', real=True, nonnegative=True)
        sigma_theta_sym = sp.symbols('sigma_theta', real=True, nonnegative=True)

        self.state_vars = [x, x_dot, theta, theta_dot]
        self.control_vars = [F]
        self.order = 1

        self.parameters = {
            M_sym: M,
            m_sym: m,
            L_sym: L,
            g_sym: g,
            b_cart_sym: b_cart,
            b_pole_sym: b_pole,
            sigma_x_sym: sigma_x,
            sigma_theta_sym: sigma_theta,
        }

        # Nonlinear dynamics (deterministic part)
        sin_theta = sp.sin(theta)
        cos_theta = sp.cos(theta)
        denom = M_sym + m_sym - m_sym * cos_theta**2

        # Angular acceleration
        theta_ddot = (
            (M_sym + m_sym) * g_sym * sin_theta
            - cos_theta * (F - b_cart_sym * x_dot + m_sym * L_sym * theta_dot**2 * sin_theta)
            - b_pole_sym * theta_dot / L_sym
        ) / (L_sym * denom)

        # Linear acceleration
        x_ddot = (
            F - b_cart_sym * x_dot
            + m_sym * L_sym * (theta_dot**2 * sin_theta - theta_ddot * cos_theta)
        ) / (M_sym + m_sym)

        # DRIFT (deterministic dynamics)
        self._f_sym = sp.Matrix([
            x_dot,
            x_ddot,
            theta_dot,
            theta_ddot
        ])

        # DIFFUSION (stochastic part)
        # Noise on cart velocity (x_dot equation) and pole velocity (theta_dot equation)
        self.diffusion_expr = sp.Matrix([
            [0, 0],              # No noise on dx/dt = x_dot
            [sigma_x_sym, 0],    # Noise on cart acceleration
            [0, 0],              # No noise on dθ/dt = theta_dot
            [0, sigma_theta_sym] # Noise on pole acceleration
        ])

        # Itô SDE
        self.sde_type = "ito"

        # Output: Typically measure x and θ (not velocities)
        self._h_sym = sp.Matrix([x, theta])

    def setup_equilibria(self):
        """
        Set up equilibrium points (deterministic part).

        Note: With noise, cannot maintain exact equilibrium.
        These are nominal targets that fluctuate.
        """
        # Upright (unstable - main control target)
        self.add_equilibrium(
            "upright",
            x_eq=np.array([0.0, 0.0, 0.0, 0.0]),
            u_eq=np.array([0.0]),
            verify=True,
            stability="unstable",
            notes="Unstable upright equilibrium. With noise, state fluctuates around this. "
                  "Falling occurs when noise pushes |θ| beyond threshold. "
                  "Mean time to failure exponentially depends on σ_θ."
        )

        # Downward (stable)
        self.add_equilibrium(
            "downward",
            x_eq=np.array([0.0, 0.0, np.pi, 0.0]),
            u_eq=np.array([0.0]),
            verify=True,
            stability="stable",
            notes="Stable downward equilibrium. Noise causes oscillations but no falling."
        )

        self.set_default_equilibrium("upright")

    def compute_energy(self, x: np.ndarray) -> float:
        """
        Compute total mechanical energy.

        Parameters
        ----------
        x : np.ndarray
            State [x, ẋ, θ, θ̇]

        Returns
        -------
        float
            Total energy [J]

        Notes
        -----
        Energy fluctuates due to noise (not conserved).

        Examples
        --------
        >>> cartpole = StochasticCartPole()
        >>> x = np.array([0.0, 0.5, 0.1, 0.2])
        >>> E = cartpole.compute_energy(x)
        >>> print(f"Energy: {E:.3f} J")
        """
        _, x_dot, theta, theta_dot = x
        
        # Cart kinetic energy
        KE_cart = 0.5 * self.M * x_dot**2
        
        # Pole kinetic energy (translational + rotational)
        KE_pole = 0.5 * self.m * (
            x_dot**2 + (self.L * theta_dot)**2 + 
            2 * x_dot * self.L * theta_dot * np.cos(theta)
        )
        
        # Pole potential energy
        PE = self.m * self.g * self.L * (np.cos(theta) + 1)
        
        return KE_cart + KE_pole + PE

    def get_noise_intensities(self) -> dict:
        """
        Get noise intensity parameters.

        Returns
        -------
        dict
            {'sigma_x': ..., 'sigma_theta': ...}

        Examples
        --------
        >>> cartpole = StochasticCartPole(sigma_x=0.1, sigma_theta=0.05)
        >>> noise = cartpole.get_noise_intensities()
        >>> print(f"Pole noise (critical): {noise['sigma_theta']}")
        """
        return {
            'sigma_x': self.parameters[sp.symbols('sigma_x')],
            'sigma_theta': self.parameters[sp.symbols('sigma_theta')],
        }

    def estimate_mean_time_to_failure(
        self,
        theta_max: float = 0.2,
        control_gain: Optional[float] = None,
    ) -> float:
        """
        Estimate mean time to failure (very rough approximation).

        Uses Kramers-like formula:
            MTF ~ exp(θ_max²·κ/σ_θ²)

        Parameters
        ----------
        theta_max : float, default=0.2
            Failure threshold [rad]
        control_gain : Optional[float]
            Effective stabilization strength (if None, estimate from g/L)

        Returns
        -------
        float
            Estimated MTF [s]

        Notes
        -----
        This is very approximate! For accurate MTF, use Monte Carlo.

        Examples
        --------
        >>> cartpole = StochasticCartPole(sigma_theta=0.05)
        >>> mtf = cartpole.estimate_mean_time_to_failure(theta_max=0.2)
        >>> print(f"Estimated MTF: {mtf:.2e} seconds")
        """
        sigma_theta = self.parameters[sp.symbols('sigma_theta')]
        
        if control_gain is None:
            # Rough estimate from natural frequency
            control_gain = np.sqrt(self.g / self.L)
        
        # Kramers-like approximation
        barrier = theta_max**2 * control_gain
        mtf = np.exp(barrier / sigma_theta**2)
        
        return mtf


# Convenience functions
def create_laboratory_cartpole(
    noise_level: str = 'low',
) -> StochasticCartPole:
    """
    Create cart-pole for laboratory experiments.

    Parameters
    ----------
    noise_level : str, default='low'
        'low', 'medium', or 'high'

    Returns
    -------
    StochasticCartPole

    Examples
    --------
    >>> # Clean lab environment
    >>> lab_low = create_laboratory_cartpole('low')
    >>> 
    >>> # Typical lab with some vibration
    >>> lab_med = create_laboratory_cartpole('medium')
    """
    noise_presets = {
        'low': {'sigma_x': 0.01, 'sigma_theta': 0.01},
        'medium': {'sigma_x': 0.05, 'sigma_theta': 0.03},
        'high': {'sigma_x': 0.2, 'sigma_theta': 0.1},
    }
    
    preset = noise_presets.get(noise_level, noise_presets['medium'])
    
    return StochasticCartPole(
        M=1.0,
        m=0.1,
        L=0.5,
        g=9.81,
        **preset
    )


def create_segway_model(
    rider_mass: float = 70.0,
    noise_level: str = 'medium',
) -> StochasticCartPole:
    """
    Create Segway-like model with rider.

    Parameters
    ----------
    rider_mass : float, default=70.0
        Rider mass [kg]
    noise_level : str, default='medium'
        Environmental noise level

    Returns
    -------
    StochasticCartPole

    Examples
    --------
    >>> # Outdoor Segway
    >>> segway = create_segway_model(rider_mass=70, noise_level='medium')
    """
    noise_presets = {
        'low': {'sigma_x': 0.05, 'sigma_theta': 0.02},
        'medium': {'sigma_x': 0.1, 'sigma_theta': 0.05},
        'high': {'sigma_x': 0.3, 'sigma_theta': 0.1},
    }
    
    preset = noise_presets.get(noise_level, noise_presets['medium'])
    
    # Segway-like parameters
    return StochasticCartPole(
        M=10.0,          # Platform mass
        m=rider_mass,    # Rider mass
        L=0.5,           # CoM height
        g=9.81,
        b_cart=1.0,      # Rolling resistance
        b_pole=0.1,      # Air resistance
        **preset
    )