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
Discrete Stochastic Queue - Random Arrivals and Service Dynamics
=================================================================

This module provides discrete-time stochastic queue models, fundamental for
analyzing systems with random arrivals and service. Stochastic queues serve as:

- The canonical model for congestion in networks, call centers, and traffic
- A framework for capacity planning under uncertainty (servers, bandwidth, inventory)
- A benchmark for understanding stability in stochastic systems (traffic intensity)
- An illustration of how randomness creates waiting even when capacity exceeds demand
- A foundation for operations research, telecommunications, and service systems

Discrete-time queues model systems sampled at regular intervals (time slots):
- Arrivals per slot: Random (Poisson, binomial)
- Service per slot: Random (state-dependent)
- Queue evolves: Q[k+1] = max(0, Q[k] + A[k] - S[k])

The discrete formulation is natural for:
- Packet-switched networks (time slots)
- Scheduled systems (appointments, time windows)
- Sampled-data control (digital observation)
- Simulation efficiency (event-based can be slow)

Physical Context
----------------

**Queueing Systems Everywhere:**

1. **Telecommunications:**
   - Packet queues in routers/switches
   - Call center phone queues
   - Network congestion (Internet)
   - 5G resource allocation

2. **Computer Systems:**
   - Process scheduling (CPU queue)
   - I/O buffers
   - Database transaction queues
   - Cloud service requests

3. **Transportation:**
   - Traffic signals (vehicle queues)
   - Airport check-in lines
   - Ride-sharing pickup queues
   - Logistics (warehouse, loading docks)

4. **Healthcare:**
   - Emergency room waiting
   - Operating room scheduling
   - Patient appointment systems
   - ICU bed availability

5. **Manufacturing:**
   - Work-in-process inventory
   - Machine utilization queues
   - Assembly line buffers
   - Supply chain nodes

6. **Service Industry:**
   - Retail checkout lines
   - Restaurant reservations
   - Bank teller queues
   - Customer support tickets

**Why Randomness Matters:**

Even with average capacity > average demand, queues form due to variability:
- Random arrivals: Sometimes bursts, sometimes quiet
- Random service: Sometimes fast, sometimes slow
- Mismatch creates waiting (can't store excess capacity)

**Key Insight:**
Deterministic system (constant arrivals/service): No queue if capacity ≥ demand
Stochastic system: Queue forms even if E[service] > E[arrivals]!

Mathematical Background
-----------------------

**Classic Queueing Theory (Kendall Notation):**

M/M/1: Markovian arrivals, Markovian service, 1 server
- Continuous-time Markov chain
- Poisson arrivals, exponential service
- Analytical solutions (steady-state)

**Discrete-Time Extension:**

Geo/Geo/1: Geometric arrivals, geometric service, 1 server
- Discrete-time Markov chain
- Arrivals/service per slot: Geometric distribution
- Tractable but limiting

**General Discrete Queue (This Implementation):**

Queue length dynamics:
    Q[k+1] = max(0, Q[k] + A[k] - S[k])

where:
- Q[k]: Queue length at slot k (number in system)
- A[k]: Arrivals during slot k (random)
- S[k]: Service completions during slot k (random, state-dependent)
- max(0, ...): Cannot have negative queue

**State-Dependent Service:**

Service rate depends on queue state:
    S[k] ~ f(Q[k], μ)

Examples:
- Constant: S = μ (if Q > 0)
- Linear: S = min(Q, μ) (can't serve more than present)
- Saturating: S approaches μ_max as Q → ∞

Mathematical Formulation
------------------------

**Discrete-Time Queue Dynamics:**

Deterministic update:
    Q_det[k+1] = max(0, Q[k] + λ - μ(Q[k]))

With stochastic arrivals and service:
    Q[k+1] = max(0, Q[k] + A[k] - S[k])

**Linearized (for small fluctuations around Q̄):**
    Q[k+1] ≈ Q[k] + (λ - μ) + w_A[k] - w_S[k]

where w_A, w_S represent fluctuations.

**Traffic Intensity:**
    ρ = λ/μ

Critical parameter:
- ρ < 1: Stable (queue finite on average)
- ρ = 1: Critical (queue grows slowly)
- ρ > 1: Unstable (queue grows unboundedly)

**Stability Condition:**
Average arrivals < average service capacity:
    E[A] < E[S]

Otherwise queue grows to infinity (unstable).

Key Properties
--------------

**1. Non-Negativity:**
Q[k] ≥ 0 always (cannot have negative queue).

Max operator enforces this.

**2. Stability (ρ < 1):**
If traffic intensity ρ < 1, queue has stationary distribution.

Mean queue length (steady-state):
    E[Q] ≈ ρ/(1-ρ) (for simple M/M/1 analog)

**3. Utilization:**
Server busy probability:
    U = P(Q > 0) ≈ ρ (for stable queue)

**4. Waiting Time:**
Little's Law:
    E[W] = E[Q]/λ

Average wait = average queue / arrival rate.

**5. Variance:**
Queue length variance increases near ρ = 1 (critical slowing).

**6. Heavy Traffic:**
As ρ → 1⁻:
- E[Q] → ∞ (queue explodes)
- Fluctuations increase
- System fragile

Stochastic Effects
------------------

**Why Randomness Creates Queues:**

**Deterministic:**
- Arrivals: Exactly λ per slot
- Service: Exactly μ per slot
- Queue: Q = 0 if μ ≥ λ (no waiting!)

**Stochastic:**
- Arrivals: Random around λ (sometimes 0, sometimes 5)
- Service: Random around μ
- Queue: Forms even if μ > λ

**Example:**
- λ = 4 customers/hour (average)
- μ = 5 customers/hour (capacity)
- ρ = 0.8 (80% utilization, safe)

But:
- Some hours: 8 arrivals (burst) → queue grows
- Some hours: 1 arrival (quiet) → queue drains
- Net: Queue oscillates around E[Q] ≈ 4

**Variability Impact:**

Higher variance in A or S:
- Larger queues (even with same means)
- More waiting
- Lower throughput

**Squared Coefficient of Variation (SCV):**
    C² = Var[A]/E[A]²

Higher C² → worse performance for same ρ.

Applications
------------

**1. Network Engineering:**

**Packet Queues:**
- Router buffers (packets waiting transmission)
- Arrival: Poisson (network traffic)
- Service: Link bandwidth
- QoS: Control delay via queue management

**Bandwidth Allocation:**
- Optimize: Minimize delay subject to cost
- Admission control: Reject when queue too long

**2. Call Centers:**

**Staffing Optimization:**
- Arrivals: Call volume (time-varying)
- Service: Agent handling time
- Optimize: # agents vs wait time

**Service Level:**
Target: 80% of calls answered within 20 seconds.
Use queue model to determine staffing.

**3. Manufacturing:**

**Work-in-Process (WIP):**
- Arrivals: Jobs entering station
- Service: Machine processing
- Goal: Minimize WIP (inventory cost)

**Bottleneck Analysis:**
Identify stations with ρ near 1 (congestion points).

**4. Healthcare:**

**Emergency Department:**
- Arrivals: Patient arrivals (non-stationary, peaks)
- Service: Physician time
- Optimize: Reduce wait, prevent boarding

**Operating Room Scheduling:**
- Arrivals: Scheduled surgeries
- Service: Surgery duration (random)
- Buffer: Time between cases

**5. Cloud Computing:**

**Request Queues:**
- Arrivals: User requests (bursty)
- Service: Server processing
- Auto-scaling: Add servers when Q[k] > threshold

**6. Transportation:**

**Traffic Signals:**
- Arrivals: Vehicles per cycle
- Service: Green light duration
- Optimize: Signal timing

Discrete-Time Modeling
-----------------------

**Why Discrete Time?**

1. **Natural Time Slots:**
   - Network packets (fixed-length slots)
   - Scheduled appointments (15-min blocks)
   - Sampling periods (data every Δt)

2. **Computational Efficiency:**
   - Fixed time steps (no event scheduling)
   - Vectorizable (batch processing)
   - Predictable computation time

3. **Control Design:**
   - Discrete-time MPC
   - Periodic decisions (every slot)
   - Digital implementation

**Continuous vs Discrete:**

Continuous (M/M/1):
- Event-driven (arrivals and departures trigger state changes)
- Poisson arrivals, exponential service
- Analytical solutions exist

Discrete:
- Time-driven (state updates every Δt)
- Arrivals/service per slot
- Simulation-friendly

**Conversion:**

For small Δt:
- Discrete arrival rate ≈ λ·Δt
- Discrete service rate ≈ μ·Δt

Common Pitfalls
---------------

1. **Ignoring Non-Negativity:**
   - Forgetting max(0, ...) operation
   - Can get negative queues (unphysical)

2. **Wrong Traffic Intensity:**
   - Using ρ = λ/μ_max instead of E[S]
   - Overestimating capacity

3. **Deterministic Thinking:**
   - "Capacity > demand, so no queue"
   - Variability creates queues!

4. **Steady-State Assumption:**
   - Real systems non-stationary (time-varying λ)
   - Morning rush ≠ afternoon lull
   - Need time-dependent models

5. **Independence Assumption:**
   - Arrivals often correlated (bursty)
   - Service times may depend on queue length
   - Model may be too simple

6. **Single Server:**
   - Real systems: Multiple servers
   - Need M/M/c extension
   
**Impact:**
Queueing theory enabled:
- Telephone network design (Erlang)
- Internet architecture (packet switching)
- Service industry optimization
- Understanding randomness creates waiting
"""

import numpy as np
import sympy as sp
from typing import Optional

from src.systems.base.core.discrete_stochastic_system import DiscreteStochasticSystem


class DiscreteStochasticQueue(DiscreteStochasticSystem):
    """
    Discrete-time stochastic queue with random arrivals and service.

    Models a queueing system sampled at discrete time intervals, with
    random arrivals and state-dependent service. Fundamental for analyzing
    congestion, capacity planning, and performance under uncertainty.

    Queue Dynamics
    --------------
    Basic queue evolution:
        Q[k+1] = max(0, Q[k] + A[k] - S[k])

    where:
        - Q[k]: Queue length (number in system) at time k
        - A[k]: Arrivals during slot k (random, ≥ 0)
        - S[k]: Service completions during slot k (random, ≤ Q[k])
        - max(0, ...): Queue cannot be negative

    **Linearized (around equilibrium Q̄):**
        Q[k+1] ≈ Q[k] + (λ - μ) + w_A[k] - w_S[k]

    where w_A, w_S represent arrival and service noise.

    **State-Space (Simplified Diffusion Approximation):**
        Q[k+1] = Q[k] + (λ - μ)·Δt + σ_net·w[k]

    where σ_net combines arrival and service variability.

    Physical Interpretation
    -----------------------
    **Queue Length Q[k]:**
    - Number of customers/packets/jobs in system
    - Q = 0: Empty (idle server)
    - Large Q: Congestion (long waits)

    **Arrivals A[k]:**
    - New customers entering system
    - Distribution: Poisson (λ parameter), binomial, general
    - Typical: 0-10 per slot (depends on λ·Δt)

    **Service S[k]:**
    - Completions (customers leaving)
    - Depends on Q: Can't serve more than present
    - Capacity: μ (maximum service rate)

    **Traffic Intensity:**
        ρ = λ/μ

    Critical parameter:
    - ρ < 1: Stable (queue bounded on average)
    - ρ = 1: Critical (marginal)
    - ρ > 1: Unstable (queue grows to infinity)

    Key Features
    ------------
    **Non-Negativity:**
    Q ≥ 0 enforced via max(0, ...) operation.

    **Stability Threshold:**
    ρ = λ/μ determines long-run behavior.

    **State-Dependent Service:**
    When Q = 0, no service possible (idle server).

    **Stochastic Fluctuations:**
    Even stable queue (ρ < 1) exhibits large fluctuations.

    **Heavy Traffic:**
    Near ρ = 1, queue becomes very sensitive to perturbations.

    Mathematical Properties
    -----------------------
    **Stability:**
    For ρ < 1, queue has stationary distribution.

    **Mean Queue Length (Steady-State):**
    Approximate (diffusion limit):
        E[Q] ≈ ρ/(1-ρ) (for M/M/1 analog)

    **Waiting Time:**
    Little's Law (exact):
        E[W] = E[Q]/λ

    Average waiting time = average queue / arrival rate.

    **Variance:**
    Queue length variance high near ρ = 1.

    **Distribution:**
    For M/M/1 continuous analog:
        P(Q = n) = (1-ρ)·ρⁿ (geometric)

    Discrete-time: Similar but modified.

    Physical Interpretation
    -----------------------
    **Arrival Rate λ:**
    - Units: [customers/time]
    - Average arrivals per unit time
    - Typical: 0.1-10 depending on system

    **Service Rate μ:**
    - Units: [customers/time]
    - Average service capacity
    - Must be μ > λ for stability

    **Traffic Intensity ρ = λ/μ:**
    - Dimensionless (utilization)
    - ρ = 0.5: 50% utilized (stable, low wait)
    - ρ = 0.8: 80% utilized (moderate wait)
    - ρ = 0.95: 95% utilized (high wait, fragile)
    - ρ > 1: Overloaded (unstable)

    **Design Rule:**
    Keep ρ < 0.8 for reliable operation.
    - Buffer for variability
    - Acceptable waiting times
    - Robustness to perturbations

    State Space
    -----------
    State: Q ∈ ℤ₊ = {0, 1, 2, ...}
        - Non-negative integer (count of items)
        - Unbounded (unless capacity limit)

    Control: u (optional)
        - Admission control (reject arrivals)
        - Service rate adjustment
        - Server allocation

    Noise: w_A, w_S
        - Arrival noise (Poisson fluctuations)
        - Service noise (randomness in completion)

    Parameters
    ----------
    lambda_rate : float, default=0.8
        Mean arrival rate [customers/slot]
        - Must be positive
        - Typical: 0.1-10

    mu_rate : float, default=1.0
        Mean service rate [customers/slot]
        - Must be positive
        - Must have μ > λ for stability

    sigma_arrival : float, default=0.3
        Arrival noise std dev [customers/slot]
        - From Poisson: σ_A ≈ √λ
        - Can be different (overdispersion)

    sigma_service : float, default=0.3
        Service noise std dev [customers/slot]
        - Variability in service times
        - Typical: 0.1-1.0

    dt : float, default=1.0
        Time slot duration [s]

    Stochastic Properties
    ---------------------
    - System Type: NONLINEAR (max operator)
    - Noise Type: ADDITIVE (approximate)
    - State: Discrete (integer-valued ideally)
    - Stationary: If ρ < 1
    - Heavy Tails: Geometric distribution (M/M/1 analog)

    Applications
    ------------
    **1. Network Engineering:**
    - Router queue sizing
    - Bandwidth provisioning
    - QoS guarantees

    **2. Call Centers:**
    - Staffing optimization
    - Service level targets
    - Abandonment modeling

    **3. Manufacturing:**
    - WIP inventory control
    - Buffer sizing
    - Throughput analysis

    **4. Healthcare:**
    - ER capacity planning
    - Surgery scheduling
    - Bed management

    **5. Cloud Computing:**
    - Auto-scaling triggers
    - Load balancing
    - SLA compliance

    Numerical Simulation
    --------------------
    **Direct Simulation:**

    ```python
    Q = np.zeros(N+1)
    Q[0] = Q0
    
    for k in range(N):
        A_k = np.random.poisson(lambda_rate)
        S_k = np.random.poisson(min(Q[k], mu_rate))
        Q[k+1] = max(0, Q[k] + A_k - S_k)
    ```

    **Issues:**
    - Integer-valued (exact Poisson)
    - Max operator (non-smooth)
    - State-dependent service

    **Diffusion Approximation:**
    For large λ, μ, treat as continuous:
        Q[k+1] ≈ Q[k] + (λ-μ)·Δt + σ_net·w[k]

    Smoother, easier analysis.

    Performance Metrics
    -------------------
    **Mean Queue Length:**
        E[Q] (average number waiting)

    **Mean Waiting Time:**
        E[W] = E[Q]/λ (Little's Law)

    **Server Utilization:**
        ρ = λ/μ (fraction time busy)

    **Probability of Delay:**
        P(Q > 0) (chance of waiting)

    **Percentiles:**
        P(Q ≤ q_95) = 0.95 (95th percentile)

    Comparison with M/M/1
    ---------------------
    **M/M/1 (Continuous-Time):**
    - Poisson arrivals (rate λ)
    - Exponential service (rate μ)
    - Analytical formulas

    **Discrete Queue:**
    - Arrivals per slot (mean λ·Δt)
    - Service per slot (mean μ·Δt)
    - Numerical/simulation

    **Convergence:**
    As Δt → 0, discrete → continuous M/M/1.

    Limitations
    -----------
    - Single server (extend to M/M/c)
    - Infinite capacity (no buffer limit)
    - FIFO discipline (no priorities)
    - Stationary (time-invariant λ, μ)
    - Independence (no batch arrivals)

    Extensions
    ----------
    - M/M/c: Multiple servers
    - Finite capacity: M/M/1/K
    - Priority queues
    - Time-varying rates: λ(t), μ(t)
    - Network of queues (Jackson networks)

    See Also
    --------
    DiscreteAR1 : Simpler linear stochastic system
    DiscreteRandomWalk : No bounds (queue has Q ≥ 0)
    """

    def define_system(
        self,
        lambda_rate: float = 0.8,
        mu_rate: float = 1.0,
        sigma_arrival: float = 0.3,
        sigma_service: float = 0.3,
        dt: float = 1.0,
    ):
        """
        Define discrete stochastic queue dynamics.

        Parameters
        ----------
        lambda_rate : float, default=0.8
            Mean arrival rate [customers/slot]
            - Must be positive
            - Typical: 0.1-10

        mu_rate : float, default=1.0
            Mean service rate [customers/slot]
            - Must be positive
            - Must have μ > λ for stability

        sigma_arrival : float, default=0.3
            Arrival noise std dev
            - For Poisson: σ_A = √λ
            - Can differ (overdispersion)

        sigma_service : float, default=0.3
            Service noise std dev

        dt : float, default=1.0
            Time slot duration [s]

        Raises
        ------
        ValueError
            If λ ≤ 0, μ ≤ 0, or ρ ≥ 1

        UserWarning
            If ρ > 0.9 (near critical, fragile)

        Notes
        -----
        **Traffic Intensity:**
            ρ = λ/μ

        **Stability:**
        - ρ < 1: Stable (queue finite on average)
        - ρ ≥ 1: Unstable (queue → ∞)

        **Mean Queue (Approximate):**
            E[Q] ≈ ρ/(1-ρ)

        For ρ = 0.8: E[Q] ≈ 4
        For ρ = 0.9: E[Q] ≈ 9
        For ρ = 0.95: E[Q] ≈ 19

        **Design Guideline:**
        Keep ρ < 0.8 for good performance:
        - Reasonable wait times
        - Robustness to variability
        - Avoid heavy traffic regime

        **Noise Structure:**

        Arrival noise: From Poisson statistics
            σ_A ≈ √λ (for Poisson)

        Service noise: From variability
            σ_S depends on distribution

        Net noise:
            σ_net = √(σ_A² + σ_S²)

        **Simplified Model:**

        This uses diffusion approximation:
            Q[k+1] = max(0, Q[k] + (λ-μ) + w[k])

        where w[k] ~ N(0, σ_net²).

        More accurate: Exact Poisson simulation (Gillespie-like).
        """
        if lambda_rate <= 0:
            raise ValueError(f"lambda_rate must be positive, got {lambda_rate}")
        if mu_rate <= 0:
            raise ValueError(f"mu_rate must be positive, got {mu_rate}")

        # Traffic intensity
        rho = lambda_rate / mu_rate

        if rho >= 1.0:
            raise ValueError(
                f"Traffic intensity ρ = {rho:.3f} ≥ 1. "
                f"Queue is unstable (arrivals ≥ service). "
                f"Require λ < μ for stability."
            )

        if rho > 0.9:
            import warnings
            warnings.warn(
                f"Traffic intensity ρ = {rho:.3f} > 0.9 (heavy traffic). "
                f"Queue will be large and sensitive to perturbations. "
                f"Consider increasing service rate or reducing arrival rate.",
                UserWarning
            )

        # Store parameters
        self.lambda_rate = lambda_rate
        self.mu_rate = mu_rate
        self.rho = rho

        # State variable: Queue length
        Q = sp.symbols("Q", real=True, nonnegative=True)

        # Parameters
        lambda_sym = sp.symbols("lambda", positive=True)
        mu_sym = sp.symbols("mu", positive=True)
        sigma_A_sym = sp.symbols("sigma_A", positive=True)
        sigma_S_sym = sp.symbols("sigma_S", positive=True)

        self.state_vars = [Q]
        self.control_vars = []  # No control in basic model

        # DETERMINISTIC PART (net flow)
        # Q[k+1] = Q[k] + λ - μ (ignoring max and stochasticity)
        # Simplified: Linear approximation
        net_flow = lambda_sym - mu_sym
        Q_next = Q + net_flow

        self._f_sym = sp.Matrix([Q_next])

        # Net noise (combined arrival and service variability)
        sigma_net = sp.sqrt(sigma_A_sym**2 + sigma_S_sym**2)

        self.parameters = {
            lambda_sym: lambda_rate,
            mu_sym: mu_rate,
            sigma_A_sym: sigma_arrival,
            sigma_S_sym: sigma_service,
        }
        self.order = 1
        self._dt = dt

        # STOCHASTIC PART
        # Net noise from arrivals (+) and service (-)
        self.diffusion_expr = sp.Matrix([sigma_net])

        self.sde_type = "ito"

        # Output: Queue length
        self._h_sym = sp.Matrix([Q])

    def get_mean_queue_length(self) -> float:
        """
        Get theoretical mean queue length E[Q] ≈ ρ/(1-ρ).

        Returns
        -------
        float
            Mean queue length (steady-state)

        Notes
        -----
        Approximate formula (diffusion/M/M/1 analog).

        Examples
        --------
        >>> queue = DiscreteStochasticQueue(lambda_rate=0.8, mu_rate=1.0)
        >>> E_Q = queue.get_mean_queue_length()
        >>> print(f"Mean queue: {E_Q:.2f}")
        """
        return self.rho / (1 - self.rho)

    def get_mean_waiting_time(self) -> float:
        """
        Get mean waiting time via Little's Law: E[W] = E[Q]/λ.

        Returns
        -------
        float
            Mean waiting time [slots]

        Examples
        --------
        >>> queue = DiscreteStochasticQueue(lambda_rate=0.8, mu_rate=1.0)
        >>> E_W = queue.get_mean_waiting_time()
        >>> print(f"Mean wait: {E_W:.2f} slots")
        """
        E_Q = self.get_mean_queue_length()
        return E_Q / self.lambda_rate

    def get_utilization(self) -> float:
        """
        Get server utilization (traffic intensity) ρ = λ/μ.

        Returns
        -------
        float
            Utilization (0 to 1)

        Examples
        --------
        >>> queue = DiscreteStochasticQueue(lambda_rate=0.8, mu_rate=1.0)
        >>> util = queue.get_utilization()
        >>> print(f"Utilization: {util:.1%}")
        """
        return self.rho


# Convenience functions
def create_call_center_queue(
    calls_per_hour: float = 48,
    agents: int = 1,
    avg_handle_time_minutes: float = 1.0,
) -> DiscreteStochasticQueue:
    """
    Create queue model for call center.

    Parameters
    ----------
    calls_per_hour : float
        Average call arrival rate [calls/hour]
    agents : int
        Number of agents (servers)
    avg_handle_time_minutes : float
        Average call duration [minutes]

    Returns
    -------
    DiscreteStochasticQueue

    Notes
    -----
    Converts to per-minute rates for discrete-time model.

    Examples
    --------
    >>> # 1 agent, 48 calls/hour, 1 min average
    >>> center = create_call_center_queue(
    ...     calls_per_hour=48,
    ...     agents=1,
    ...     avg_handle_time_minutes=1.0
    ... )
    >>> print(f"ρ = {center.get_utilization():.2f}")
    """
    # Convert to per-minute rates
    lambda_per_min = calls_per_hour / 60
    mu_per_min = agents / avg_handle_time_minutes

    # Poisson noise
    sigma_A = np.sqrt(lambda_per_min)
    sigma_S = np.sqrt(mu_per_min) * 0.5  # Approximate

    return DiscreteStochasticQueue(
        lambda_rate=lambda_per_min,
        mu_rate=mu_per_min,
        sigma_arrival=sigma_A,
        sigma_service=sigma_S,
        dt=1.0  # 1 minute slots
    )


def create_network_queue(
    packet_rate_mbps: float = 800,
    link_capacity_mbps: float = 1000,
    packet_size_bits: float = 1500*8,
) -> DiscreteStochasticQueue:
    """
    Create queue for network router/switch.

    Parameters
    ----------
    packet_rate_mbps : float
        Arrival rate [Mbps]
    link_capacity_mbps : float
        Link bandwidth [Mbps]
    packet_size_bits : float
        Average packet size [bits]

    Returns
    -------
    DiscreteStochasticQueue

    Examples
    --------
    >>> # Router with 80% utilization
    >>> router = create_network_queue(
    ...     packet_rate_mbps=800,
    ...     link_capacity_mbps=1000
    ... )
    """
    # Packets per second
    lambda_pps = packet_rate_mbps * 1e6 / packet_size_bits
    mu_pps = link_capacity_mbps * 1e6 / packet_size_bits

    # Use millisecond slots
    dt_ms = 1.0
    lambda_per_slot = lambda_pps / 1000
    mu_per_slot = mu_pps / 1000

    return DiscreteStochasticQueue(
        lambda_rate=lambda_per_slot,
        mu_rate=mu_per_slot,
        sigma_arrival=np.sqrt(lambda_per_slot),
        sigma_service=np.sqrt(mu_per_slot) * 0.5,
        dt=dt_ms
    )