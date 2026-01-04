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

from . import deterministic, stochastic

# NOTE: LogisticMap name collision
# Both deterministic.discrete and stochastic.discrete define LogisticMap:
#   - deterministic.LogisticMap: Classic chaotic map x_{k+1} = r * x_k * (1 - x_k)
#   - stochastic.LogisticMap: Alias for DiscreteStochasticLogisticMap (with noise)
# Only the deterministic LogisticMap is exported at this level.
# Access the stochastic version explicitly:
#   - from controldesymulation.systems.builtin.stochastic import LogisticMap
#   - from controldesymulation.systems.builtin.stochastic.discrete import DiscreteStochasticLogisticMap
# Deterministic systems
from .deterministic import (  # Continuous - Cart-pole; Continuous - Reactor systems; Continuous - Oscillators; Discrete - Mobile robots; Discrete - Reactor systems; Discrete - Mechanical systems; Discrete - Economic models; Continuous - Vehicles; Continuous - Mechanical systems; Continuous - Linear systems; Discrete - Chaotic maps; Continuous - Chaotic systems; Continuous - Pendulum; Continuous - Aerial systems
    PVTOL,
    AutonomousLinearSystem,
    CartPole,
    CartPoleSystem,
    ContinuousBatchReactor,
    ContinuousCSTR,
    ControlledLorenz,
    ControlledVanDerPolOscillator,
    CoupledOscillatorSystem,
    DifferentialDriveRobot,
    DiscreteBatchReactor,
    DiscreteCartPole,
    DiscreteCSTR,
    DiscreteDoubleIntegrator,
    DiscreteDoubleIntegratorWithForce,
    DiscreteOscillator,
    DiscretePendulum,
    DiscreteRobotArm,
    DiscreteSolowModel,
    DubinsVehicle,
    DuffingOscillator,
    FifthOrderMechanicalSystem,
    HenonMap,
    LinearSystem,
    LinearSystem2D,
    LogisticMap,
    Lorenz,
    Manipulator2Link,
    NonlinearChainSystem,
    PathTracking,
    PendulumSystem,
    StandardMap,
    SymbolicPendulum,
    SymbolicPendulum2ndOrder,
    SymbolicQuadrotor2D,
    SymbolicQuadrotor2DLidar,
    VanDerPolOscillator,
)

# Stochastic systems
from .stochastic import (  # Continuous - Brownian motion; Continuous - Reactor systems; Continuous - Cox-Ingersoll-Ross process; Discrete - Time series models; Discrete - Financial models; Discrete - Reactor systems; Discrete - Stochastic control systems; Discrete - Nonlinear dynamics; Discrete - Queueing systems; Discrete - Multivariate models; Continuous - Langevin dynamics; Continuous - Ornstein-Uhlenbeck process; Continuous - Stochastic control systems; Continuous - Epidemiological models
    AR1,
    ARMA11,
    CIR,
    GARCH11,
    GBM,
    OU,
    SIR,
    VAR1,
    BrownianMotion,
    BrownianMotion2D,
    BrownianMotionWithDrift,
    ContinuousStochasticBatchReactor,
    ContinuousStochasticCSTR,
    ContinuousStochasticPendulum,
    CoxIngersollRoss,
    DiscreteAR1,
    DiscreteARMA11,
    DiscreteGARCH11,
    DiscreteRandomWalk,
    DiscreteStochasticBatchReactor,
    DiscreteStochasticCSTR,
    DiscreteStochasticDoubleIntegrator,
    DiscreteStochasticLogisticMap,
    DiscreteStochasticPendulum,
    DiscreteStochasticQueue,
    DiscreteVAR1,
    DiscreteWhiteNoise,
    GeometricBrownianMotion,
    LangevinDynamics,
    MultivariateOrnsteinUhlenbeck,
    OrnsteinUhlenbeck,
    RandomWalk,
    StochasticCartPole,
    StochasticDoubleIntegrator,
    StochasticLorenz,
    StochasticQueue,
    StochasticSIR,
    WhiteNoise,
    create_biased_random_walk,
    create_bivariate_var,
    create_call_center_queue,
    create_chaotic_regime,
    create_digital_servo,
    create_discrete_batch_reactor_with_noise,
    create_discrete_stochastic_cstr_with_noise,
    create_economic_arma,
    create_edge_of_chaos,
    create_equity_garch,
    create_fixed_point_regime,
    create_furuta_pendulum,
    create_fx_garch,
    create_lqg_benchmark_discrete,
    create_macro_var,
    create_measurement_noise,
    create_network_queue,
    create_rl_pendulum,
    create_sensor_arma,
    create_spacecraft_discrete,
    create_standard_white_noise,
    create_symmetric_random_walk,
    create_thermal_noise,
)

__all__ = [
    # Submodules
    "deterministic",
    "stochastic",
    # Deterministic Continuous - Linear systems
    "LinearSystem",
    "AutonomousLinearSystem",
    "LinearSystem2D",
    # Deterministic Continuous - Pendulum
    "SymbolicPendulum",
    "SymbolicPendulum2ndOrder",
    "PendulumSystem",
    # Deterministic Continuous - Cart-pole
    "CartPole",
    "CartPoleSystem",
    # Deterministic Continuous - Aerial systems
    "SymbolicQuadrotor2D",
    "SymbolicQuadrotor2DLidar",
    "PVTOL",
    # Deterministic Continuous - Vehicles
    "DubinsVehicle",
    "PathTracking",
    # Deterministic Continuous - Oscillators
    "CoupledOscillatorSystem",
    "NonlinearChainSystem",
    "VanDerPolOscillator",
    "ControlledVanDerPolOscillator",
    "DuffingOscillator",
    # Deterministic Continuous - Chaotic systems
    "Lorenz",
    "ControlledLorenz",
    # Deterministic Continuous - Mechanical systems
    "FifthOrderMechanicalSystem",
    "Manipulator2Link",
    # Deterministic Continuous - Reactor systems
    "ContinuousBatchReactor",
    "ContinuousCSTR",
    # Deterministic Discrete - Mechanical systems
    "DiscreteOscillator",
    "DiscretePendulum",
    "DiscreteDoubleIntegrator",
    "DiscreteDoubleIntegratorWithForce",
    "DiscreteRobotArm",
    "DiscreteCartPole",
    # Deterministic Discrete - Mobile robots
    "DifferentialDriveRobot",
    # Deterministic Discrete - Reactor systems
    "DiscreteBatchReactor",
    "DiscreteCSTR",
    # Deterministic Discrete - Economic models
    "DiscreteSolowModel",
    # Deterministic Discrete - Chaotic maps
    "LogisticMap",
    "HenonMap",
    "StandardMap",
    # Stochastic Continuous - Brownian motion
    "BrownianMotion",
    "BrownianMotion2D",
    "GeometricBrownianMotion",
    "BrownianMotionWithDrift",
    "GBM",
    # Stochastic Continuous - Ornstein-Uhlenbeck process
    "OrnsteinUhlenbeck",
    "MultivariateOrnsteinUhlenbeck",
    "OU",
    # Stochastic Continuous - Cox-Ingersoll-Ross process
    "CoxIngersollRoss",
    "CIR",
    # Stochastic Continuous - Langevin dynamics
    "LangevinDynamics",
    # Stochastic Continuous - Control systems
    "StochasticDoubleIntegrator",
    "ContinuousStochasticPendulum",
    "StochasticCartPole",
    "StochasticLorenz",
    # Stochastic Continuous - Epidemiological models
    "StochasticSIR",
    "SIR",
    # Stochastic Continuous - Reactor systems
    "ContinuousStochasticBatchReactor",
    "ContinuousStochasticCSTR",
    # Stochastic Discrete - Time series models
    "DiscreteAR1",
    "AR1",
    "DiscreteWhiteNoise",
    "WhiteNoise",
    "create_standard_white_noise",
    "create_measurement_noise",
    "create_thermal_noise",
    "DiscreteRandomWalk",
    "RandomWalk",
    "create_symmetric_random_walk",
    "create_biased_random_walk",
    "DiscreteARMA11",
    "ARMA11",
    "create_economic_arma",
    "create_sensor_arma",
    # Stochastic Discrete - Multivariate models
    "DiscreteVAR1",
    "VAR1",
    "create_bivariate_var",
    "create_macro_var",
    # Stochastic Discrete - Financial models
    "DiscreteGARCH11",
    "GARCH11",
    "create_equity_garch",
    "create_fx_garch",
    # Stochastic Discrete - Control systems
    "DiscreteStochasticDoubleIntegrator",
    "create_digital_servo",
    "create_lqg_benchmark_discrete",
    "create_spacecraft_discrete",
    "DiscreteStochasticPendulum",
    "create_furuta_pendulum",
    "create_rl_pendulum",
    # Stochastic Discrete - Queueing systems
    "DiscreteStochasticQueue",
    "StochasticQueue",
    "create_call_center_queue",
    "create_network_queue",
    # Stochastic Discrete - Reactor systems
    "DiscreteStochasticBatchReactor",
    "DiscreteStochasticCSTR",
    "create_discrete_batch_reactor_with_noise",
    "create_discrete_stochastic_cstr_with_noise",
    # Stochastic Discrete - Nonlinear dynamics
    "DiscreteStochasticLogisticMap",
    "create_fixed_point_regime",
    "create_chaotic_regime",
    "create_edge_of_chaos",
]
