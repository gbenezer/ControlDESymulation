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
ControlDESymulation: Symbolic and Numerical Control Systems Simulation

A comprehensive framework for control systems with multi-backend support
(NumPy, PyTorch, JAX) and extensive numerical integration capabilities.
"""

# Submodules
from . import control, observers, systems, types, visualization

# Core system classes - most commonly used
from .systems.base.core import (
    ContinuousStochasticSystem,
    ContinuousSymbolicSystem,
    DiscreteStochasticSystem,
    DiscreteSymbolicSystem,
    DiscretizedSystem,
    discretize,
)

# Builtin deterministic systems
from .systems.builtin.deterministic.continuous import (
    CartPoleSystem,
    LinearSystem,
    Lorenz,
    PendulumSystem,
    VanDerPolOscillator,
)
from .systems.builtin.deterministic.discrete import (
    DiscreteCartPole,
    DiscreteDoubleIntegrator,
    DiscretePendulum,
)

# Builtin stochastic systems
from .systems.builtin.stochastic.continuous import (
    BrownianMotion,
    CoxIngersollRoss,
    GeometricBrownianMotion,
    LangevinDynamics,
    OrnsteinUhlenbeck,
)
from .systems.builtin.stochastic.discrete import (
    DiscreteAR1,
    DiscreteGARCH11,
    DiscreteRandomWalk,
    DiscreteWhiteNoise,
)

# Aliases for backward compatibility
OrnsteinUhlenbeckProcess = OrnsteinUhlenbeck
OU = OrnsteinUhlenbeck
GBM = GeometricBrownianMotion
CIR = CoxIngersollRoss

# Integrator factories
# Control utilities
from .control import (
    ControlSynthesis,
    SystemAnalysis,
    analyze_controllability,
    analyze_observability,
    analyze_stability,
    design_kalman_filter,
    design_lqg,
    design_lqr,
)

# Observers
from .observers import (
    EKF,
    ExtendedKalmanFilter,
    LinearObserver,
)
from .systems.base.numerical_integration.integrator_factory import IntegratorFactory
from .systems.base.numerical_integration.stochastic.sde_integrator_factory import (
    SDEIntegratorFactory,
)

# Types for type hints
from .types import (  # Core types; Control results; Trajectories
    Backend,
    ControlVector,
    InputMatrix,
    LQGResult,
    LQRResult,
    MPCResult,
    OutputMatrix,
    OutputVector,
    SimulationResult,
    StateMatrix,
    StateTrajectory,
    StateVector,
    TimePoints,
)

# Visualization
from .visualization import (
    ColorSchemes,
    ControlPlotter,
    PhasePortraitPlotter,
    PlotThemes,
    TrajectoryPlotter,
)

__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",
    # Submodules
    "systems",
    "control",
    "observers",
    "visualization",
    "types",
    # Core system classes
    "ContinuousSymbolicSystem",
    "ContinuousStochasticSystem",
    "DiscreteSymbolicSystem",
    "DiscreteStochasticSystem",
    "DiscretizedSystem",
    "discretize",
    # Builtin deterministic continuous
    "LinearSystem",
    "PendulumSystem",
    "CartPoleSystem",
    "Lorenz",
    "VanDerPolOscillator",
    # Builtin deterministic discrete
    "DiscreteDoubleIntegrator",
    "DiscretePendulum",
    "DiscreteCartPole",
    # Builtin stochastic continuous
    "BrownianMotion",
    "OrnsteinUhlenbeck",
    "OrnsteinUhlenbeckProcess",
    "OU",
    "GeometricBrownianMotion",
    "GBM",
    "CoxIngersollRoss",
    "CIR",
    "LangevinDynamics",
    # Builtin stochastic discrete
    "DiscreteAR1",
    "DiscreteWhiteNoise",
    "DiscreteRandomWalk",
    "DiscreteGARCH11",
    # Integration
    "IntegratorFactory",
    "SDEIntegratorFactory",
    # Visualization
    "TrajectoryPlotter",
    "PhasePortraitPlotter",
    "ControlPlotter",
    "ColorSchemes",
    "PlotThemes",
    # Control
    "ControlSynthesis",
    "SystemAnalysis",
    "design_lqr",
    "design_lqg",
    "design_kalman_filter",
    "analyze_stability",
    "analyze_controllability",
    "analyze_observability",
    # Observers
    "LinearObserver",
    "ExtendedKalmanFilter",
    "EKF",
    # Types
    "Backend",
    "StateVector",
    "ControlVector",
    "OutputVector",
    "StateMatrix",
    "InputMatrix",
    "OutputMatrix",
    "TimePoints",
    "SimulationResult",
    "StateTrajectory",
    "LQRResult",
    "LQGResult",
    "MPCResult",
]
