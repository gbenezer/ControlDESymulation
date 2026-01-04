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

# Library version
__version__ = "1.0.0"

# Submodules
from . import systems
from . import control
from . import observers
from . import visualization
from . import types

# Core system classes - most commonly used
from .systems.base.core import (
    ContinuousSymbolicSystem,
    ContinuousStochasticSystem,
    DiscreteSymbolicSystem,
    DiscreteStochasticSystem,
    DiscretizedSystem,
    discretize,
)

# Builtin deterministic systems
from .systems.builtin.deterministic.continuous import (
    LinearSystem,
    PendulumSystem,
    CartPoleSystem,
    Lorenz,
    VanDerPolOscillator,
)

from .systems.builtin.deterministic.discrete import (
    DiscreteDoubleIntegrator,
    DiscretePendulum,
    DiscreteCartPole,
)

# Builtin stochastic systems
from .systems.builtin.stochastic.continuous import (
    BrownianMotion,
    OrnsteinUhlenbeck,
    GeometricBrownianMotion,
    CoxIngersollRoss,
    LangevinDynamics,
)

from .systems.builtin.stochastic.discrete import (
    DiscreteAR1,
    DiscreteWhiteNoise,
    DiscreteRandomWalk,
    DiscreteGARCH11,
)

# Aliases for backward compatibility
OrnsteinUhlenbeckProcess = OrnsteinUhlenbeck
OU = OrnsteinUhlenbeck
GBM = GeometricBrownianMotion
CIR = CoxIngersollRoss

# Integrator factories
from .systems.base.numerical_integration.integrator_factory import IntegratorFactory
from .systems.base.numerical_integration.stochastic.sde_integrator_factory import SDEIntegratorFactory

# Visualization
from .visualization import (
    TrajectoryPlotter,
    PhasePortraitPlotter,
    ControlPlotter,
    ColorSchemes,
    PlotThemes,
)

# Control utilities
from .control import (
    ControlSynthesis,
    SystemAnalysis,
    design_lqr,
    design_lqg,
    design_kalman_filter,
    analyze_stability,
    analyze_controllability,
    analyze_observability,
)

# Observers
from .observers import (
    LinearObserver,
    ExtendedKalmanFilter,
    EKF,
)

# Types for type hints
from .types import (
    # Core types
    Backend,
    StateVector,
    ControlVector,
    OutputVector,
    StateMatrix,
    InputMatrix,
    OutputMatrix,
    # Trajectories
    TimePoints,
    SimulationResult,
    StateTrajectory,
    # Control results
    LQRResult,
    LQGResult,
    MPCResult,
)

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