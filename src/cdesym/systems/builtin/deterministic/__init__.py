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

from .continuous import (
    PVTOL,
    AutonomousLinearSystem,
    CartPole,
    CartPoleSystem,
    ContinuousBatchReactor,
    ContinuousCSTR,
    ControlledLorenz,
    ControlledVanDerPolOscillator,
    CoupledOscillatorSystem,
    DubinsVehicle,
    DuffingOscillator,
    FifthOrderMechanicalSystem,
    LinearSystem,
    LinearSystem2D,
    Lorenz,
    Manipulator2Link,
    NonlinearChainSystem,
    PathTracking,
    PendulumSystem,
    SymbolicPendulum,
    SymbolicPendulum2ndOrder,
    SymbolicQuadrotor2D,
    SymbolicQuadrotor2DLidar,
    VanDerPolOscillator,
)
from .discrete import (
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
    HenonMap,
    LogisticMap,
    StandardMap,
)

__all__ = [
    # Continuous - Linear systems
    "LinearSystem",
    "AutonomousLinearSystem",
    "LinearSystem2D",
    # Continuous - Pendulum
    "SymbolicPendulum",
    "SymbolicPendulum2ndOrder",
    "PendulumSystem",
    # Continuous - Cart-pole
    "CartPole",
    "CartPoleSystem",
    # Continuous - Aerial systems
    "SymbolicQuadrotor2D",
    "SymbolicQuadrotor2DLidar",
    "PVTOL",
    # Continuous - Vehicles
    "DubinsVehicle",
    "PathTracking",
    # Continuous - Oscillators
    "CoupledOscillatorSystem",
    "NonlinearChainSystem",
    "VanDerPolOscillator",
    "ControlledVanDerPolOscillator",
    "DuffingOscillator",
    # Continuous - Chaotic systems
    "Lorenz",
    "ControlledLorenz",
    # Continuous - Mechanical systems
    "FifthOrderMechanicalSystem",
    "Manipulator2Link",
    # Continuous - Reactor systems
    "ContinuousBatchReactor",
    "ContinuousCSTR",
    # Discrete - Mechanical systems
    "DiscreteOscillator",
    "DiscretePendulum",
    "DiscreteDoubleIntegrator",
    "DiscreteDoubleIntegratorWithForce",
    "DiscreteRobotArm",
    "DiscreteCartPole",
    # Discrete - Mobile robots
    "DifferentialDriveRobot",
    # Discrete - Reactor systems
    "DiscreteBatchReactor",
    "DiscreteCSTR",
    # Discrete - Economic models
    "DiscreteSolowModel",
    # Discrete - Chaotic maps
    "LogisticMap",
    "HenonMap",
    "StandardMap",
]
