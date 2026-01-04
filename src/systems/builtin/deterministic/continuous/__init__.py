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

from .linear_systems import LinearSystem, AutonomousLinearSystem, LinearSystem2D
from .pendulum import SymbolicPendulum, SymbolicPendulum2ndOrder
from .cartpole import CartPole
from .aerial_systems import SymbolicQuadrotor2D, SymbolicQuadrotor2DLidar, PVTOL
from .dubins_vehicle import DubinsVehicle
from .coupled_oscillators import CoupledOscillatorSystem, NonlinearChainSystem
from .path_tracking import PathTracking
from .van_der_pol import VanDerPolOscillator, ControlledVanDerPolOscillator
from .lorenz import Lorenz, ControlledLorenz
from .duffing import DuffingOscillator
from .fifth_order_system import FifthOrderMechanicalSystem
from .manipulator_2_link import Manipulator2Link
from .continuous_reactors import ContinuousBatchReactor, ContinuousCSTR

# Aliases for convenience
PendulumSystem = SymbolicPendulum
ContinuousPendulum = SymbolicPendulum
ContinuousPendulum2ndOrder = SymbolicPendulum2ndOrder
Pendulum = SymbolicPendulum
CartPoleSystem = CartPole

__all__ = [
    # Linear systems
    "LinearSystem",
    "AutonomousLinearSystem",
    "LinearSystem2D",
    # Pendulum
    "SymbolicPendulum",
    "SymbolicPendulum2ndOrder",
    "PendulumSystem",
    "ContinuousPendulum",
    "ContinuousPendulum2ndOrder",
    "Pendulum",
    # Cart-pole
    "CartPole",
    "CartPoleSystem",
    # Aerial systems
    "SymbolicQuadrotor2D",
    "SymbolicQuadrotor2DLidar",
    "PVTOL",
    # Vehicles
    "DubinsVehicle",
    "PathTracking",
    # Oscillators
    "CoupledOscillatorSystem",
    "NonlinearChainSystem",
    "VanDerPolOscillator",
    "ControlledVanDerPolOscillator",
    "DuffingOscillator",
    # Chaotic systems
    "Lorenz",
    "ControlledLorenz",
    # Mechanical systems
    "FifthOrderMechanicalSystem",
    "Manipulator2Link",
    # Reactor Systems
    "ContinuousBatchReactor",
    "ContinuousCSTR"
]
