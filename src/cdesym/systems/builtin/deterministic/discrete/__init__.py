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

from .differential_drive_robot import DifferentialDriveRobot
from .discrete_cartpole import DiscreteCartPole
from .discrete_double_integrator import DiscreteDoubleIntegrator, DiscreteDoubleIntegratorWithForce
from .discrete_oscillator import DiscreteOscillator
from .discrete_pendulum import DiscretePendulum
from .discrete_reactors import DiscreteBatchReactor, DiscreteCSTR
from .discrete_robot_arm import DiscreteRobotArm
from .discrete_solow_model import DiscreteSolowModel
from .henon_map import HenonMap
from .logistic_map import LogisticMap
from .standard_map import StandardMap

__all__ = [
    # Discrete mechanical systems
    "DiscreteOscillator",
    "DiscretePendulum",
    "DiscreteDoubleIntegrator",
    "DiscreteDoubleIntegratorWithForce",
    "DiscreteRobotArm",
    "DiscreteCartPole",
    # Mobile robots
    "DifferentialDriveRobot",
    # Reactor systems
    "DiscreteBatchReactor",
    "DiscreteCSTR",
    # Economic models
    "DiscreteSolowModel",
    # Chaotic maps
    "LogisticMap",
    "HenonMap",
    "StandardMap",
]
