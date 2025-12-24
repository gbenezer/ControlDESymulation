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
Here for documentation of planning

This is the upcoming refactoring

from abc import ABC, abstractmethod

class DiscreteTimeSystemBase(ABC):
    "
    Abstract base for ALL discrete-time systems.
    
    Defines common interface that native discrete and wrappers both implement.
    "
    
    @abstractmethod
    def step(self, x, u):
        "Compute x[k+1] from x[k], u[k]"
        pass
    
    @abstractmethod
    def linearize(self, x_eq, u_eq):
        "Compute discrete-time linearization"
        pass


class DiscreteSymbolicSystem(DiscreteTimeSystemBase):
    "Native discrete system - implements interface directly"
    
    def step(self, x, u):
        return self(x, u)
    
    def linearize(self, x_eq, u_eq):
        return self.linearized_dynamics(x_eq, u_eq)


class DiscretizationWrapper(DiscreteTimeSystemBase):
    "Wrapper for continuous system - delegates to discretizer"
    
    def __init__(self, continuous_system, discretizer):
        if isinstance(continuous_system, DiscreteSymbolicSystem):
            raise TypeError("Cannot wrap discrete system - use directly!")
        self.continuous_system = continuous_system
        self.discretizer = discretizer
    
    def step(self, x, u):
        return self.discretizer.step(x, u)
    
    def linearize(self, x_eq, u_eq):
        return self.discretizer.linearize(x_eq, u_eq)
"""