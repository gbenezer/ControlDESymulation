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


class TrajectoryPlotter:
    """
    Handles all trajectory visualization for dynamical systems.

    Separates plotting logic from system dynamics.
    """

    def __init__(self, system):
        """
        Args:
            system: GenericDiscreteTimeSystem or SymbolicDynamicalSystem
        """
        self.system = system

    def plot_trajectory(self, trajectory, state_names=None, control_sequence=None, **kwargs):
        """Plot time-series trajectory"""
        # ... move from GenericDiscreteTimeSystem ...

    def plot_trajectory_3d(self, trajectory, state_indices=(0, 1, 2), **kwargs):
        """Plot 3D trajectory"""
        # ... move from GenericDiscreteTimeSystem ...

    def plot_phase_portrait_2d(self, trajectory, **kwargs):
        """Plot 2D phase portrait"""
        # ... move from GenericDiscreteTimeSystem ...


# Convenience in GenericDiscreteTimeSystem
class GenericDiscreteTimeSystem:
    @property
    def plot(self):
        """Get plotter for this system"""
        if not hasattr(self, "_plotter"):
            from src.visualization import TrajectoryPlotter

            self._plotter = TrajectoryPlotter(self)
        return self._plotter
