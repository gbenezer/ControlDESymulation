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

from .brownian_motion import BrownianMotion, BrownianMotion2D
from .geometric_brownian_motion import GeometricBrownianMotion, BrownianMotionWithDrift
from .ornstein_uhlenbeck import OrnsteinUhlenbeck, MultivariateOrnsteinUhlenbeck
from .cox_ingersoll_ross import CoxIngersollRoss
from .langevin_dynamics import LangevinDynamics
from .stochastic_double_integrator import StochasticDoubleIntegrator
from .continuous_stochastic_pendulum import ContinuousStochasticPendulum
from .stochastic_cartpole import StochasticCartPole
from .stochastic_lorenz import StochasticLorenz
from .continuous_sir import StochasticSIR
from .continuous_stochastic_reactors import ContinuousStochasticBatchReactor, ContinuousStochasticCSTR

# Aliases for convenience
StochasticPendulum = ContinuousStochasticPendulum
CIR = CoxIngersollRoss
OU = OrnsteinUhlenbeck
GBM = GeometricBrownianMotion
SIR = StochasticSIR

__all__ = [
    # Brownian motion
    "BrownianMotion",
    "BrownianMotion2D",
    "GeometricBrownianMotion",
    "BrownianMotionWithDrift",
    "GBM",
    # Ornstein-Uhlenbeck process
    "OrnsteinUhlenbeck",
    "MultivariateOrnsteinUhlenbeck",
    "OU",
    # Cox-Ingersoll-Ross process
    "CoxIngersollRoss",
    "CIR",
    # Langevin dynamics
    "LangevinDynamics",
    # Stochastic control systems
    "StochasticDoubleIntegrator",
    "ContinuousStochasticPendulum",
    "StochasticPendulum",
    "StochasticCartPole",
    "StochasticLorenz",
    # Epidemiological models
    "StochasticSIR",
    "SIR",
    # Reactor systems
    "ContinuousStochasticBatchReactor",
    "ContinuousStochasticCSTR",
]
