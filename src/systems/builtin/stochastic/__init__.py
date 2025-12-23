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
Built-in Stochastic Systems
============================

Collection of commonly used stochastic differential equation systems.
"""

from .ornstein_uhlenbeck import (
    OrnsteinUhlenbeck,
    create_ou_process,
    create_vasicek_model
)

from .geometric_brownian_motion import (
    GeometricBrownianMotion,
    BrownianMotionWithDrift,
    create_stock_price_model
)

from .brownian_motion import (
    BrownianMotion,
    BrownianMotion2D,
    BrownianBridge,
    create_standard_brownian_motion,
    create_scaled_brownian_motion
)

__all__ = [
    # Ornstein-Uhlenbeck
    'OrnsteinUhlenbeck',
    'create_ou_process',
    'create_vasicek_model',
    
    # Geometric Brownian Motion
    'GeometricBrownianMotion',
    'BrownianMotionWithDrift',
    'create_stock_price_model',
    
    # Brownian Motion
    'BrownianMotion',
    'BrownianMotion2D',
    'BrownianBridge',
    'create_standard_brownian_motion',
    'create_scaled_brownian_motion',
]