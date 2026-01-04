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
Base System Infrastructure
==========================

This module provides the foundational infrastructure for defining and simulating
dynamical systems in the ControlDESymulation framework.

Submodules
----------
- **core**: Abstract base classes for continuous, discrete, and stochastic systems
- **numerical_integration**: ODE and SDE integrators for multiple backends
- **utils**: Utility classes for backend management, linearization, code generation

System Classes
--------------
>>> from controldesymulation.systems.base import (
...     # Continuous-time deterministic
...     ContinuousDynamicalSystem,
...     ContinuousSymbolicSystem,
...     # Discrete-time deterministic
...     DiscreteDynamicalSystem,
...     DiscreteSymbolicSystem,
...     # Stochastic systems
...     ContinuousStochasticSystem,
...     DiscreteStochasticSystem,
...     # Discretization
...     DiscretizedSystem,
...     discretize,
... )

Numerical Integration
---------------------
>>> from controldesymulation.systems.base.numerical_integration import (
...     IntegratorFactory,
...     create_integrator,
... )
>>> from controldesymulation.systems.base.numerical_integration.stochastic import (
...     SDEIntegratorFactory,
...     create_sde_integrator,
... )

Utilities
---------
>>> from controldesymulation.systems.base.utils import (
...     BackendManager,
...     DynamicsEvaluator,
...     LinearizationEngine,
...     CodeGenerator,
... )

Authors
-------
Gil Benezer

License
-------
GNU Affero General Public License v3.0
"""

# Core system classes (re-exported for convenience)
# Re-export submodules for convenient access
from . import core, numerical_integration, utils
from .core import (  # Layer 2: Deterministic symbolic systems; Layer 3: Stochastic systems; Layer 1: Abstract base classes; Layer 4: Discretization
    ContinuousDynamicalSystem,
    ContinuousStochasticSystem,
    ContinuousSymbolicSystem,
    ContinuousSystemBase,
    DiscreteDynamicalSystem,
    DiscreteStochasticSystem,
    DiscreteSymbolicSystem,
    DiscreteSystemBase,
    DiscretizationMode,
    DiscretizedSystem,
    StochasticDynamicalSystem,
    SymbolicDynamicalSystem,
    SymbolicSystemBase,
    analyze_discretization_error,
    compute_discretization_quality,
    discretize,
    discretize_batch,
    recommend_dt,
)

# Export public API
__all__ = [
    # Submodules
    "core",
    "numerical_integration",
    "utils",
    # Layer 1: Abstract base classes
    "SymbolicSystemBase",
    "ContinuousSystemBase",
    "DiscreteSystemBase",
    # Layer 2: Deterministic symbolic systems
    "ContinuousSymbolicSystem",
    "ContinuousDynamicalSystem",
    "SymbolicDynamicalSystem",
    "DiscreteDynamicalSystem",
    "DiscreteSymbolicSystem",
    # Layer 3: Stochastic systems
    "ContinuousStochasticSystem",
    "StochasticDynamicalSystem",
    "DiscreteStochasticSystem",
    # Layer 4: Discretization
    "DiscretizationMode",
    "DiscretizedSystem",
    "discretize",
    "discretize_batch",
    "analyze_discretization_error",
    "recommend_dt",
    "compute_discretization_quality",
]
