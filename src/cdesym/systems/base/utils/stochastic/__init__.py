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
Stochastic System Utilities
===========================

This module provides utilities for working with stochastic differential equations (SDEs),
including validation, noise analysis, and diffusion matrix handling.

Validation
----------
Validate SDE system definitions and check mathematical properties:

>>> from controldesymulation.systems.base.utils.stochastic import SDEValidator, validate_sde_system
>>>
>>> # Quick validation
>>> info = validate_sde_system(sde_system)
>>> print(info['is_valid'], info['noise_type'])
>>>
>>> # Detailed validation
>>> validator = SDEValidator(sde_system)
>>> info = validator.validate()

Noise Analysis
--------------
Analyze noise structure and characteristics:

>>> from controldesymulation.systems.base.utils.stochastic import NoiseCharacterizer, analyze_noise_structure
>>>
>>> # Quick analysis
>>> chars = analyze_noise_structure(sde_system)
>>> print(chars.noise_type, chars.is_additive)
>>>
>>> # Detailed analysis
>>> characterizer = NoiseCharacterizer(sde_system)
>>> chars = characterizer.analyze()

Diffusion Handling
------------------
Handle diffusion matrix evaluation and caching:

>>> from controldesymulation.systems.base.utils.stochastic import DiffusionHandler, create_diffusion_handler
>>>
>>> handler = create_diffusion_handler(sde_system)
>>> G = handler.evaluate(x, u)

Authors
-------
Gil Benezer

License
-------
GNU Affero General Public License v3.0
"""

# Validation
# Diffusion handling
from .diffusion_handler import DiffusionHandler, create_diffusion_handler

# Noise analysis
from .noise_analysis import (
    NoiseCharacteristics,
    NoiseCharacterizer,
    analyze_noise_structure,
)
from .sde_validator import (
    SDEValidationInfo,
    SDEValidator,
    ValidationError,
    validate_sde_system,
)

# Export public API
__all__ = [
    # Validation
    "ValidationError",
    "SDEValidationInfo",
    "SDEValidator",
    "validate_sde_system",
    # Noise analysis
    "NoiseCharacteristics",
    "NoiseCharacterizer",
    "analyze_noise_structure",
    # Diffusion handling
    "DiffusionHandler",
    "create_diffusion_handler",
]
