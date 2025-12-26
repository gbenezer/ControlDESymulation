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
Unit Tests for Noise Analysis Module

Tests noise characterization, classification, and solver recommendations.
Pure SymPy analysis - no code generation or numerical evaluation.
"""

from typing import Set

import pytest
import sympy as sp

from src.systems.base.utils.stochastic.noise_analysis import (
    NoiseCharacteristics,
    NoiseCharacterizer,
    NoiseType,
    SDEType,
)

# ============================================================================
# Fixtures - Test Diffusion Expressions
# ============================================================================


@pytest.fixture
def additive_noise_2d():
    """2D system with additive (constant) noise."""
    x1, x2 = sp.symbols("x1 x2")
    u = sp.symbols("u")

    # Constant diffusion - doesn't depend on anything
    diffusion = sp.Matrix([[0.1], [0.2]])

    return {
        "diffusion": diffusion,
        "state_vars": [x1, x2],
        "control_vars": [u],
        "nx": 2,
        "nw": 1,
    }


@pytest.fixture
def multiplicative_noise_1d():
    """1D system with state-dependent (multiplicative) noise."""
    x = sp.symbols("x")
    u = sp.symbols("u")

    # State-dependent diffusion
    diffusion = sp.Matrix([[0.2 * x]])

    return {
        "diffusion": diffusion,
        "state_vars": [x],
        "control_vars": [u],
        "nx": 1,
        "nw": 1,
    }


@pytest.fixture
def diagonal_noise_3d():
    """3D system with diagonal multiplicative noise."""
    x1, x2, x3 = sp.symbols("x1 x2 x3")
    u = sp.symbols("u")

    # Diagonal diffusion - each state has independent noise
    diffusion = sp.Matrix([[0.1 * x1, 0, 0], [0, 0.2 * x2, 0], [0, 0, 0.3 * x3]])

    return {
        "diffusion": diffusion,
        "state_vars": [x1, x2, x3],
        "control_vars": [u],
        "nx": 3,
        "nw": 3,
    }


@pytest.fixture
def scalar_noise_2d():
    """2D system with scalar noise (single Wiener process)."""
    x1, x2 = sp.symbols("x1 x2")
    u = sp.symbols("u")

    # Single noise source affects both states
    diffusion = sp.Matrix([[0.1], [0.2]])

    return {
        "diffusion": diffusion,
        "state_vars": [x1, x2],
        "control_vars": [u],
        "nx": 2,
        "nw": 1,
    }


@pytest.fixture
def control_dependent_noise():
    """System where noise depends on control input."""
    x = sp.symbols("x")
    u = sp.symbols("u")

    # Control-modulated noise
    diffusion = sp.Matrix([[0.1 * (1 + u)]])

    return {
        "diffusion": diffusion,
        "state_vars": [x],
        "control_vars": [u],
        "nx": 1,
        "nw": 1,
    }


@pytest.fixture
def time_varying_noise():
    """System with time-varying noise."""
    x = sp.symbols("x")
    u = sp.symbols("u")
    t = sp.symbols("t")

    # Time-varying diffusion
    diffusion = sp.Matrix([[0.1 * sp.sin(t)]])

    return {
        "diffusion": diffusion,
        "state_vars": [x],
        "control_vars": [u],
        "time_var": t,
        "nx": 1,
        "nw": 1,
    }


@pytest.fixture
def general_noise_2d():
    """2D system with general (fully coupled) noise."""
    x1, x2 = sp.symbols("x1 x2")
    u = sp.symbols("u")

    # Full matrix - states couple through noise
    diffusion = sp.Matrix([[0.1 * x1, 0.05 * x2], [0.05 * x1, 0.2 * x2]])

    return {
        "diffusion": diffusion,
        "state_vars": [x1, x2],
        "control_vars": [u],
        "nx": 2,
        "nw": 2,
    }


@pytest.fixture
def rectangular_noise():
    """System with rectangular diffusion matrix (nx != nw)."""
    x1, x2, x3 = sp.symbols("x1 x2 x3")
    u = sp.symbols("u")

    # 3 states, 2 noise sources
    diffusion = sp.Matrix([[0.1, 0.0], [0.0, 0.2], [0.1, 0.1]])

    return {
        "diffusion": diffusion,
        "state_vars": [x1, x2, x3],
        "control_vars": [u],
        "nx": 3,
        "nw": 2,
    }


@pytest.fixture
def mixed_dependency_noise():
    """System with noise depending on state, control, and time."""
    x = sp.symbols("x")
    u = sp.symbols("u")
    t = sp.symbols("t")

    # Depends on everything
    diffusion = sp.Matrix([[0.1 * x * (1 + u) * sp.cos(t)]])

    return {
        "diffusion": diffusion,
        "state_vars": [x],
        "control_vars": [u],
        "time_var": t,
        "nx": 1,
        "nw": 1,
    }


# ============================================================================
# Test NoiseType Enum
# ============================================================================


class TestNoiseTypeEnum:
    """Test NoiseType enumeration."""

    def test_all_types_defined(self):
        """Test that all expected noise types exist."""
        assert hasattr(NoiseType, "ADDITIVE")
        assert hasattr(NoiseType, "MULTIPLICATIVE")
        assert hasattr(NoiseType, "DIAGONAL")
        assert hasattr(NoiseType, "SCALAR")
        assert hasattr(NoiseType, "GENERAL")

    def test_enum_values(self):
        """Test enum string values."""
        assert NoiseType.ADDITIVE.value == "additive"
        assert NoiseType.MULTIPLICATIVE.value == "multiplicative"
        assert NoiseType.DIAGONAL.value == "diagonal"
        assert NoiseType.SCALAR.value == "scalar"
        assert NoiseType.GENERAL.value == "general"

    def test_enum_comparison(self):
        """Test enum comparison and equality."""
        assert NoiseType.ADDITIVE == NoiseType.ADDITIVE
        assert NoiseType.ADDITIVE != NoiseType.MULTIPLICATIVE


# ============================================================================
# Test SDEType Enum
# ============================================================================


class TestSDETypeEnum:
    """Test SDEType enumeration."""

    def test_sde_types_defined(self):
        """Test that SDE interpretation types exist."""
        assert hasattr(SDEType, "ITO")
        assert hasattr(SDEType, "STRATONOVICH")

    def test_sde_values(self):
        """Test SDE type values."""
        assert SDEType.ITO.value == "ito"
        assert SDEType.STRATONOVICH.value == "stratonovich"


# ============================================================================
# Test NoiseCharacteristics Dataclass
# ============================================================================


class TestNoiseCharacteristics:
    """Test NoiseCharacteristics dataclass."""

    def test_dataclass_creation(self):
        """Test creating NoiseCharacteristics instance."""
        char = NoiseCharacteristics(
            noise_type=NoiseType.ADDITIVE,
            num_wiener=1,
            is_additive=True,
            is_multiplicative=False,
            is_diagonal=False,
            is_scalar=True,
            depends_on_state=False,
            depends_on_control=False,
            depends_on_time=False,
            state_dependencies=set(),
            control_dependencies=set(),
        )

        assert char.noise_type == NoiseType.ADDITIVE
        assert char.num_wiener == 1
        assert char.is_additive
        assert char.is_scalar
        assert not char.is_multiplicative

    def test_recommended_solvers_jax_additive(self):
        """Test JAX solver recommendations for additive noise."""
        char = NoiseCharacteristics(
            noise_type=NoiseType.ADDITIVE,
            num_wiener=1,
            is_additive=True,
            is_multiplicative=False,
            is_diagonal=False,
            is_scalar=True,
            depends_on_state=False,
            depends_on_control=False,
            depends_on_time=False,
            state_dependencies=set(),
            control_dependencies=set(),
        )

        solvers = char.recommended_solvers("jax")
        assert "sea" in solvers
        assert "shark" in solvers
        assert "sra1" in solvers

    def test_recommended_solvers_jax_scalar(self):
        """Test JAX solver recommendations for scalar noise."""
        char = NoiseCharacteristics(
            noise_type=NoiseType.SCALAR,
            num_wiener=1,
            is_additive=False,
            is_multiplicative=True,
            is_diagonal=False,
            is_scalar=True,
            depends_on_state=True,
            depends_on_control=False,
            depends_on_time=False,
            state_dependencies={sp.Symbol("x")},
            control_dependencies=set(),
        )

        solvers = char.recommended_solvers("jax")
        assert "euler_heun" in solvers
        assert "heun" in solvers

    def test_recommended_solvers_jax_diagonal(self):
        """Test JAX solver recommendations for diagonal noise."""
        char = NoiseCharacteristics(
            noise_type=NoiseType.DIAGONAL,
            num_wiener=3,
            is_additive=False,
            is_multiplicative=True,
            is_diagonal=True,
            is_scalar=False,
            depends_on_state=True,
            depends_on_control=False,
            depends_on_time=False,
            state_dependencies={sp.Symbol("x1"), sp.Symbol("x2")},
            control_dependencies=set(),
        )

        solvers = char.recommended_solvers("jax")
        assert "euler_heun" in solvers
        assert "spark" in solvers

    def test_recommended_solvers_torch(self):
        """Test PyTorch solver recommendations."""
        char = NoiseCharacteristics(
            noise_type=NoiseType.ADDITIVE,
            num_wiener=1,
            is_additive=True,
            is_multiplicative=False,
            is_diagonal=False,
            is_scalar=True,
            depends_on_state=False,
            depends_on_control=False,
            depends_on_time=False,
            state_dependencies=set(),
            control_dependencies=set(),
        )

        solvers = char.recommended_solvers("torch")
        assert len(solvers) > 0
        assert any(s in ["euler", "milstein", "srk"] for s in solvers)

    def test_recommended_solvers_numpy(self):
        """Test NumPy (DiffEqPy) solver recommendations."""
        char = NoiseCharacteristics(
            noise_type=NoiseType.ADDITIVE,
            num_wiener=1,
            is_additive=True,
            is_multiplicative=False,
            is_diagonal=False,
            is_scalar=True,
            depends_on_state=False,
            depends_on_control=False,
            depends_on_time=False,
            state_dependencies=set(),
            control_dependencies=set(),
        )

        solvers = char.recommended_solvers("numpy")
        assert len(solvers) > 0
        assert any(s.startswith("SRA") for s in solvers)


# ============================================================================
# Test NoiseCharacterizer Initialization
# ============================================================================


class TestNoiseCharacterizerInit:
    """Test NoiseCharacterizer initialization."""

    def test_basic_initialization(self, additive_noise_2d):
        """Test basic characterizer creation."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        assert char.nx == 2
        assert char.nw == 1
        assert len(char.state_vars) == 2
        assert len(char.control_vars) == 1
        assert char.time_var is None

    def test_initialization_with_time(self, time_varying_noise):
        """Test initialization with time variable."""
        char = NoiseCharacterizer(
            time_varying_noise["diffusion"],
            time_varying_noise["state_vars"],
            time_varying_noise["control_vars"],
            time_var=time_varying_noise["time_var"],
        )

        assert char.time_var is not None
        assert char.time_var == time_varying_noise["time_var"]

    def test_dimension_extraction(self, diagonal_noise_3d):
        """Test automatic dimension extraction."""
        char = NoiseCharacterizer(
            diagonal_noise_3d["diffusion"],
            diagonal_noise_3d["state_vars"],
            diagonal_noise_3d["control_vars"],
        )

        assert char.nx == 3
        assert char.nw == 3

    def test_rectangular_matrix(self, rectangular_noise):
        """Test with rectangular diffusion matrix."""
        char = NoiseCharacterizer(
            rectangular_noise["diffusion"],
            rectangular_noise["state_vars"],
            rectangular_noise["control_vars"],
        )

        assert char.nx == 3
        assert char.nw == 2


# ============================================================================
# Test Additive Noise Detection
# ============================================================================


class TestAdditiveNoiseDetection:
    """Test detection of additive (constant) noise."""

    def test_detect_additive_noise(self, additive_noise_2d):
        """Test detection of additive noise."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        result = char.analyze()

        assert result.is_additive
        assert result.noise_type == NoiseType.ADDITIVE
        assert not result.depends_on_state
        assert not result.depends_on_control
        assert not result.depends_on_time

    def test_additive_has_no_dependencies(self, additive_noise_2d):
        """Test that additive noise has no dependencies."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        result = char.analyze()

        assert len(result.state_dependencies) == 0
        assert len(result.control_dependencies) == 0

    def test_additive_scalar_noise(self, scalar_noise_2d):
        """Test additive noise with scalar noise source."""
        char = NoiseCharacterizer(
            scalar_noise_2d["diffusion"],
            scalar_noise_2d["state_vars"],
            scalar_noise_2d["control_vars"],
        )

        result = char.analyze()

        assert result.is_additive
        assert result.is_scalar
        assert result.noise_type == NoiseType.ADDITIVE  # Additive takes priority


# ============================================================================
# Test Multiplicative Noise Detection
# ============================================================================


class TestMultiplicativeNoiseDetection:
    """Test detection of multiplicative (state-dependent) noise."""

    def test_detect_multiplicative_noise(self, multiplicative_noise_1d):
        """Test detection of multiplicative noise."""
        char = NoiseCharacterizer(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
        )

        result = char.analyze()

        assert result.is_multiplicative
        assert result.depends_on_state
        assert not result.is_additive

    def test_multiplicative_state_dependencies(self, multiplicative_noise_1d):
        """Test tracking of state dependencies."""
        char = NoiseCharacterizer(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
        )

        result = char.analyze()

        assert len(result.state_dependencies) == 1
        assert multiplicative_noise_1d["state_vars"][0] in result.state_dependencies

    def test_multiplicative_general_matrix(self, general_noise_2d):
        """Test multiplicative noise with general (coupled) matrix."""
        char = NoiseCharacterizer(
            general_noise_2d["diffusion"],
            general_noise_2d["state_vars"],
            general_noise_2d["control_vars"],
        )

        result = char.analyze()

        assert result.is_multiplicative
        assert result.depends_on_state
        assert len(result.state_dependencies) == 2  # Both x1 and x2


# ============================================================================
# Test Diagonal Noise Detection
# ============================================================================


class TestDiagonalNoiseDetection:
    """Test detection of diagonal noise structure."""

    def test_detect_diagonal_noise(self, diagonal_noise_3d):
        """Test detection of diagonal noise matrix."""
        char = NoiseCharacterizer(
            diagonal_noise_3d["diffusion"],
            diagonal_noise_3d["state_vars"],
            diagonal_noise_3d["control_vars"],
        )

        result = char.analyze()

        assert result.is_diagonal
        assert result.noise_type == NoiseType.DIAGONAL

    def test_non_diagonal_fails(self, general_noise_2d):
        """Test that coupled noise is not classified as diagonal."""
        char = NoiseCharacterizer(
            general_noise_2d["diffusion"],
            general_noise_2d["state_vars"],
            general_noise_2d["control_vars"],
        )

        result = char.analyze()

        assert not result.is_diagonal

    def test_rectangular_not_diagonal(self, rectangular_noise):
        """Test that rectangular matrices cannot be diagonal."""
        char = NoiseCharacterizer(
            rectangular_noise["diffusion"],
            rectangular_noise["state_vars"],
            rectangular_noise["control_vars"],
        )

        result = char.analyze()

        assert not result.is_diagonal  # nx != nw

    def test_diagonal_with_zero_elements(self):
        """Test diagonal detection with some zero diagonal elements."""
        x1, x2, x3 = sp.symbols("x1 x2 x3")
        u = sp.symbols("u")

        # Diagonal with zero on diagonal
        diffusion = sp.Matrix(
            [[0.1 * x1, 0, 0], [0, 0, 0], [0, 0, 0.3 * x3]]  # Zero element on diagonal
        )

        char = NoiseCharacterizer(diffusion, [x1, x2, x3], [u])
        result = char.analyze()

        assert result.is_diagonal  # Still diagonal structure


# ============================================================================
# Test Scalar Noise Detection
# ============================================================================


class TestScalarNoiseDetection:
    """Test detection of scalar noise (single Wiener process)."""

    def test_detect_scalar_noise(self, scalar_noise_2d):
        """Test detection of scalar noise."""
        char = NoiseCharacterizer(
            scalar_noise_2d["diffusion"],
            scalar_noise_2d["state_vars"],
            scalar_noise_2d["control_vars"],
        )

        result = char.analyze()

        assert result.is_scalar
        assert result.num_wiener == 1

    def test_multiplicative_scalar(self, multiplicative_noise_1d):
        """Test scalar multiplicative noise."""
        char = NoiseCharacterizer(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
        )

        result = char.analyze()

        assert result.is_scalar
        assert result.is_multiplicative
        assert result.noise_type == NoiseType.SCALAR  # Scalar takes priority over multiplicative

    def test_non_scalar_multiple_wiener(self, diagonal_noise_3d):
        """Test that multiple Wiener processes are not scalar."""
        char = NoiseCharacterizer(
            diagonal_noise_3d["diffusion"],
            diagonal_noise_3d["state_vars"],
            diagonal_noise_3d["control_vars"],
        )

        result = char.analyze()

        assert not result.is_scalar
        assert result.num_wiener == 3


# ============================================================================
# Test Control Dependency Detection
# ============================================================================


class TestControlDependencyDetection:
    """Test detection of control-dependent noise."""

    def test_detect_control_dependency(self, control_dependent_noise):
        """Test detection of control-dependent noise."""
        char = NoiseCharacterizer(
            control_dependent_noise["diffusion"],
            control_dependent_noise["state_vars"],
            control_dependent_noise["control_vars"],
        )

        result = char.analyze()

        assert result.depends_on_control
        assert not result.is_additive
        assert len(result.control_dependencies) == 1

    def test_control_dependency_tracking(self, control_dependent_noise):
        """Test tracking of specific control dependencies."""
        char = NoiseCharacterizer(
            control_dependent_noise["diffusion"],
            control_dependent_noise["state_vars"],
            control_dependent_noise["control_vars"],
        )

        result = char.analyze()

        assert control_dependent_noise["control_vars"][0] in result.control_dependencies

    def test_no_control_dependency_additive(self, additive_noise_2d):
        """Test that additive noise has no control dependency."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        result = char.analyze()

        assert not result.depends_on_control
        assert len(result.control_dependencies) == 0


# ============================================================================
# Test Time Dependency Detection
# ============================================================================


class TestTimeDependencyDetection:
    """Test detection of time-varying noise."""

    def test_detect_time_dependency(self, time_varying_noise):
        """Test detection of time-varying noise."""
        char = NoiseCharacterizer(
            time_varying_noise["diffusion"],
            time_varying_noise["state_vars"],
            time_varying_noise["control_vars"],
            time_var=time_varying_noise["time_var"],
        )

        result = char.analyze()

        assert result.depends_on_time
        assert not result.is_additive

    def test_no_time_dependency_without_time_var(self, additive_noise_2d):
        """Test that systems without time_var don't have time dependency."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
            time_var=None,
        )

        result = char.analyze()

        assert not result.depends_on_time


# ============================================================================
# Test Mixed Dependencies
# ============================================================================


class TestMixedDependencies:
    """Test noise with multiple dependencies."""

    def test_mixed_dependencies(self, mixed_dependency_noise):
        """Test noise depending on state, control, and time."""
        char = NoiseCharacterizer(
            mixed_dependency_noise["diffusion"],
            mixed_dependency_noise["state_vars"],
            mixed_dependency_noise["control_vars"],
            time_var=mixed_dependency_noise["time_var"],
        )

        result = char.analyze()

        assert result.depends_on_state
        assert result.depends_on_control
        assert result.depends_on_time
        assert not result.is_additive
        assert result.is_multiplicative

    def test_all_dependencies_tracked(self, mixed_dependency_noise):
        """Test that all dependencies are tracked."""
        char = NoiseCharacterizer(
            mixed_dependency_noise["diffusion"],
            mixed_dependency_noise["state_vars"],
            mixed_dependency_noise["control_vars"],
            time_var=mixed_dependency_noise["time_var"],
        )

        result = char.analyze()

        assert len(result.state_dependencies) == 1
        assert len(result.control_dependencies) == 1


# ============================================================================
# Test Noise Type Classification Priority
# ============================================================================


class TestNoiseTypeClassification:
    """Test noise type classification logic and priority."""

    def test_additive_priority(self, additive_noise_2d):
        """Test that additive has highest priority."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        result = char.analyze()
        assert result.noise_type == NoiseType.ADDITIVE

    def test_scalar_priority_over_multiplicative(self, multiplicative_noise_1d):
        """Test that scalar has priority over multiplicative."""
        char = NoiseCharacterizer(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
        )

        result = char.analyze()
        # Should be SCALAR, not MULTIPLICATIVE (even though it is multiplicative)
        assert result.noise_type == NoiseType.SCALAR
        assert result.is_scalar
        assert result.is_multiplicative  # But still marked as multiplicative

    def test_diagonal_priority(self, diagonal_noise_3d):
        """Test diagonal classification."""
        char = NoiseCharacterizer(
            diagonal_noise_3d["diffusion"],
            diagonal_noise_3d["state_vars"],
            diagonal_noise_3d["control_vars"],
        )

        result = char.analyze()
        assert result.noise_type == NoiseType.DIAGONAL

    def test_general_fallback(self, general_noise_2d):
        """Test that general is the fallback type."""
        char = NoiseCharacterizer(
            general_noise_2d["diffusion"],
            general_noise_2d["state_vars"],
            general_noise_2d["control_vars"],
        )

        result = char.analyze()
        assert result.noise_type == NoiseType.GENERAL


# ============================================================================
# Test Validation Methods
# ============================================================================


class TestValidation:
    """Test noise type validation methods."""

    def test_validate_additive_claim_correct(self, additive_noise_2d):
        """Test validation of correct additive claim."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        # Should not raise
        assert char.validate_noise_type_claim("additive")

    def test_validate_additive_claim_incorrect(self, multiplicative_noise_1d):
        """Test validation catches incorrect additive claim."""
        char = NoiseCharacterizer(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
        )

        with pytest.raises(ValueError, match="Claimed noise_type='additive'"):
            char.validate_noise_type_claim("additive")

    def test_validate_diagonal_claim_correct(self, diagonal_noise_3d):
        """Test validation of correct diagonal claim."""
        char = NoiseCharacterizer(
            diagonal_noise_3d["diffusion"],
            diagonal_noise_3d["state_vars"],
            diagonal_noise_3d["control_vars"],
        )

        assert char.validate_noise_type_claim("diagonal")

    def test_validate_diagonal_claim_incorrect(self, general_noise_2d):
        """Test validation catches incorrect diagonal claim."""
        char = NoiseCharacterizer(
            general_noise_2d["diffusion"],
            general_noise_2d["state_vars"],
            general_noise_2d["control_vars"],
        )

        with pytest.raises(ValueError, match="Claimed noise_type='diagonal'"):
            char.validate_noise_type_claim("diagonal")

    def test_validate_scalar_claim_correct(self, scalar_noise_2d):
        """Test validation of correct scalar claim."""
        char = NoiseCharacterizer(
            scalar_noise_2d["diffusion"],
            scalar_noise_2d["state_vars"],
            scalar_noise_2d["control_vars"],
        )

        assert char.validate_noise_type_claim("scalar")

    def test_validate_scalar_claim_incorrect(self, diagonal_noise_3d):
        """Test validation catches incorrect scalar claim."""
        char = NoiseCharacterizer(
            diagonal_noise_3d["diffusion"],
            diagonal_noise_3d["state_vars"],
            diagonal_noise_3d["control_vars"],
        )

        with pytest.raises(ValueError, match="Claimed noise_type='scalar'"):
            char.validate_noise_type_claim("scalar")


# ============================================================================
# Test Optimization Hints
# ============================================================================


class TestOptimizationHints:
    """Test optimization hint generation."""

    def test_optimization_hints_additive(self, additive_noise_2d):
        """Test optimization hints for additive noise."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        hints = char.get_optimization_hints()

        assert hints["can_precompute_diffusion"]
        assert hints["complexity"] == "O(1) - constant, precomputable"

    def test_optimization_hints_diagonal(self, diagonal_noise_3d):
        """Test optimization hints for diagonal noise."""
        char = NoiseCharacterizer(
            diagonal_noise_3d["diffusion"],
            diagonal_noise_3d["state_vars"],
            diagonal_noise_3d["control_vars"],
        )

        hints = char.get_optimization_hints()

        assert hints["can_use_diagonal_solver"]
        assert not hints["can_precompute_diffusion"]
        assert hints["complexity"] == "O(nx) - element-wise"

    def test_optimization_hints_scalar(self, multiplicative_noise_1d):
        """Test optimization hints for scalar noise."""
        char = NoiseCharacterizer(
            multiplicative_noise_1d["diffusion"],
            multiplicative_noise_1d["state_vars"],
            multiplicative_noise_1d["control_vars"],
        )

        hints = char.get_optimization_hints()

        assert hints["can_use_scalar_solver"]
        assert hints["complexity"] == "O(nx) - scalar multiplication"

    def test_optimization_hints_general(self, general_noise_2d):
        """Test optimization hints for general noise."""
        char = NoiseCharacterizer(
            general_noise_2d["diffusion"],
            general_noise_2d["state_vars"],
            general_noise_2d["control_vars"],
        )

        hints = char.get_optimization_hints()

        assert not hints["can_precompute_diffusion"]
        assert not hints["can_use_diagonal_solver"]
        assert hints["complexity"] == "O(nx * nw) - full matrix"

    def test_backend_recommendations(self, additive_noise_2d):
        """Test backend recommendations."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        hints = char.get_optimization_hints()

        assert "recommended_backends" in hints
        backends = hints["recommended_backends"]
        assert "jax" in backends
        assert backends[0] == "jax"  # JAX should be first for additive


# ============================================================================
# Test Characteristics Property (Lazy Loading)
# ============================================================================


class TestCharacteristicsProperty:
    """Test characteristics property and lazy loading."""

    def test_characteristics_lazy_loading(self, additive_noise_2d):
        """Test that characteristics are computed lazily."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        # Should be None before access
        assert char._characteristics is None

        # Access triggers analysis
        result = char.characteristics

        # Now cached
        assert char._characteristics is not None
        assert result is char._characteristics

    def test_characteristics_cached(self, additive_noise_2d):
        """Test that characteristics are cached after first access."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        result1 = char.characteristics
        result2 = char.characteristics

        # Should be same object
        assert result1 is result2


# ============================================================================
# Test String Representations
# ============================================================================


class TestStringRepresentations:
    """Test string representation methods."""

    def test_repr_before_analysis(self, additive_noise_2d):
        """Test __repr__ before analysis."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        repr_str = repr(char)
        assert "not yet analyzed" in repr_str

    def test_repr_after_analysis(self, additive_noise_2d):
        """Test __repr__ after analysis."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        # Trigger analysis
        _ = char.characteristics

        repr_str = repr(char)
        assert "NoiseCharacterizer" in repr_str
        assert "additive" in repr_str


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_zero_diffusion(self):
        """Test with zero diffusion matrix."""
        x = sp.symbols("x")
        u = sp.symbols("u")
        diffusion = sp.Matrix([[0]])

        char = NoiseCharacterizer(diffusion, [x], [u])
        result = char.analyze()

        # Zero is constant -> additive
        assert result.is_additive
        assert result.noise_type == NoiseType.ADDITIVE

    def test_identity_diffusion(self):
        """Test with identity diffusion matrix."""
        x1, x2 = sp.symbols("x1 x2")
        u = sp.symbols("u")
        diffusion = sp.Matrix([[1, 0], [0, 1]])

        char = NoiseCharacterizer(diffusion, [x1, x2], [u])
        result = char.analyze()

        assert result.is_additive
        assert result.is_diagonal

    def test_single_element_matrix(self):
        """Test with 1x1 diffusion matrix."""
        x = sp.symbols("x")
        u = sp.symbols("u")
        diffusion = sp.Matrix([[0.1]])

        char = NoiseCharacterizer(diffusion, [x], [u])
        result = char.analyze()

        assert result.is_scalar
        assert result.is_additive
        assert result.is_diagonal  # 1x1 is trivially diagonal

    def test_complex_symbolic_expression(self):
        """Test with complex symbolic expressions."""
        x1, x2 = sp.symbols("x1 x2")
        u = sp.symbols("u")

        # Complex expression with trig functions
        diffusion = sp.Matrix([[sp.sin(x1) + sp.cos(x2)]])

        char = NoiseCharacterizer(diffusion, [x1, x2], [u])
        result = char.analyze()

        assert result.is_multiplicative
        assert result.depends_on_state
        assert len(result.state_dependencies) == 2


# ============================================================================
# Test Integration
# ============================================================================


class TestIntegration:
    """Integration tests with complete workflows."""

    def test_full_workflow_additive(self, additive_noise_2d):
        """Test complete workflow for additive noise."""
        char = NoiseCharacterizer(
            additive_noise_2d["diffusion"],
            additive_noise_2d["state_vars"],
            additive_noise_2d["control_vars"],
        )

        # Analyze
        result = char.analyze()

        # Verify classification
        assert result.noise_type == NoiseType.ADDITIVE
        assert result.is_additive
        assert result.is_scalar

        # Get recommendations
        jax_solvers = result.recommended_solvers("jax")
        assert len(jax_solvers) > 0

        # Get hints
        hints = char.get_optimization_hints()
        assert hints["can_precompute_diffusion"]

        # Validate claim
        assert char.validate_noise_type_claim("additive")

    def test_full_workflow_multiplicative(self, diagonal_noise_3d):
        """Test complete workflow for multiplicative diagonal noise."""
        char = NoiseCharacterizer(
            diagonal_noise_3d["diffusion"],
            diagonal_noise_3d["state_vars"],
            diagonal_noise_3d["control_vars"],
        )

        # Access through property (lazy loading)
        result = char.characteristics

        # Verify classification
        assert result.noise_type == NoiseType.DIAGONAL
        assert result.is_diagonal
        assert result.is_multiplicative
        assert not result.is_additive

        # Check dependencies
        assert result.depends_on_state
        assert len(result.state_dependencies) == 3

        # Get optimization hints
        hints = char.get_optimization_hints()
        assert hints["can_use_diagonal_solver"]
        assert not hints["can_precompute_diffusion"]

        # Validate
        assert char.validate_noise_type_claim("diagonal")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
