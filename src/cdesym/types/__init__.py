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
Type System for Control Differential Equation Simulation Library

This package provides comprehensive type definitions for:

1. Core Types (vectors, matrices, dimensions)
   - StateVector, ControlVector, OutputVector
   - StateMatrix, InputMatrix, OutputMatrix, etc.
   - SystemDimensions, ArrayLike

2. Backend Support (NumPy, PyTorch, JAX)
   - Backend type literal
   - BackendConfig TypedDict

3. System Execution Results (organized in system_results module)
   - Integration: IntegrationResult, SDEIntegrationResult
   - Simulation: SimulationResult, SDESimulationResult, DiscreteSimulationResult
   - Rollout: RolloutResult, SDERolloutResult, DiscreteRolloutResult
   - Stochastic variants: DiscreteStochasticSimulationResult, etc.
   - Union types: SystemResult (all results), ContinuousSimulationResultUnion, etc.

4. Trajectories and Time Series
   - TimePoints, StateTrajectory, ControlSequence
   - TrajectoryStatistics, TrajectorySegment

5. Linearization
   - LinearizationResult, DeterministicLinearization
   - StochasticLinearization

6. Control Design
   - Classical: LQRResult, LQGResult, KalmanFilterResult
   - Advanced: MPCResult, MHEResult, H2ControlResult, HInfControlResult

7. State Estimation
   - EKFResult, UKFResult, ParticleFilterResult

8. System Identification
   - SystemIDResult, SubspaceIDResult
   - ERAResult, DMDResult, SINDyResult

9. Symbolic Computation
   - SymbolicExpression, SymbolicMatrix
   - SymbolicStateEquations, SymbolicJacobian

10. Reachability and Safety
    - ReachabilityResult, ROAResult
    - CBFResult, CLFResult, BarrierCertificateResult

11. Robustness Analysis
    - RobustStabilityResult, TubeMPCResult
    - StochasticMPCResult, RiskSensitiveResult

12. Optimization
    - OptimizationResult, TrajectoryOptimizationResult
    - ConvexOptimizationResult

13. Machine Learning
    - TrainingResult, RLTrainingResult
    - ImitationLearningResult

14. Conformal Prediction
    - ConformalPredictionResult, AdaptiveConformalResult

15. Contraction Theory
    - ContractionAnalysisResult, CCMResult
    - FunnelingResult

16. Differential Flatness
    - DifferentialFlatnessResult, TrajectoryPlanningResult

17. Model Reduction
    - BalancedRealizationResult, ReducedOrderModelResult

18. Utilities
    - Type guards, converters, validators
    - Protocols for structural subtyping

Usage
-----
>>> from cdesym.types import (
...     StateVector, ControlVector, StateMatrix,
...     LQRResult, MPCResult, SimulationResult, SystemResult,
... )
>>>
>>> # Type-annotated function
>>> def simulate(x0: StateVector, u: ControlVector) -> SimulationResult:
...     ...
>>>
>>> # Polymorphic function using union type
>>> def plot_trajectory(result: SystemResult) -> None:
...     plt.plot(result['t'], result['x'])

Import Patterns
--------------
# Import specific types
from cdesym.types import StateVector, LQRResult, SystemResult

# Import from specific modules
from cdesym.types.core import StateVector, SystemDimensions
from cdesym.types.control_classical import LQRResult
from cdesym.types.system_results import IntegrationResult, SystemResult

# Import entire module
from cdesym.types import core, estimation, system_results
"""

# =============================================================================
# Core Types - Fundamental vectors and matrices
# =============================================================================
from .core import (
    ArrayLike,
    ControlInput,
    ControlPolicy,
    ControlVector,
    CovarianceMatrix,
    DiffusionFunction,
    DiffusionMatrix,
    DynamicsFunction,
    FeedbackController,
    FeedthroughMatrix,
    GainMatrix,
    InputMatrix,
    NoiseVector,
    OutputFunction,
    OutputMatrix,
    OutputVector,
    ParameterVector,
    ScalarLike,
    StateMatrix,
    StateVector,
    SystemDimensions,
    TimeVaryingControl,
)

# =============================================================================
# Backend Support - Multi-backend array operations
# =============================================================================
from .backends import (
    Backend,
    BackendConfig,
    ConvergenceType,
    Device,
    DiscretizationMethod,
    DiscretizerConfig,
    IntegrationMethod,
    IntegratorConfig,
    NoiseType,
    OptimizationMethod,
    SDEIntegrationMethod,
    SDEIntegratorConfig,
    SDEType,
    SystemConfig,
    get_backend_default_method,
    validate_backend,
    validate_device,
)

# =============================================================================
# System Execution Results - Integration, simulation, rollout results
# =============================================================================
from .system_results import (
    # Base types
    DiscreteSimulationResultBase,
    IntegrationResultBase,
    RolloutResultBase,
    SimulationResultBase,
    # Continuous deterministic
    IntegrationResult,
    RolloutResult,
    SimulationResult,
    # Continuous stochastic
    SDEIntegrationResult,
    SDERolloutResult,
    SDESimulationResult,
    # Discrete deterministic
    DiscreteRolloutResult,
    DiscreteSimulationResult,
    # Discrete stochastic
    DiscreteStochasticRolloutResult,
    DiscreteStochasticSimulationResult,
    # Union types for polymorphic code
    ContinuousIntegrationResultUnion,
    ContinuousRolloutResultUnion,
    ContinuousSimulationResultUnion,
    DiscreteRolloutResultUnion,
    DiscreteSimulationResultUnion,
    SystemResult,
)

# =============================================================================
# Trajectories - Time series data types
# =============================================================================
from .trajectories import (
    ControlSequence,
    NoiseSequence,
    OutputSequence,
    StateTrajectory,
    TimePoints,
    TimeSpan,
    TrajectorySegment,
    TrajectoryStatistics,
)

# =============================================================================
# Linearization - Jacobian matrices and linearization results
# =============================================================================
from .linearization import (
    ContinuousLinearization,
    ContinuousStochasticLinearization,
    ControlJacobian,
    DeterministicLinearization,
    DiffusionJacobian,
    DiscreteLinearization,
    DiscreteStochasticLinearization,
    FullLinearization,
    FullStochasticLinearization,
    LinearizationCacheKey,
    LinearizationResult,
    ObservationLinearization,
    OutputJacobian,
    StateJacobian,
    StochasticLinearization,
)

# =============================================================================
# Classical Control - LQR, LQG, Kalman, pole placement
# =============================================================================
from .control_classical import (
    ControllabilityInfo,
    KalmanFilterResult,
    LQGResult,
    LQRResult,
    LuenbergerObserverResult,
    ObservabilityInfo,
    PolePlacementResult,
    StabilityInfo,
)

# =============================================================================
# Advanced Control - MPC, MHE, H2, H-infinity
# =============================================================================
from .control_advanced import (
    AdaptiveControlResult,
    H2ControlResult,
    HInfControlResult,
    LMIResult,
    MHEResult,
    MPCResult,
    SlidingModeResult,
)

# =============================================================================
# State Estimation - EKF, UKF, Particle filters
# =============================================================================
from .estimation import (
    EKFResult,
    ParticleFilterResult,
    UKFResult,
)

# =============================================================================
# System Identification - Data-driven model estimation
# =============================================================================
from .identification import (
    DMDResult,
    ERAResult,
    HankelMatrix,
    KoopmanResult,
    MarkovParameters,
    SINDyResult,
    SubspaceIDResult,
    SystemIDResult,
    ToeplitzMatrix,
    TrajectoryMatrix,
)

# =============================================================================
# Symbolic Computation - SymPy types and operations
# =============================================================================
from .symbolic import (
    ParameterDict,
    SubstitutionDict,
    SymbolicDiffusionMatrix,
    SymbolicExpression,
    SymbolicGradient,
    SymbolicHessian,
    SymbolicJacobian,
    SymbolicMatrix,
    SymbolicOutputEquations,
    SymbolicStateEquations,
    SymbolicSymbol,
    SymbolDict,
)

# =============================================================================
# Reachability and Safety - Reachable sets and barrier functions
# =============================================================================
from .reachability import (
    BarrierCertificateResult,
    CBFResult,
    CLFResult,
    ReachabilityResult,
    ReachableSet,
    ROAResult,
    SafeSet,
    VerificationResult,
)

# =============================================================================
# Robustness - Robust control and uncertainty quantification
# =============================================================================
from .robustness import (
    RiskSensitiveResult,
    RobustStabilityResult,
    StochasticMPCResult,
    StructuredSingularValueResult,
    TubeDefinition,
    TubeMPCResult,
    UncertaintySet,
)

# =============================================================================
# Optimization - Trajectory and parameter optimization
# =============================================================================
from .optimization import (
    ConstrainedOptimizationResult,
    ConvexOptimizationResult,
    OptimizationBounds,
    OptimizationResult,
    ParameterOptimizationResult,
    TrajectoryOptimizationResult,
)

# =============================================================================
# Learning - Neural networks and reinforcement learning
# =============================================================================
from .learning import (
    Dataset,
    ImitationLearningResult,
    LearningRate,
    LossValue,
    NeuralDynamicsResult,
    NeuralNetworkConfig,
    OnlineAdaptationResult,
    PolicyEvaluationResult,
    RLTrainingResult,
    TrainingBatch,
    TrainingResult,
)

# =============================================================================
# Conformal Prediction - Distribution-free uncertainty quantification
# =============================================================================
from .conformal import (
    AdaptiveConformalResult,
    ConformalCalibrationResult,
    ConformalPredictionResult,
    ConformalPredictionSet,
    NonconformityScore,
)

# =============================================================================
# Contraction Theory - Contraction analysis and CCM
# =============================================================================
from .contraction import (
    CCMResult,
    ContractionAnalysisResult,
    ContractionMetric,
    ContractionRate,
    FunnelingResult,
    IncrementalStabilityResult,
)

# =============================================================================
# Differential Flatness - Flatness analysis and trajectory planning
# =============================================================================
from .flatness import (
    DifferentialFlatnessResult,
    FlatnessOutput,
    TrajectoryPlanningResult,
)

# =============================================================================
# Model Reduction - Balanced truncation and model order reduction
# =============================================================================
from .model_reduction import (
    BalancedRealizationResult,
    ReducedOrderModelResult,
)

# =============================================================================
# System Protocols - Structural subtyping for systems
# =============================================================================
from .protocols import (
    CompilableSystemProtocol,
    ContinuousSystemProtocol,
    DiscreteSystemProtocol,
    LinearizableContinuousProtocol,
    LinearizableDiscreteProtocol,
    ParametricSystemProtocol,
    StochasticSystemProtocol,
    SymbolicContinuousProtocol,
    SymbolicDiscreteProtocol,
)

# =============================================================================
# Utilities - Type guards, converters, validators
# =============================================================================
from .utilities import (
    ArrayConverter,
    CacheKey,
    CacheStatistics,
    ExecutionStats,
    LinearizableProtocol,
    Metadata,
    PerformanceMetrics,
    SimulatableProtocol,
    StochasticProtocol,
    SymbolicValidationResult,
    ValidationResult,
    check_control_shape,
    check_state_shape,
    ensure_backend,
    ensure_numpy,
    extract_dimensions,
    get_array_shape,
    get_backend,
    get_batch_size,
    is_batched,
    is_jax,
    is_numpy,
    is_torch,
)

# =============================================================================
# Public API - Exported types
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Core Types
    # -------------------------------------------------------------------------
    "ArrayLike",
    "ScalarLike",
    # Vector types
    "StateVector",
    "ControlVector",
    "OutputVector",
    "NoiseVector",
    "ParameterVector",
    # Matrix types
    "StateMatrix",
    "InputMatrix",
    "OutputMatrix",
    "DiffusionMatrix",
    "FeedthroughMatrix",
    "GainMatrix",
    "CovarianceMatrix",
    "SystemDimensions",
    # Function types
    "DynamicsFunction",
    "OutputFunction",
    "DiffusionFunction",
    "ControlPolicy",
    "TimeVaryingControl",
    "FeedbackController",
    "ControlInput",
    # -------------------------------------------------------------------------
    # Backend Support
    # -------------------------------------------------------------------------
    "Backend",
    "Device",
    "BackendConfig",
    "IntegrationMethod",
    "DiscretizationMethod",
    "SDEIntegrationMethod",
    "OptimizationMethod",
    "NoiseType",
    "SDEType",
    "ConvergenceType",
    "SystemConfig",
    "IntegratorConfig",
    "DiscretizerConfig",
    "SDEIntegratorConfig",
    "get_backend_default_method",
    "validate_backend",
    "validate_device",
    # -------------------------------------------------------------------------
    # System Execution Results
    # -------------------------------------------------------------------------
    # Base types
    "IntegrationResultBase",
    "SimulationResultBase",
    "RolloutResultBase",
    "DiscreteSimulationResultBase",
    # Continuous deterministic
    "IntegrationResult",
    "SimulationResult",
    "RolloutResult",
    # Continuous stochastic
    "SDEIntegrationResult",
    "SDESimulationResult",
    "SDERolloutResult",
    # Discrete deterministic
    "DiscreteSimulationResult",
    "DiscreteRolloutResult",
    # Discrete stochastic
    "DiscreteStochasticSimulationResult",
    "DiscreteStochasticRolloutResult",
    # Union types
    "ContinuousIntegrationResultUnion",
    "ContinuousSimulationResultUnion",
    "ContinuousRolloutResultUnion",
    "DiscreteSimulationResultUnion",
    "DiscreteRolloutResultUnion",
    "SystemResult",
    # -------------------------------------------------------------------------
    # Trajectories and Time Series
    # -------------------------------------------------------------------------
    "TimePoints",
    "TimeSpan",
    "StateTrajectory",
    "ControlSequence",
    "OutputSequence",
    "NoiseSequence",
    "TrajectoryStatistics",
    "TrajectorySegment",
    # -------------------------------------------------------------------------
    # Linearization
    # -------------------------------------------------------------------------
    "DeterministicLinearization",
    "StochasticLinearization",
    "LinearizationResult",
    "ObservationLinearization",
    "ContinuousLinearization",
    "DiscreteLinearization",
    "ContinuousStochasticLinearization",
    "DiscreteStochasticLinearization",
    "FullLinearization",
    "FullStochasticLinearization",
    "StateJacobian",
    "ControlJacobian",
    "OutputJacobian",
    "DiffusionJacobian",
    "LinearizationCacheKey",
    # -------------------------------------------------------------------------
    # Classical Control
    # -------------------------------------------------------------------------
    "StabilityInfo",
    "ControllabilityInfo",
    "ObservabilityInfo",
    "LQRResult",
    "KalmanFilterResult",
    "LQGResult",
    "PolePlacementResult",
    "LuenbergerObserverResult",
    # -------------------------------------------------------------------------
    # Advanced Control
    # -------------------------------------------------------------------------
    "MPCResult",
    "MHEResult",
    "H2ControlResult",
    "HInfControlResult",
    "LMIResult",
    "AdaptiveControlResult",
    "SlidingModeResult",
    # -------------------------------------------------------------------------
    # State Estimation
    # -------------------------------------------------------------------------
    "EKFResult",
    "UKFResult",
    "ParticleFilterResult",
    # -------------------------------------------------------------------------
    # System Identification
    # -------------------------------------------------------------------------
    "HankelMatrix",
    "ToeplitzMatrix",
    "TrajectoryMatrix",
    "MarkovParameters",
    "SystemIDResult",
    "SubspaceIDResult",
    "ERAResult",
    "DMDResult",
    "SINDyResult",
    "KoopmanResult",
    # -------------------------------------------------------------------------
    # Symbolic Computation
    # -------------------------------------------------------------------------
    "SymbolicExpression",
    "SymbolicMatrix",
    "SymbolicSymbol",
    "SymbolDict",
    "SymbolicStateEquations",
    "SymbolicOutputEquations",
    "SymbolicDiffusionMatrix",
    "ParameterDict",
    "SubstitutionDict",
    "SymbolicJacobian",
    "SymbolicGradient",
    "SymbolicHessian",
    # -------------------------------------------------------------------------
    # Reachability and Safety
    # -------------------------------------------------------------------------
    "ReachableSet",
    "SafeSet",
    "ReachabilityResult",
    "ROAResult",
    "VerificationResult",
    "BarrierCertificateResult",
    "CBFResult",
    "CLFResult",
    # -------------------------------------------------------------------------
    # Robustness
    # -------------------------------------------------------------------------
    "UncertaintySet",
    "RobustStabilityResult",
    "StructuredSingularValueResult",
    "TubeDefinition",
    "TubeMPCResult",
    "StochasticMPCResult",
    "RiskSensitiveResult",
    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------
    "OptimizationBounds",
    "OptimizationResult",
    "ConstrainedOptimizationResult",
    "TrajectoryOptimizationResult",
    "ConvexOptimizationResult",
    "ParameterOptimizationResult",
    # -------------------------------------------------------------------------
    # Learning
    # -------------------------------------------------------------------------
    "Dataset",
    "TrainingBatch",
    "LearningRate",
    "LossValue",
    "NeuralNetworkConfig",
    "TrainingResult",
    "NeuralDynamicsResult",
    "RLTrainingResult",
    "PolicyEvaluationResult",
    "ImitationLearningResult",
    "OnlineAdaptationResult",
    # -------------------------------------------------------------------------
    # Conformal Prediction
    # -------------------------------------------------------------------------
    "ConformalPredictionSet",
    "NonconformityScore",
    "ConformalCalibrationResult",
    "ConformalPredictionResult",
    "AdaptiveConformalResult",
    # -------------------------------------------------------------------------
    # Contraction Theory
    # -------------------------------------------------------------------------
    "ContractionMetric",
    "ContractionRate",
    "ContractionAnalysisResult",
    "CCMResult",
    "FunnelingResult",
    "IncrementalStabilityResult",
    # -------------------------------------------------------------------------
    # Differential Flatness
    # -------------------------------------------------------------------------
    "FlatnessOutput",
    "DifferentialFlatnessResult",
    "TrajectoryPlanningResult",
    # -------------------------------------------------------------------------
    # Model Reduction
    # -------------------------------------------------------------------------
    "BalancedRealizationResult",
    "ReducedOrderModelResult",
    # -------------------------------------------------------------------------
    # System Protocols
    # -------------------------------------------------------------------------
    "DiscreteSystemProtocol",
    "LinearizableDiscreteProtocol",
    "SymbolicDiscreteProtocol",
    "ContinuousSystemProtocol",
    "LinearizableContinuousProtocol",
    "SymbolicContinuousProtocol",
    "StochasticSystemProtocol",
    "CompilableSystemProtocol",
    "ParametricSystemProtocol",
    # -------------------------------------------------------------------------
    # Utilities - Protocols
    # -------------------------------------------------------------------------
    "LinearizableProtocol",
    "SimulatableProtocol",
    "StochasticProtocol",
    # -------------------------------------------------------------------------
    # Utilities - Type Guards
    # -------------------------------------------------------------------------
    "is_batched",
    "get_batch_size",
    "is_numpy",
    "is_torch",
    "is_jax",
    "get_backend",
    # -------------------------------------------------------------------------
    # Utilities - Converters
    # -------------------------------------------------------------------------
    "ensure_numpy",
    "ensure_backend",
    "ArrayConverter",
    # -------------------------------------------------------------------------
    # Utilities - Validators
    # -------------------------------------------------------------------------
    "check_state_shape",
    "check_control_shape",
    "get_array_shape",
    "extract_dimensions",
    # -------------------------------------------------------------------------
    # Utilities - Cache and Metadata
    # -------------------------------------------------------------------------
    "CacheKey",
    "CacheStatistics",
    "Metadata",
    # -------------------------------------------------------------------------
    # Utilities - Validation and Performance
    # -------------------------------------------------------------------------
    "ExecutionStats",
    "ValidationResult",
    "SymbolicValidationResult",
    "PerformanceMetrics",
]
