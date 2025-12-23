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
SDE Integrator Factory - Unified Interface for Creating SDE Integrators

Provides a convenient factory class for creating the appropriate SDE integrator
based on backend, method, and requirements. Simplifies integrator selection
and configuration for stochastic differential equations.

Supports Julia's DifferentialEquations.jl (NumPy), Diffrax (JAX), and TorchSDE (PyTorch).
All integrators support both controlled and autonomous systems (nu=0).

Examples
--------
>>> # Automatic selection
>>> integrator = SDEIntegratorFactory.create(sde_system, backend='jax')
>>>
>>> # Specific method
>>> integrator = SDEIntegratorFactory.create(
...     sde_system, backend='numpy', method='SRIW1'
... )
>>>
>>> # Autonomous system
>>> integrator = SDEIntegratorFactory.create(autonomous_sde_system, backend='torch')
>>> result = integrator.integrate(
...     x0=torch.tensor([1.0, 0.0]),
...     u_func=lambda t, x: None,  # No control
...     t_span=(0.0, 10.0)
... )
>>>
>>> # Quick helpers
>>> integrator = SDEIntegratorFactory.auto(sde_system)  # Best for system
>>> integrator = SDEIntegratorFactory.for_optimization(sde_system)  # Best for gradients
>>> integrator = SDEIntegratorFactory.for_julia(sde_system)  # Best Julia solver
"""

from typing import Optional, Dict, Any, List, TYPE_CHECKING
from enum import Enum

from src.systems.base.numerical_integration.integrator_base import StepMode
from src.systems.base.numerical_integration.stochastic.sde_integrator_base import (
    SDEIntegratorBase,
    ConvergenceType,
    SDEType
)

if TYPE_CHECKING:
    from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem


class SDEIntegratorType(Enum):
    """
    SDE integrator type categories.
    
    Used for automatic selection based on use case.
    """
    PRODUCTION = "production"      # Julia DiffEqPy (SRIW1, EM)
    OPTIMIZATION = "optimization"  # Diffrax (Euler, SEA)
    NEURAL_SDE = "neural_sde"     # TorchSDE (euler with adjoint)
    JULIA = "julia"               # DiffEqPy (specialized algorithms)
    MONTE_CARLO = "monte_carlo"   # High-order weak methods
    SIMPLE = "simple"             # Euler-Maruyama


class SDEIntegratorFactory:
    """
    Factory for creating SDE numerical integrators.
    
    Provides convenient methods for creating SDE integrators based on:
    - Backend (numpy, torch, jax)
    - Method (EM, SRIW1, euler, etc.)
    - Use case (production, optimization, neural SDE, Julia)
    - Noise structure (additive, diagonal, general)
    
    Supports:
    - DiffEqPy (numpy): EM, SRIW1, SRA1, ImplicitEM, etc. (Julia solvers)
    - TorchSDE (torch): euler, milstein, srk, etc.
    - Diffrax (jax): Euler, ItoMilstein, SEA, SHARK, etc.
    
    All integrators support autonomous systems (nu=0) by passing u=None.
    
    Examples
    --------
    >>> # Create integrator by backend and method
    >>> integrator = SDEIntegratorFactory.create(
    ...     sde_system,
    ...     backend='numpy',
    ...     method='EM'
    ... )
    >>>
    >>> # Julia solver for high accuracy
    >>> integrator = SDEIntegratorFactory.create(
    ...     sde_system,
    ...     backend='numpy',
    ...     method='SRIW1'
    ... )
    >>>
    >>> # JAX for optimization
    >>> integrator = SDEIntegratorFactory.create(
    ...     sde_system,
    ...     backend='jax',
    ...     method='Euler'
    ... )
    >>>
    >>> # Automatic selection
    >>> integrator = SDEIntegratorFactory.auto(sde_system)
    >>>
    >>> # Use case-specific
    >>> integrator = SDEIntegratorFactory.for_optimization(sde_system)
    >>> integrator = SDEIntegratorFactory.for_neural_sde(neural_sde)
    >>> integrator = SDEIntegratorFactory.for_julia(sde_system, algorithm='SRA1')
    """
    
    # Default methods for each backend
    _BACKEND_DEFAULTS = {
        "numpy": "EM",      # Julia Euler-Maruyama
        "torch": "euler",   # TorchSDE euler
        "jax": "Euler",     # Diffrax Euler
    }
    
    # Method to backend mapping
    _METHOD_TO_BACKEND = {
        # Julia DiffEqPy methods (numpy only)
        # Euler-Maruyama family
        "EM": "numpy",
        "LambaEM": "numpy",
        "EulerHeun": "numpy",
        
        # Stochastic Runge-Kutta
        "SRIW1": "numpy",
        "SRIW2": "numpy",
        "SOSRI": "numpy",
        "SOSRI2": "numpy",
        "SRA": "numpy",
        "SRA1": "numpy",
        "SRA2": "numpy",
        "SRA3": "numpy",
        "SOSRA": "numpy",
        "SOSRA2": "numpy",
        
        # Milstein family
        "RKMil": "numpy",
        "RKMilCommute": "numpy",
        "RKMilGeneral": "numpy",
        
        # Implicit methods
        "ImplicitEM": "numpy",
        "ImplicitEulerHeun": "numpy",
        "ImplicitRKMil": "numpy",
        
        # IMEX methods
        "SKenCarp": "numpy",
        
        # Adaptive
        "AutoEM": "numpy",
        
        # Optimized
        "SRI": "numpy",
        "SRIW1Optimized": "numpy",
        "SRIW2Optimized": "numpy",
        
        # TorchSDE methods (torch only)
        "euler": "torch",
        "milstein": "torch",
        "srk": "torch",
        "midpoint": "torch",
        "reversible_heun": "torch",
        "adaptive_heun": "torch",
        
        # Diffrax methods (jax only)
        "Euler": "jax",
        "EulerHeun": "jax",
        "Heun": "jax",
        "ItoMilstein": "jax",
        "StratonovichMilstein": "jax",
        "SEA": "jax",
        "SHARK": "jax",
        "SRA1": "jax",  # Note: SRA1 exists in both Julia and Diffrax
        "ReversibleHeun": "jax",
    }
    
    @classmethod
    def create(
        cls,
        sde_system: 'StochasticDynamicalSystem',
        backend: str = 'numpy',
        method: Optional[str] = None,
        dt: Optional[float] = 0.01,
        step_mode: StepMode = StepMode.FIXED,
        sde_type: Optional[SDEType] = None,
        convergence_type: ConvergenceType = ConvergenceType.STRONG,
        seed: Optional[int] = None,
        **options
    ) -> SDEIntegratorBase:
        """
        Create an SDE integrator with specified backend and method.
        
        Parameters
        ----------
        sde_system : StochasticDynamicalSystem
            SDE system to integrate (controlled or autonomous)
        backend : str, optional
            Backend: 'numpy', 'torch', 'jax'. Default: 'numpy'
        method : Optional[str]
            Solver method. If None, uses backend default.
            - numpy: 'EM' (Julia Euler-Maruyama)
            - torch: 'euler' (TorchSDE euler)
            - jax: 'Euler' (Diffrax Euler)
        dt : Optional[float]
            Time step (default: 0.01)
        step_mode : StepMode
            FIXED or ADAPTIVE stepping (most SDE solvers use FIXED)
        sde_type : Optional[SDEType]
            SDE interpretation (None = use system's type)
        convergence_type : ConvergenceType
            Strong or weak convergence
        seed : Optional[int]
            Random seed for reproducibility
        **options
            Additional integrator options (rtol, atol, adjoint, etc.)
        
        Returns
        -------
        SDEIntegratorBase
            Configured SDE integrator
        
        Raises
        ------
        ValueError
            If backend/method combination is invalid
        ImportError
            If required package not installed
        
        Examples
        --------
        >>> # Use defaults (Julia EM)
        >>> integrator = SDEIntegratorFactory.create(sde_system)
        >>>
        >>> # Julia high-accuracy solver
        >>> integrator = SDEIntegratorFactory.create(
        ...     sde_system, backend='numpy', method='SRIW1'
        ... )
        >>>
        >>> # JAX for optimization
        >>> integrator = SDEIntegratorFactory.create(
        ...     sde_system, backend='jax', method='Euler', seed=42
        ... )
        >>>
        >>> # PyTorch neural SDE
        >>> integrator = SDEIntegratorFactory.create(
        ...     neural_sde,
        ...     backend='torch',
        ...     method='euler',
        ...     adjoint=True
        ... )
        >>>
        >>> # Autonomous system
        >>> integrator = SDEIntegratorFactory.create(autonomous_sde_system)
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: None,
        ...     t_span=(0.0, 10.0)
        ... )
        """
        # Use default method if not specified
        if method is None:
            method = cls._BACKEND_DEFAULTS.get(backend, "EM")
        
        # Validate backend
        valid_backends = ["numpy", "torch", "jax"]
        if backend not in valid_backends:
            raise ValueError(
                f"Invalid backend '{backend}'. Choose from: {valid_backends}"
            )
        
        # Check if method requires specific backend
        if method in cls._METHOD_TO_BACKEND:
            required_backend = cls._METHOD_TO_BACKEND[method]
            
            if required_backend != backend:
                raise ValueError(
                    f"Method '{method}' requires backend='{required_backend}', "
                    f"got backend='{backend}'"
                )
        
        # Create appropriate integrator
        if backend == "numpy":
            return cls._create_numpy_sde_integrator(
                sde_system, method, dt, step_mode, sde_type, 
                convergence_type, seed, **options
            )
        elif backend == "torch":
            return cls._create_torch_sde_integrator(
                sde_system, method, dt, step_mode, sde_type,
                convergence_type, seed, **options
            )
        elif backend == "jax":
            return cls._create_jax_sde_integrator(
                sde_system, method, dt, step_mode, sde_type,
                convergence_type, seed, **options
            )
    
    @classmethod
    def _create_numpy_sde_integrator(
        cls,
        sde_system,
        method: str,
        dt: Optional[float],
        step_mode: StepMode,
        sde_type: Optional[SDEType],
        convergence_type: ConvergenceType,
        seed: Optional[int],
        **options
    ):
        """Create NumPy-based SDE integrator (Julia DiffEqPy)."""
        
        # All NumPy SDE methods use Julia DiffEqPy
        try:
            from src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator import (
                DiffEqPySDEIntegrator
            )
            
            return DiffEqPySDEIntegrator(
                sde_system,
                dt=dt,
                step_mode=step_mode,
                backend='numpy',
                algorithm=method,
                sde_type=sde_type,
                convergence_type=convergence_type,
                seed=seed,
                **options
            )
        except ImportError:
            raise ImportError(
                f"Julia method '{method}' requires diffeqpy. "
                f"Install Julia + DifferentialEquations.jl + diffeqpy"
            )
    
    @classmethod
    def _create_torch_sde_integrator(
        cls,
        sde_system,
        method: str,
        dt: Optional[float],
        step_mode: StepMode,
        sde_type: Optional[SDEType],
        convergence_type: ConvergenceType,
        seed: Optional[int],
        **options
    ):
        """Create PyTorch-based SDE integrator (TorchSDE)."""
        
        from src.systems.base.numerical_integration.stochastic.torchsde_integrator import (
            TorchSDEIntegrator
        )
        
        return TorchSDEIntegrator(
            sde_system,
            dt=dt,
            step_mode=step_mode,
            backend='torch',
            method=method,
            sde_type=sde_type,
            convergence_type=convergence_type,
            seed=seed,
            **options
        )
    
    @classmethod
    def _create_jax_sde_integrator(
        cls,
        sde_system,
        method: str,
        dt: Optional[float],
        step_mode: StepMode,
        sde_type: Optional[SDEType],
        convergence_type: ConvergenceType,
        seed: Optional[int],
        **options
    ):
        """Create JAX-based SDE integrator (Diffrax)."""
        
        from src.systems.base.numerical_integration.stochastic.diffrax_sde_integrator import (
            DiffraxSDEIntegrator
        )
        
        return DiffraxSDEIntegrator(
            sde_system,
            dt=dt,
            step_mode=step_mode,
            backend='jax',
            solver=method,
            sde_type=sde_type,
            convergence_type=convergence_type,
            seed=seed,
            **options
        )
    
    # ========================================================================
    # Convenience Methods - Use Case-Specific Creation
    # ========================================================================
    
    @classmethod
    def auto(
        cls,
        sde_system: 'StochasticDynamicalSystem',
        prefer_backend: Optional[str] = None,
        **options
    ) -> SDEIntegratorBase:
        """
        Automatically select best SDE integrator for system.
        
        Selection logic:
        1. If JAX available and no preference → Diffrax (fast + gradients)
        2. If PyTorch available and no preference → TorchSDE
        3. Otherwise → Julia DiffEqPy (best accuracy/reliability)
        
        Parameters
        ----------
        sde_system : StochasticDynamicalSystem
            SDE system to integrate (controlled or autonomous)
        prefer_backend : Optional[str]
            Preferred backend if available
        **options
            Additional options
        
        Returns
        -------
        SDEIntegratorBase
            Best available SDE integrator
        
        Examples
        --------
        >>> integrator = SDEIntegratorFactory.auto(sde_system)
        >>> integrator = SDEIntegratorFactory.auto(sde_system, prefer_backend='jax')
        >>>
        >>> # Works with autonomous systems
        >>> integrator = SDEIntegratorFactory.auto(autonomous_sde_system)
        """
        # Check backend availability
        backends_available = []
        
        try:
            import jax
            backends_available.append("jax")
        except ImportError:
            pass
        
        try:
            import torch
            backends_available.append("torch")
        except ImportError:
            pass
        
        backends_available.append("numpy")  # Always available
        
        # Select backend
        if prefer_backend and prefer_backend in backends_available:
            backend = prefer_backend
        elif "jax" in backends_available:
            backend = "jax"  # Prefer JAX (best for optimization)
        elif "torch" in backends_available:
            backend = "torch"
        else:
            backend = "numpy"
        
        return cls.create(sde_system, backend=backend, **options)
    
    @classmethod
    def for_production(
        cls,
        sde_system: 'StochasticDynamicalSystem',
        use_julia: bool = True,
        noise_type: Optional[str] = None,
        **options
    ) -> SDEIntegratorBase:
        """
        Create integrator for production use.
        
        Uses Julia's high-quality SDE solvers (default) or optimized
        alternatives based on noise structure.
        
        Parameters
        ----------
        sde_system : StochasticDynamicalSystem
            SDE system to integrate
        use_julia : bool
            If True, use Julia DiffEqPy. Default: True
        noise_type : Optional[str]
            Noise structure: 'additive', 'diagonal', 'general'
            Auto-detected if None
        **options
            Additional options (rtol, atol, etc.)
        
        Returns
        -------
        SDEIntegratorBase
            Production-grade SDE integrator
        
        Examples
        --------
        >>> # Julia (default, best accuracy)
        >>> integrator = SDEIntegratorFactory.for_production(
        ...     sde_system, rtol=1e-6, atol=1e-8
        ... )
        >>>
        >>> # Optimized for additive noise
        >>> integrator = SDEIntegratorFactory.for_production(
        ...     additive_sde, noise_type='additive'
        ... )
        """
        if use_julia:
            # Detect noise type for algorithm selection
            if noise_type is None:
                if sde_system.is_additive_noise():
                    noise_type = 'additive'
                elif sde_system.is_diagonal_noise():
                    noise_type = 'diagonal'
                else:
                    noise_type = 'general'
            
            # Select Julia algorithm based on noise
            if noise_type == 'additive':
                algorithm = 'SRA3'  # Optimized for additive
            elif noise_type == 'diagonal':
                algorithm = 'SRIW1'  # High accuracy for diagonal
            else:
                algorithm = 'EM'  # Robust general purpose
            
            # Set conservative defaults
            default_options = {
                "rtol": 1e-6,
                "atol": 1e-8,
            }
            default_options.update(options)
            
            return cls.for_julia(sde_system, algorithm=algorithm, **default_options)
        else:
            # Non-Julia fallback
            try:
                import jax
                return cls.create(
                    sde_system,
                    backend='jax',
                    method='Euler',
                    **options
                )
            except ImportError:
                try:
                    import torch
                    return cls.create(
                        sde_system,
                        backend='torch',
                        method='euler',
                        **options
                    )
                except ImportError:
                    raise ImportError(
                        "No SDE backend available. "
                        "Install JAX (diffrax) or PyTorch (torchsde)"
                    )
    
    @classmethod
    def for_julia(
        cls,
        sde_system: 'StochasticDynamicalSystem',
        algorithm: str = 'EM',
        **options
    ) -> SDEIntegratorBase:
        """
        Create Julia DiffEqPy SDE integrator.
        
        Provides access to Julia's extensive SDE solver ecosystem.
        Best for high accuracy, stiff SDEs, or specialized solvers.
        
        Parameters
        ----------
        sde_system : StochasticDynamicalSystem
            SDE system to integrate
        algorithm : str
            Julia algorithm name. Default: 'EM'
            Options: EM, SRIW1, SRA1, SRA3, ImplicitEM, etc.
        **options
            Additional options (rtol, atol, etc.)
        
        Returns
        -------
        SDEIntegratorBase
            DiffEqPy SDE integrator
        
        Raises
        ------
        ImportError
            If Julia/diffeqpy not installed
        
        Examples
        --------
        >>> # Default (Euler-Maruyama)
        >>> integrator = SDEIntegratorFactory.for_julia(sde_system)
        >>>
        >>> # High accuracy diagonal noise
        >>> integrator = SDEIntegratorFactory.for_julia(
        ...     sde_system,
        ...     algorithm='SRIW1',
        ...     rtol=1e-6
        ... )
        >>>
        >>> # Additive noise optimization
        >>> integrator = SDEIntegratorFactory.for_julia(
        ...     additive_sde,
        ...     algorithm='SRA3'
        ... )
        >>>
        >>> # Stiff drift
        >>> integrator = SDEIntegratorFactory.for_julia(
        ...     stiff_sde,
        ...     algorithm='ImplicitEM'
        ... )
        """
        try:
            from src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator import (
                DiffEqPySDEIntegrator
            )
        except ImportError:
            raise ImportError(
                "Julia SDE integration requires diffeqpy.\n\n"
                "Installation:\n"
                "1. Install Julia from https://julialang.org/downloads/\n"
                "2. Install DifferentialEquations.jl:\n"
                "   julia> using Pkg\n"
                "   julia> Pkg.add('DifferentialEquations')\n"
                "3. Install Python package:\n"
                "   pip install diffeqpy"
            )
        
        return DiffEqPySDEIntegrator(
            sde_system,
            backend='numpy',
            algorithm=algorithm,
            **options
        )
    
    @classmethod
    def for_optimization(
        cls,
        sde_system: 'StochasticDynamicalSystem',
        prefer_backend: str = 'jax',
        **options
    ) -> SDEIntegratorBase:
        """
        Create SDE integrator for optimization/parameter estimation.
        
        Prioritizes gradient computation and JIT compilation.
        
        Parameters
        ----------
        sde_system : StochasticDynamicalSystem
            SDE system to integrate
        prefer_backend : str
            Preferred backend ('jax' or 'torch')
        **options
            Additional options
        
        Returns
        -------
        SDEIntegratorBase
            SDE integrator with gradient support
        
        Examples
        --------
        >>> integrator = SDEIntegratorFactory.for_optimization(sde_system)
        >>>
        >>> # PyTorch optimization
        >>> integrator = SDEIntegratorFactory.for_optimization(
        ...     sde_system, prefer_backend='torch'
        ... )
        """
        # Try preferred backend first
        try:
            if prefer_backend == 'jax':
                import jax
                from src.systems.base.numerical_integration.stochastic.diffrax_sde_integrator import (
                    DiffraxSDEIntegrator
                )
                
                # Choose solver based on noise type
                if sde_system.is_additive_noise():
                    solver = 'SEA'  # Optimized for additive
                else:
                    solver = 'Euler'
                
                return DiffraxSDEIntegrator(
                    sde_system,
                    dt=options.pop('dt', 0.01),
                    step_mode=options.pop('step_mode', StepMode.FIXED),
                    backend='jax',
                    solver=solver,
                    adjoint=options.pop('adjoint', 'recursive_checkpoint'),
                    seed=options.pop('seed', None),
                    **options
                )
            elif prefer_backend == 'torch':
                import torch
                from src.systems.base.numerical_integration.stochastic.torchsde_integrator import (
                    TorchSDEIntegrator
                )
                
                return TorchSDEIntegrator(
                    sde_system,
                    dt=options.pop('dt', 0.01),
                    step_mode=options.pop('step_mode', StepMode.FIXED),
                    backend='torch',
                    method='euler',
                    adjoint=options.pop('adjoint', False),  # User can enable
                    seed=options.pop('seed', None),
                    **options
                )
        except ImportError:
            pass
        
        # Fallback: try JAX, then torch, then Julia
        try:
            import jax
            from src.systems.base.numerical_integration.stochastic.diffrax_sde_integrator import (
                DiffraxSDEIntegrator
            )
            
            return DiffraxSDEIntegrator(
                sde_system,
                dt=options.pop('dt', 0.01),
                step_mode=StepMode.FIXED,
                backend='jax',
                solver='Euler',
                seed=options.pop('seed', None),
                **options
            )
        except ImportError:
            pass
        
        try:
            import torch
            from src.systems.base.numerical_integration.stochastic.torchsde_integrator import (
                TorchSDEIntegrator
            )
            
            return TorchSDEIntegrator(
                sde_system,
                dt=options.pop('dt', 0.01),
                step_mode=StepMode.FIXED,
                backend='torch',
                method='euler',
                seed=options.pop('seed', None),
                **options
            )
        except ImportError:
            pass
        
        # Last resort: Julia (no gradients)
        from src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator import (
            DiffEqPySDEIntegrator
        )
        
        return DiffEqPySDEIntegrator(
            sde_system,
            backend='numpy',
            algorithm='EM',
            dt=options.pop('dt', 0.01),
            seed=options.pop('seed', None),
            **options
        )
    
    @classmethod
    def for_neural_sde(
        cls,
        neural_sde_system,
        backend: str = 'torch',
        **options
    ) -> SDEIntegratorBase:
        """
        Create integrator for Neural SDE training.
        
        Uses PyTorch with adjoint method for memory efficiency.
        
        Parameters
        ----------
        neural_sde_system : StochasticDynamicalSystem
            Neural network defining SDE dynamics
        backend : str
            'torch' (recommended) or 'jax'
        **options
            Additional options
        
        Returns
        -------
        SDEIntegratorBase
            SDE integrator with adjoint method
        
        Raises
        ------
        ImportError
            If PyTorch/JAX not installed
        
        Examples
        --------
        >>> class NeuralSDE(nn.Module):
        ...     def drift(self, x, u):
        ...         return self.drift_net(x)
        ...     def diffusion(self, x, u):
        ...         return self.diffusion_net(x)
        >>>
        >>> neural_sde = NeuralSDE()
        >>> integrator = SDEIntegratorFactory.for_neural_sde(neural_sde)
        """
        if backend == 'torch':
            try:
                from src.systems.base.numerical_integration.stochastic.torchsde_integrator import (
                    TorchSDEIntegrator
                )
            except ImportError:
                raise ImportError(
                    "PyTorch is required for Neural SDE integration. "
                    "Install with: pip install torch torchsde"
                )
            
            return TorchSDEIntegrator(
                neural_sde_system,
                backend='torch',
                method='euler',
                adjoint=True,  # Memory-efficient for neural networks
                **options
            )
        elif backend == 'jax':
            try:
                from src.systems.base.numerical_integration.stochastic.diffrax_sde_integrator import (
                    DiffraxSDEIntegrator
                )
            except ImportError:
                raise ImportError(
                    "JAX is required for Neural SDE integration. "
                    "Install with: pip install jax diffrax"
                )
            
            return DiffraxSDEIntegrator(
                neural_sde_system,
                backend='jax',
                solver='Euler',
                adjoint='recursive_checkpoint',
                **options
            )
        else:
            raise ValueError(
                f"Backend '{backend}' not supported for Neural SDEs. "
                f"Use 'torch' or 'jax'"
            )
    
    @classmethod
    def for_monte_carlo(
        cls,
        sde_system: 'StochasticDynamicalSystem',
        prefer_backend: str = 'numpy',
        **options
    ) -> SDEIntegratorBase:
        """
        Create integrator optimized for Monte Carlo simulations.
        
        Prioritizes weak convergence (moment accuracy) over strong
        convergence (pathwise accuracy).
        
        Parameters
        ----------
        sde_system : StochasticDynamicalSystem
            SDE system to integrate
        prefer_backend : str
            Preferred backend (default: 'numpy' for Julia's weak solvers)
        **options
            Additional options
        
        Returns
        -------
        SDEIntegratorBase
            SDE integrator optimized for weak convergence
        
        Examples
        --------
        >>> integrator = SDEIntegratorFactory.for_monte_carlo(
        ...     sde_system, prefer_backend='numpy'
        ... )
        >>>
        >>> # Run Monte Carlo
        >>> result = integrator.integrate_monte_carlo(
        ...     x0, u_func, t_span, n_paths=10000
        ... )
        >>> stats = result.get_statistics()
        """
        # Select weak convergence algorithm
        if prefer_backend == 'numpy':
            # Julia has best weak convergence algorithms
            if sde_system.is_additive_noise():
                algorithm = 'SRA3'  # Order 2.0 weak for additive
            elif sde_system.is_diagonal_noise():
                algorithm = 'SRA1'  # Order 2.0 weak for diagonal
            else:
                algorithm = 'EM'    # Order 1.0 weak (general)
            
            return cls.for_julia(
                sde_system,
                algorithm=algorithm,
                convergence_type=ConvergenceType.WEAK,
                **options
            )
        elif prefer_backend == 'jax':
            # JAX/Diffrax weak solvers
            if sde_system.is_additive_noise():
                solver = 'SRA1'  # Order 2.0 weak
            else:
                solver = 'Euler'
            
            return cls.create(
                sde_system,
                backend='jax',
                method=solver,
                convergence_type=ConvergenceType.WEAK,
                **options
            )
        else:  # torch
            return cls.create(
                sde_system,
                backend='torch',
                method='euler',
                convergence_type=ConvergenceType.WEAK,
                **options
            )
    
    @classmethod
    def for_simple_simulation(
        cls,
        sde_system: 'StochasticDynamicalSystem',
        dt: float = 0.01,
        backend: str = 'numpy',
        seed: Optional[int] = None,
        **options
    ) -> SDEIntegratorBase:
        """
        Create simple Euler-Maruyama integrator.
        
        Good for prototyping and educational purposes.
        
        Parameters
        ----------
        sde_system : StochasticDynamicalSystem
            SDE system to integrate
        dt : float
            Time step. Default: 0.01
        backend : str
            Backend to use. Default: 'numpy'
        seed : Optional[int]
            Random seed for reproducibility
        **options
            Additional options
        
        Returns
        -------
        SDEIntegratorBase
            Euler-Maruyama SDE integrator
        
        Examples
        --------
        >>> integrator = SDEIntegratorFactory.for_simple_simulation(
        ...     sde_system, dt=0.01, seed=42
        ... )
        """
        if backend == 'numpy':
            method = 'EM'
        elif backend == 'torch':
            method = 'euler'
        elif backend == 'jax':
            method = 'Euler'
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        return cls.create(
            sde_system,
            backend=backend,
            method=method,
            dt=dt,
            seed=seed,
            **options
        )
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    @staticmethod
    def list_methods(backend: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List available SDE methods for each backend.
        
        Returns
        -------
        Dict[str, List[str]]
            SDE methods available for each backend
        
        Examples
        --------
        >>> methods = SDEIntegratorFactory.list_methods()
        >>> print(methods['numpy'][:5])
        ['EM', 'SRIW1', 'SRA1', 'ImplicitEM', ...]
        >>>
        >>> # Just one backend
        >>> numpy_methods = SDEIntegratorFactory.list_methods('numpy')
        """
        all_methods = {
            "numpy": [
                # Julia DiffEqPy methods
                # Euler-Maruyama family
                "EM",
                "LambaEM",
                "EulerHeun",
                # Stochastic RK (high accuracy)
                "SRIW1",
                "SRIW2",
                "SOSRI",
                "SRA1",
                "SRA3",
                "SOSRA",
                # Milstein family
                "RKMil",
                "RKMilCommute",
                # Implicit (for stiff)
                "ImplicitEM",
                "ImplicitRKMil",
                # IMEX
                "SKenCarp",
                # Adaptive
                "AutoEM",
            ],
            "torch": [
                # TorchSDE methods
                "euler",
                "milstein",
                "srk",
                "midpoint",
                "reversible_heun",
                "adaptive_heun",
            ],
            "jax": [
                # Diffrax methods
                "Euler",
                "EulerHeun",
                "Heun",
                "ItoMilstein",
                "StratonovichMilstein",
                "SEA",
                "SHARK",
                "SRA1",
                "ReversibleHeun",
            ],
        }
        
        if backend:
            return {backend: all_methods.get(backend, [])}
        return all_methods
    
    @staticmethod
    def recommend(
        use_case: str,
        noise_type: str = 'general',
        has_jax: bool = False,
        has_torch: bool = False,
        has_gpu: bool = False
    ) -> Dict[str, Any]:
        """
        Get SDE integrator recommendation based on use case.
        
        Parameters
        ----------
        use_case : str
            One of: 'production', 'optimization', 'neural_sde',
            'monte_carlo', 'prototype', 'simple', 'julia'
        noise_type : str
            'additive', 'diagonal', 'general'
        has_jax : bool
            Whether JAX is available
        has_torch : bool
            Whether PyTorch is available
        has_gpu : bool
            Whether GPU is available
        
        Returns
        -------
        Dict[str, Any]
            Recommendation with 'backend', 'method', 'convergence_type'
        
        Examples
        --------
        >>> rec = SDEIntegratorFactory.recommend('production')
        >>> print(rec)
        {'backend': 'numpy', 'method': 'SRIW1', 'convergence_type': 'STRONG'}
        """
        recommendations = {
            "production": {
                "backend": "numpy",
                "method": "SRIW1" if noise_type == 'diagonal' else "EM",
                "convergence_type": ConvergenceType.STRONG,
                "reason": "Julia DiffEqPy - most reliable, high accuracy",
            },
            "optimization": {
                "backend": "jax" if has_jax else "torch" if has_torch else "numpy",
                "method": "Euler" if has_jax else "euler" if has_torch else "EM",
                "convergence_type": ConvergenceType.STRONG,
                "reason": "Gradient support, JIT compilation",
            },
            "neural_sde": {
                "backend": "torch",
                "method": "euler",
                "convergence_type": ConvergenceType.STRONG,
                "adjoint": True,
                "reason": "Memory-efficient backprop through SDE",
            },
            "julia": {
                "backend": "numpy",
                "method": "SRIW1",
                "convergence_type": ConvergenceType.STRONG,
                "reason": "Access to Julia's powerful SDE solver ecosystem",
            },
            "monte_carlo": {
                "backend": "numpy",
                "method": "SRA3" if noise_type == 'additive' else "SRA1",
                "convergence_type": ConvergenceType.WEAK,
                "reason": "Weak convergence optimized for moment accuracy",
            },
            "simple": {
                "backend": "numpy",
                "method": "EM",
                "convergence_type": ConvergenceType.STRONG,
                "dt": 0.01,
                "reason": "Simple, fast, easy to debug",
            },
            "prototype": {
                "backend": "numpy",
                "method": "EM",
                "convergence_type": ConvergenceType.STRONG,
                "dt": 0.01,
                "reason": "Simple, fast, easy to debug",
            },
        }
        
        if use_case not in recommendations:
            raise ValueError(
                f"Unknown use case '{use_case}'. "
                f"Choose from: {list(recommendations.keys())}"
            )
        
        rec = recommendations[use_case].copy()
        
        # Adjust for GPU availability
        if has_gpu and use_case == "optimization":
            if has_torch:
                rec["backend"] = "torch"
                rec["method"] = "euler"
            elif has_jax:
                rec["backend"] = "jax"
                rec["method"] = "Euler"
        
        return rec
    
    @staticmethod
    def get_info(backend: str, method: str) -> Dict[str, Any]:
        """
        Get information about a specific SDE integrator configuration.
        
        Parameters
        ----------
        backend : str
            Backend name
        method : str
            Method name
        
        Returns
        -------
        Dict[str, Any]
            Information about the SDE integrator
        
        Examples
        --------
        >>> info = SDEIntegratorFactory.get_info('jax', 'Euler')
        >>> print(info['description'])
        'Basic Euler-Maruyama, fast and robust'
        """
        # Delegate to integrator-specific info functions
        if backend == 'numpy':
            try:
                from src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator import (
                    DiffEqPySDEIntegrator
                )
                return DiffEqPySDEIntegrator.get_algorithm_info(method)
            except ImportError:
                return {
                    'name': f'Julia: {method}',
                    'description': 'Julia SDE algorithm (diffeqpy not installed)'
                }
        
        elif backend == 'torch':
            from src.systems.base.numerical_integration.stochastic.torchsde_integrator import (
                TorchSDEIntegrator
            )
            return TorchSDEIntegrator.get_method_info(method)
        
        elif backend == 'jax':
            from src.systems.base.numerical_integration.stochastic.diffrax_sde_integrator import (
                DiffraxSDEIntegrator
            )
            return DiffraxSDEIntegrator.get_solver_info(method)
        
        return {
            'name': method,
            'description': 'No information available',
            'backend': backend
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_sde_integrator(
    sde_system: 'StochasticDynamicalSystem',
    backend: str = 'numpy',
    method: Optional[str] = None,
    **options
) -> SDEIntegratorBase:
    """
    Convenience function for creating SDE integrators.
    
    Alias for SDEIntegratorFactory.create().
    
    Examples
    --------
    >>> integrator = create_sde_integrator(sde_system)
    >>> integrator = create_sde_integrator(sde_system, backend='jax', method='Euler')
    >>> integrator = create_sde_integrator(sde_system, backend='numpy', method='SRIW1')
    >>>
    >>> # Autonomous system
    >>> integrator = create_sde_integrator(autonomous_sde_system)
    """
    return SDEIntegratorFactory.create(sde_system, backend, method, **options)


def auto_sde_integrator(
    sde_system: 'StochasticDynamicalSystem',
    **options
) -> SDEIntegratorBase:
    """
    Automatically select best available SDE integrator.
    
    Alias for SDEIntegratorFactory.auto().
    
    Examples
    --------
    >>> integrator = auto_sde_integrator(sde_system, seed=42)
    >>>
    >>> # Autonomous system
    >>> integrator = auto_sde_integrator(autonomous_sde_system)
    """
    return SDEIntegratorFactory.auto(sde_system, **options)