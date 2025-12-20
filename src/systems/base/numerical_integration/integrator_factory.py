"""
Integrator Factory - Unified Interface for Creating Numerical Integrators

Provides a convenient factory class for creating the appropriate integrator
based on backend, method, and requirements. Simplifies integrator selection
and configuration.

Examples
--------
>>> # Automatic selection
>>> integrator = IntegratorFactory.create(system, backend='numpy')
>>> 
>>> # Specific method
>>> integrator = IntegratorFactory.create(
...     system, backend='jax', method='tsit5'
... )
>>> 
>>> # Quick helpers
>>> integrator = IntegratorFactory.auto(system)  # Best for system
>>> integrator = IntegratorFactory.for_optimization(system)  # Best for gradients
"""

from typing import Optional, Dict, Any, TYPE_CHECKING
from enum import Enum

from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    StepMode
)

if TYPE_CHECKING:
    from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem


class IntegratorType(Enum):
    """
    Integrator type categories.
    
    Used for automatic selection based on use case.
    """
    PRODUCTION = "production"      # Scipy (LSODA)
    OPTIMIZATION = "optimization"  # Diffrax (tsit5)
    NEURAL_ODE = "neural_ode"     # TorchDiffEq (dopri5, adjoint)
    SIMPLE = "simple"             # RK4 fixed-step
    EDUCATIONAL = "educational"    # Euler fixed-step


class IntegratorFactory:
    """
    Factory for creating numerical integrators.
    
    Provides convenient methods for creating integrators based on:
    - Backend (numpy, torch, jax)
    - Method (RK45, dopri5, tsit5, etc.)
    - Use case (production, optimization, neural ODE)
    
    Examples
    --------
    >>> # Create integrator by backend and method
    >>> integrator = IntegratorFactory.create(
    ...     system,
    ...     backend='numpy',
    ...     method='LSODA'
    ... )
    >>> 
    >>> # Automatic selection
    >>> integrator = IntegratorFactory.auto(system)
    >>> 
    >>> # Use case-specific
    >>> integrator = IntegratorFactory.for_optimization(system)
    >>> integrator = IntegratorFactory.for_production(system)
    >>> integrator = IntegratorFactory.for_neural_ode(neural_system)
    """
    
    # Default methods for each backend
    _BACKEND_DEFAULTS = {
        'numpy': 'LSODA',
        'torch': 'dopri5',
        'jax': 'tsit5',
    }
    
    # Method to integrator class mapping
    _METHOD_TO_BACKEND = {
        # Scipy methods (numpy only)
        'RK45': 'numpy',
        'RK23': 'numpy',
        'DOP853': 'numpy',
        'Radau': 'numpy',
        'BDF': 'numpy',
        'LSODA': 'numpy',
        
        # TorchDiffEq adaptive methods (torch only)
        'dopri5': 'torch',
        'dopri8': 'torch',
        'bosh3': 'torch',
        'adaptive_heun': 'torch',
        'fehlberg2': 'torch',
        
        # TorchDiffEq fixed-step methods (torch only)
        'explicit_adams': 'torch',
        'implicit_adams': 'torch',
        'fixed_adams': 'torch',
        'scipy_solver': 'torch',
        
        # Diffrax explicit methods (jax only)
        'tsit5': 'jax',
        'dopri5': 'jax',  # Also available in JAX
        'dopri8': 'jax',  # Also available in JAX
        'bosh3': 'jax',   # Also available in JAX
        'heun': 'jax',
        'ralston': 'jax',
        
        # Diffrax implicit methods (jax only, for stiff)
        'implicit_euler': 'jax',
        'kvaerno3': 'jax',
        'kvaerno4': 'jax',
        'kvaerno5': 'jax',
        
        # Diffrax IMEX methods (jax only, for split stiffness)
        'sil3': 'jax',
        'kencarp3': 'jax',
        'kencarp4': 'jax',
        'kencarp5': 'jax',
        
        # Diffrax special methods (jax only)
        'reversible_heun': 'jax',
        'semi_implicit_euler': 'jax',
        'leapfrog_midpoint': 'jax',
        
        # Fixed-step methods (available in all backends via manual implementation)
        'euler': 'any',
        'midpoint': 'any',
        'rk4': 'any',
    }
    
    @classmethod
    def create(
        cls,
        system: 'SymbolicDynamicalSystem',
        backend: str = 'numpy',
        method: Optional[str] = None,
        dt: Optional[float] = None,
        step_mode: StepMode = StepMode.ADAPTIVE,
        **options
    ) -> IntegratorBase:
        """
        Create an integrator with specified backend and method.
        
        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate
        backend : str, optional
            Backend: 'numpy', 'torch', 'jax'. Default: 'numpy'
        method : Optional[str]
            Solver method. If None, uses backend default.
            - numpy: 'LSODA' (auto-stiffness)
            - torch: 'dopri5' (general adaptive)
            - jax: 'tsit5' (general adaptive)
        dt : Optional[float]
            Time step (required for FIXED mode)
        step_mode : StepMode
            FIXED or ADAPTIVE stepping
        **options
            Additional integrator options (rtol, atol, etc.)
        
        Returns
        -------
        IntegratorBase
            Configured integrator
        
        Raises
        ------
        ValueError
            If backend/method combination is invalid
        ImportError
            If required package not installed
        
        Examples
        --------
        >>> # Use defaults
        >>> integrator = IntegratorFactory.create(system)
        >>> 
        >>> # Specify method
        >>> integrator = IntegratorFactory.create(
        ...     system, backend='jax', method='dopri5'
        ... )
        >>> 
        >>> # Fixed-step
        >>> integrator = IntegratorFactory.create(
        ...     system,
        ...     backend='numpy',
        ...     method='rk4',
        ...     dt=0.01,
        ...     step_mode=StepMode.FIXED
        ... )
        """
        # Use default method if not specified
        if method is None:
            method = cls._BACKEND_DEFAULTS.get(backend, 'LSODA')
        
        # Validate backend
        valid_backends = ['numpy', 'torch', 'jax']
        if backend not in valid_backends:
            raise ValueError(
                f"Invalid backend '{backend}'. Choose from: {valid_backends}"
            )
        
        # Check if method requires specific backend
        if method in cls._METHOD_TO_BACKEND:
            required_backend = cls._METHOD_TO_BACKEND[method]
            
            # Handle methods available in multiple backends
            if isinstance(required_backend, list):
                if backend not in required_backend:
                    raise ValueError(
                        f"Method '{method}' is available in backends {required_backend}, "
                        f"but got backend='{backend}'. Choose one of: {required_backend}"
                    )
            elif required_backend != 'any' and required_backend != backend:
                raise ValueError(
                    f"Method '{method}' requires backend='{required_backend}', "
                    f"got backend='{backend}'"
                )
        
        # Create appropriate integrator
        if backend == 'numpy':
            return cls._create_numpy_integrator(
                system, method, dt, step_mode, **options
            )
        elif backend == 'torch':
            return cls._create_torch_integrator(
                system, method, dt, step_mode, **options
            )
        elif backend == 'jax':
            return cls._create_jax_integrator(
                system, method, dt, step_mode, **options
            )
    
    @classmethod
    def _create_numpy_integrator(
        cls,
        system,
        method: str,
        dt: Optional[float],
        step_mode: StepMode,
        **options
    ):
        """Create NumPy-based integrator."""
        # Fixed-step methods
        if method in ['euler', 'midpoint', 'rk4']:
            if dt is None:
                raise ValueError(f"Fixed-step method '{method}' requires dt")
            
            from src.systems.base.numerical_integration.fixed_step_integrators import (
                ExplicitEulerIntegrator,
                MidpointIntegrator,
                RK4Integrator
            )
            
            integrator_map = {
                'euler': ExplicitEulerIntegrator,
                'midpoint': MidpointIntegrator,
                'rk4': RK4Integrator,
            }
            
            integrator_class = integrator_map[method]
            return integrator_class(system, dt=dt, backend='numpy', **options)
        
        # Scipy adaptive methods
        else:
            from src.systems.base.numerical_integration.scipy_integrator import ScipyIntegrator
            
            return ScipyIntegrator(
                system,
                dt=dt,
                method=method,
                backend='numpy',
                **options
            )
    
    @classmethod
    def _create_torch_integrator(
        cls,
        system,
        method: str,
        dt: Optional[float],
        step_mode: StepMode,
        **options
    ):
        """Create PyTorch-based integrator."""
        # Fixed-step methods
        if method in ['euler', 'midpoint', 'rk4']:
            if dt is None:
                raise ValueError(f"Fixed-step method '{method}' requires dt")
            
            from src.systems.base.numerical_integration.fixed_step_integrators import (
                ExplicitEulerIntegrator,
                MidpointIntegrator,
                RK4Integrator
            )
            
            integrator_map = {
                'euler': ExplicitEulerIntegrator,
                'midpoint': MidpointIntegrator,
                'rk4': RK4Integrator,
            }
            
            integrator_class = integrator_map[method]
            return integrator_class(system, dt=dt, backend='torch', **options)
        
        # TorchDiffEq methods
        else:
            from src.systems.base.numerical_integration.torchdiffeq_integrator import (
                TorchDiffEqIntegrator
            )
            
            return TorchDiffEqIntegrator(
                system,
                dt=dt,
                step_mode=step_mode,
                backend='torch',
                method=method,
                **options
            )
    
    @classmethod
    def _create_jax_integrator(
        cls,
        system,
        method: str,
        dt: Optional[float],
        step_mode: StepMode,
        **options
    ):
        """Create JAX-based integrator."""
        # Fixed-step methods
        if method in ['euler', 'midpoint', 'rk4']:
            if dt is None:
                raise ValueError(f"Fixed-step method '{method}' requires dt")
            
            from src.systems.base.numerical_integration.fixed_step_integrators import (
                ExplicitEulerIntegrator,
                MidpointIntegrator,
                RK4Integrator
            )
            
            integrator_map = {
                'euler': ExplicitEulerIntegrator,
                'midpoint': MidpointIntegrator,
                'rk4': RK4Integrator,
            }
            
            integrator_class = integrator_map[method]
            return integrator_class(system, dt=dt, backend='jax', **options)
        
        # Diffrax methods
        else:
            from src.systems.base.numerical_integration.diffrax_integrator import (
                DiffraxIntegrator
            )
            
            return DiffraxIntegrator(
                system,
                dt=dt,
                step_mode=step_mode,
                backend='jax',
                solver=method,
                **options
            )
    
    # ========================================================================
    # Convenience Methods - Use Case-Specific Creation
    # ========================================================================
    
    @classmethod
    def auto(
        cls,
        system: 'SymbolicDynamicalSystem',
        prefer_backend: Optional[str] = None,
        **options
    ) -> IntegratorBase:
        """
        Automatically select best integrator for system.
        
        Selection logic:
        1. If JAX available and no backend preference → Diffrax (fast + accurate)
        2. If PyTorch available and no backend preference → TorchDiffEq
        3. Otherwise → Scipy (always available)
        
        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate
        prefer_backend : Optional[str]
            Preferred backend if available
        **options
            Additional options
        
        Returns
        -------
        IntegratorBase
            Best available integrator
        
        Examples
        --------
        >>> integrator = IntegratorFactory.auto(system)
        >>> integrator = IntegratorFactory.auto(system, prefer_backend='jax')
        """
        # Check backend availability
        backends_available = []
        
        try:
            import jax
            backends_available.append('jax')
        except ImportError:
            pass
        
        try:
            import torch
            backends_available.append('torch')
        except ImportError:
            pass
        
        backends_available.append('numpy')  # Always available
        
        # Select backend
        if prefer_backend and prefer_backend in backends_available:
            backend = prefer_backend
        elif 'jax' in backends_available:
            backend = 'jax'  # Prefer JAX (best for optimization)
        elif 'torch' in backends_available:
            backend = 'torch'
        else:
            backend = 'numpy'
        
        return cls.create(system, backend=backend, **options)
    
    @classmethod
    def for_production(
        cls,
        system: 'SymbolicDynamicalSystem',
        **options
    ) -> IntegratorBase:
        """
        Create integrator for production use.
        
        Uses scipy.LSODA with automatic stiffness detection.
        Most reliable choice for production simulations.
        
        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate
        **options
            Additional options (rtol, atol, etc.)
        
        Returns
        -------
        IntegratorBase
            Scipy integrator with LSODA
        
        Examples
        --------
        >>> integrator = IntegratorFactory.for_production(
        ...     system, rtol=1e-9, atol=1e-11
        ... )
        """
        from src.systems.base.numerical_integration.scipy_integrator import ScipyIntegrator
        
        # Set conservative defaults
        default_options = {
            'rtol': 1e-8,
            'atol': 1e-10,
        }
        default_options.update(options)
        
        return ScipyIntegrator(
            system,
            method='LSODA',  # Auto stiffness detection
            backend='numpy',
            **default_options
        )
    
    @classmethod
    def for_optimization(
        cls,
        system: 'SymbolicDynamicalSystem',
        prefer_backend: str = 'jax',
        **options
    ) -> IntegratorBase:
        """
        Create integrator for optimization/parameter estimation.
        
        Prioritizes gradient computation and JIT compilation.
        
        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate
        prefer_backend : str
            Preferred backend ('jax' or 'torch'). Default: 'jax'
        **options
            Additional options
        
        Returns
        -------
        IntegratorBase
            Integrator with gradient support
        
        Examples
        --------
        >>> integrator = IntegratorFactory.for_optimization(system)
        >>> # Returns DiffraxIntegrator if JAX available
        >>> 
        >>> integrator = IntegratorFactory.for_optimization(
        ...     system, prefer_backend='torch'
        ... )
        """
        # Try preferred backend first
        try:
            if prefer_backend == 'jax':
                import jax
                from src.systems.base.numerical_integration.diffrax_integrator import (
                    DiffraxIntegrator
                )
                return DiffraxIntegrator(
                    system,
                    backend='jax',
                    solver='tsit5',
                    adjoint='recursive_checkpoint',
                    **options
                )
            elif prefer_backend == 'torch':
                import torch
                from src.systems.base.numerical_integration.torchdiffeq_integrator import (
                    TorchDiffEqIntegrator
                )
                return TorchDiffEqIntegrator(
                    system,
                    backend='torch',
                    method='dopri5',
                    adjoint=False,
                    **options
                )
        except ImportError:
            pass
        
        # Fallback: try JAX, then torch, then numpy
        try:
            import jax
            from src.systems.base.numerical_integration.diffrax_integrator import (
                DiffraxIntegrator
            )
            return DiffraxIntegrator(
                system, backend='jax', solver='tsit5', **options
            )
        except ImportError:
            pass
        
        try:
            import torch
            from src.systems.base.numerical_integration.torchdiffeq_integrator import (
                TorchDiffEqIntegrator
            )
            return TorchDiffEqIntegrator(
                system, backend='torch', method='dopri5', **options
            )
        except ImportError:
            pass
        
        # Last resort: scipy (no gradient support)
        from src.systems.base.numerical_integration.scipy_integrator import ScipyIntegrator
        return ScipyIntegrator(system, method='RK45', backend='numpy', **options)
    
    @classmethod
    def for_neural_ode(
        cls,
        neural_system,
        **options
    ) -> IntegratorBase:
        """
        Create integrator for Neural ODE training.
        
        Uses PyTorch with adjoint method for memory efficiency.
        
        Parameters
        ----------
        neural_system : nn.Module
            Neural network defining ODE dynamics
        **options
            Additional options
        
        Returns
        -------
        IntegratorBase
            TorchDiffEq integrator with adjoint method
        
        Raises
        ------
        ImportError
            If PyTorch not installed
        
        Examples
        --------
        >>> class NeuralODE(nn.Module):
        ...     def forward(self, t, x):
        ...         return self.net(x)
        >>> 
        >>> neural_ode = NeuralODE()
        >>> integrator = IntegratorFactory.for_neural_ode(neural_ode)
        """
        try:
            from src.systems.base.numerical_integration.torchdiffeq_integrator import (
                TorchDiffEqIntegrator
            )
        except ImportError:
            raise ImportError(
                "PyTorch is required for Neural ODE integration. "
                "Install with: pip install torch torchdiffeq"
            )
        
        return TorchDiffEqIntegrator(
            neural_system,
            backend='torch',
            method='dopri5',
            adjoint=True,  # Memory-efficient for neural networks
            **options
        )
    
    @classmethod
    def for_simple_simulation(
        cls,
        system: 'SymbolicDynamicalSystem',
        dt: float = 0.01,
        backend: str = 'numpy',
        **options
    ) -> IntegratorBase:
        """
        Create simple fixed-step RK4 integrator.
        
        Good for prototyping and educational purposes.
        
        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate
        dt : float
            Time step. Default: 0.01
        backend : str
            Backend to use. Default: 'numpy'
        **options
            Additional options
        
        Returns
        -------
        IntegratorBase
            RK4 fixed-step integrator
        
        Examples
        --------
        >>> integrator = IntegratorFactory.for_simple_simulation(
        ...     system, dt=0.01
        ... )
        """
        from src.systems.base.numerical_integration.fixed_step_integrators import (
            RK4Integrator
        )
        
        return RK4Integrator(system, dt=dt, backend=backend, **options)
    
    @classmethod
    def for_real_time(
        cls,
        system: 'SymbolicDynamicalSystem',
        dt: float,
        backend: str = 'numpy',
        **options
    ) -> IntegratorBase:
        """
        Create integrator for real-time systems.
        
        Uses fixed-step RK4 for predictable timing.
        
        Parameters
        ----------
        system : SymbolicDynamicalSystem
            System to integrate
        dt : float
            Fixed time step (must match real-time clock)
        backend : str
            Backend to use
        **options
            Additional options
        
        Returns
        -------
        IntegratorBase
            Fixed-step integrator
        
        Examples
        --------
        >>> # Real-time control at 100 Hz
        >>> integrator = IntegratorFactory.for_real_time(
        ...     system, dt=0.01  # 10ms = 100 Hz
        ... )
        """
        from src.systems.base.numerical_integration.fixed_step_integrators import (
            RK4Integrator
        )
        
        return RK4Integrator(
            system,
            dt=dt,
            backend=backend,
            **options
        )
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    @staticmethod
    def list_methods(backend: Optional[str] = None) -> Dict[str, list]:
        """
        List available methods for each backend.
        
        Parameters
        ----------
        backend : Optional[str]
            If specified, only show methods for that backend
        
        Returns
        -------
        Dict[str, list]
            Dictionary mapping backend to list of methods
        
        Examples
        --------
        >>> methods = IntegratorFactory.list_methods()
        >>> print(methods['numpy'])
        ['LSODA', 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF']
        >>> 
        >>> jax_methods = IntegratorFactory.list_methods('jax')
        """
        all_methods = {
            'numpy': [
                # Scipy adaptive methods
                'LSODA',  # Auto-stiffness (recommended) ⭐
                'RK45',   # Default, general purpose
                'RK23',   # Lower accuracy, faster
                'DOP853', # Very high accuracy (8th order)
                'Radau',  # Stiff systems (implicit)
                'BDF',    # Very stiff systems
                # Fixed-step (manual implementations)
                'euler', 'midpoint', 'rk4'
            ],
            'torch': [
                # TorchDiffEq adaptive methods
                'dopri5',         # Default, recommended ⭐
                'dopri8',         # Higher accuracy
                'bosh3',          # Bogacki-Shampine 3(2)
                'adaptive_heun',  # 2nd order adaptive
                'fehlberg2',      # Fehlberg 2(1)
                # TorchDiffEq fixed-step methods
                'euler',          # 1st order
                'midpoint',       # 2nd order
                'rk4',            # 4th order
                'explicit_adams', # Adams-Bashforth (multistep)
                'implicit_adams', # Adams-Moulton (multistep)
                'fixed_adams',    # Fixed-step Adams
                # Scipy wrapper
                'scipy_solver'    # All scipy methods via wrapper
            ],
            'jax': [
                # Diffrax explicit RK methods
                'tsit5',    # Tsitouras 5(4) - recommended ⭐
                'dopri5',   # Dormand-Prince 5(4)
                'dopri8',   # Dormand-Prince 8(7)
                'bosh3',    # Bogacki-Shampine 3(2)
                'euler',    # 1st order
                'heun',     # 2nd order
                'midpoint', # 2nd order
                'ralston',  # 2nd order (optimized coefficients)
                # Diffrax implicit methods (for stiff systems)
                'implicit_euler', # 1st order implicit
                'kvaerno3',       # 3rd order ESDIRK
                'kvaerno4',       # 4th order ESDIRK
                'kvaerno5',       # 5th order ESDIRK
                # Diffrax IMEX methods (split stiffness)
                'sil3',      # 3rd order IMEX
                'kencarp3',  # Kennedy-Carpenter 3rd order
                'kencarp4',  # Kennedy-Carpenter 4th order ⭐
                'kencarp5',  # Kennedy-Carpenter 5th order
                # Diffrax special methods
                'reversible_heun',    # Reversible 2nd order
                'semi_implicit_euler', # Symplectic (Hamiltonian systems)
                'leapfrog_midpoint',  # Linear multistep
            ]
        }
        
        if backend:
            return {backend: all_methods.get(backend, [])}
        return all_methods
    
    @staticmethod
    def recommend(
        use_case: str,
        has_jax: bool = False,
        has_torch: bool = False,
        has_gpu: bool = False
    ) -> Dict[str, Any]:
        """
        Get integrator recommendation based on use case.
        
        Parameters
        ----------
        use_case : str
            One of: 'production', 'optimization', 'neural_ode', 
            'prototype', 'educational', 'real_time'
        has_jax : bool
            Whether JAX is available
        has_torch : bool
            Whether PyTorch is available
        has_gpu : bool
            Whether GPU is available
        
        Returns
        -------
        Dict[str, Any]
            Recommendation with 'backend', 'method', 'step_mode'
        
        Examples
        --------
        >>> rec = IntegratorFactory.recommend('production')
        >>> print(rec)
        {'backend': 'numpy', 'method': 'LSODA', 'step_mode': 'ADAPTIVE'}
        """
        recommendations = {
            'production': {
                'backend': 'numpy',
                'method': 'LSODA',
                'step_mode': StepMode.ADAPTIVE,
                'reason': 'Most reliable, auto-stiffness detection'
            },
            'optimization': {
                'backend': 'jax' if has_jax else 'torch' if has_torch else 'numpy',
                'method': 'tsit5' if has_jax else 'dopri5' if has_torch else 'RK45',
                'step_mode': StepMode.ADAPTIVE,
                'reason': 'Gradient support, JIT compilation'
            },
            'neural_ode': {
                'backend': 'torch',
                'method': 'dopri5',
                'step_mode': StepMode.ADAPTIVE,
                'adjoint': True,
                'reason': 'Memory-efficient backprop through ODE'
            },
            'prototype': {
                'backend': 'numpy',
                'method': 'rk4',
                'step_mode': StepMode.FIXED,
                'dt': 0.01,
                'reason': 'Simple, fast, easy to debug'
            },
            'educational': {
                'backend': 'numpy',
                'method': 'euler',
                'step_mode': StepMode.FIXED,
                'dt': 0.001,
                'reason': 'Easiest to understand'
            },
            'real_time': {
                'backend': 'numpy',
                'method': 'rk4',
                'step_mode': StepMode.FIXED,
                'reason': 'Predictable timing'
            }
        }
        
        if use_case not in recommendations:
            raise ValueError(
                f"Unknown use case '{use_case}'. "
                f"Choose from: {list(recommendations.keys())}"
            )
        
        rec = recommendations[use_case].copy()
        
        # Adjust for GPU availability
        if has_gpu and use_case == 'optimization':
            if has_torch:
                rec['backend'] = 'torch'
                rec['method'] = 'dopri5'
            elif has_jax:
                rec['backend'] = 'jax'
                rec['method'] = 'tsit5'
        
        return rec
    
    @staticmethod
    def get_info(backend: str, method: str) -> Dict[str, Any]:
        """
        Get information about a specific integrator configuration.
        
        Parameters
        ----------
        backend : str
            Backend name
        method : str
            Method name
        
        Returns
        -------
        Dict[str, Any]
            Information about the integrator
        
        Examples
        --------
        >>> info = IntegratorFactory.get_info('jax', 'tsit5')
        >>> print(info['description'])
        'Tsitouras 5(4) - Excellent general purpose solver'
        """
        method_info = {
            # Scipy
            'LSODA': {
                'name': 'LSODA',
                'order': 'Variable (1-12)',
                'type': 'Adaptive',
                'description': 'Auto-detects stiffness, switches Adams↔BDF',
                'best_for': 'Production, unknown stiffness',
                'function_evals_per_step': '1-4'
            },
            'RK45': {
                'name': 'Dormand-Prince 5(4)',
                'order': 5,
                'type': 'Adaptive',
                'description': 'General purpose, robust',
                'best_for': 'Non-stiff systems',
                'function_evals_per_step': '6'
            },
            'DOP853': {
                'name': 'Dormand-Prince 8(5,3)',
                'order': 8,
                'type': 'Adaptive',
                'description': 'Very high accuracy',
                'best_for': 'Precision requirements',
                'function_evals_per_step': '12'
            },
            'BDF': {
                'name': 'Backward Differentiation Formula',
                'order': 'Variable (1-5)',
                'type': 'Implicit',
                'description': 'For very stiff systems',
                'best_for': 'Chemistry, circuits',
                'function_evals_per_step': '1 + Jacobian'
            },
            # Diffrax
            'tsit5': {
                'name': 'Tsitouras 5(4)',
                'order': 5,
                'type': 'Adaptive',
                'description': 'Excellent general purpose, JAX-optimized',
                'best_for': 'JAX optimization workflows',
                'function_evals_per_step': '7'
            },
            'dopri5': {
                'name': 'Dormand-Prince 5(4)',
                'order': 5,
                'type': 'Adaptive',
                'description': 'Classic robust solver',
                'best_for': 'General purpose',
                'function_evals_per_step': '6'
            },
            # Fixed-step
            'euler': {
                'name': 'Explicit Euler',
                'order': 1,
                'type': 'Fixed-step',
                'description': 'Simplest method, educational',
                'best_for': 'Learning, prototyping',
                'function_evals_per_step': '1'
            },
            'midpoint': {
                'name': 'Explicit Midpoint (RK2)',
                'order': 2,
                'type': 'Fixed-step',
                'description': 'Second-order accuracy',
                'best_for': 'Simple simulations',
                'function_evals_per_step': '2'
            },
            'rk4': {
                'name': 'Classic Runge-Kutta 4',
                'order': 4,
                'type': 'Fixed-step',
                'description': 'Excellent accuracy/cost trade-off',
                'best_for': 'Fixed-step simulations',
                'function_evals_per_step': '4'
            },
        }
        
        return method_info.get(method, {
            'name': method,
            'description': 'No information available'
        })


# ============================================================================
# Convenience Functions
# ============================================================================

def create_integrator(
    system: 'SymbolicDynamicalSystem',
    backend: str = 'numpy',
    method: Optional[str] = None,
    **options
) -> IntegratorBase:
    """
    Convenience function for creating integrators.
    
    Alias for IntegratorFactory.create().
    
    Examples
    --------
    >>> integrator = create_integrator(system)
    >>> integrator = create_integrator(system, backend='jax', method='tsit5')
    """
    return IntegratorFactory.create(system, backend, method, **options)


def auto_integrator(system: 'SymbolicDynamicalSystem', **options) -> IntegratorBase:
    """
    Automatically select best available integrator.
    
    Alias for IntegratorFactory.auto().
    
    Examples
    --------
    >>> integrator = auto_integrator(system, rtol=1e-8)
    """
    return IntegratorFactory.auto(system, **options)