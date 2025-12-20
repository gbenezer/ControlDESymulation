"""
SDE Integrator Base - Abstract Interface for Stochastic Integration

Provides a unified interface for numerical integration of stochastic differential
equations (SDEs) across different backends (NumPy, PyTorch, JAX) with support
for both pseudo-random and quasi-random Monte Carlo methods.

This module defines the abstract base class for SDE integrators, handling:
- Brownian motion generation (pseudo-random and quasi-random)
- Multiple trajectory simulation
- Strong vs weak convergence modes
- Backend-specific noise handling

Mathematical Form
-----------------
Itô SDE: dx = f(x, u, t)dt + g(x, u, t)dW
Stratonovich SDE: dx = f(x, u, t)dt + g(x, u, t)∘dW

where:
    f: Drift function (deterministic dynamics)
    g: Diffusion matrix (nx × nw)
    dW: Wiener process increments (nw independent)
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable, Dict, Any, List, Literal, TYPE_CHECKING
from enum import Enum
import warnings
import numpy as np

if TYPE_CHECKING:
    from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem
    import torch
    import jax.numpy as jnp

# Type alias for backend-agnostic arrays
from typing import Union
ArrayLike = Union[np.ndarray, "torch.Tensor", "jnp.ndarray"]


class ConvergenceMode(Enum):
    """
    SDE convergence mode.
    
    Attributes
    ----------
    STRONG : str
        Strong convergence - paths converge
        E[|X_T - X̂_T|] → 0 as dt → 0
        Use for: Path-dependent problems, neural SDEs, control
        
    WEAK : str
        Weak convergence - distributions converge
        E[f(X_T)] - E[f(X̂_T)] → 0 as dt → 0
        Use for: Expectation computation, statistical sampling, option pricing
    """
    STRONG = "strong"
    WEAK = "weak"


class NoiseType(Enum):
    """
    Type of noise generation for Monte Carlo.
    
    Attributes
    ----------
    PSEUDO : str
        Pseudo-random numbers (standard Monte Carlo)
        - Convergence: O(1/√N)
        - Works with adaptive stepping
        - Standard approach for most applications
        
    QUASI : str
        Quasi-random low-discrepancy sequences (QMC)
        - Convergence: O(1/N) or O((log N)^d/N)
        - Requires fixed time grid (incompatible with adaptive stepping)
        - Better for variance reduction in expectation problems
        - Effectiveness decreases in high dimensions
    """
    PSEUDO = "pseudo"
    QUASI = "quasi"


class SDEIntegrationResult:
    """
    Container for SDE integration results.
    
    Stores time points, trajectory ensemble, and statistics.
    
    Attributes
    ----------
    t : ArrayLike
        Time points (T,)
    x : ArrayLike
        State trajectories (num_trajectories, T, nx)
    mean : ArrayLike
        Ensemble mean trajectory (T, nx)
    std : ArrayLike
        Ensemble standard deviation (T, nx)
    success : bool
        Whether integration succeeded
    message : str
        Status message
    nfev : int
        Number of function evaluations
    nsteps : int
        Number of integration steps per trajectory
    num_trajectories : int
        Number of sample paths
    """
    
    def __init__(
        self,
        t: ArrayLike,
        x: ArrayLike,
        success: bool = True,
        message: str = "Integration successful",
        nfev: int = 0,
        nsteps: int = 0,
        **metadata
    ):
        self.t = t
        self.x = x  # (num_trajectories, T, nx)
        self.num_trajectories = x.shape[0] if len(x.shape) > 2 else 1
        
        # Compute statistics
        if len(x.shape) == 3 and x.shape[0] > 1:
            # Multiple trajectories - compute ensemble statistics
            if isinstance(x, np.ndarray):
                self.mean = np.mean(x, axis=0)
                self.std = np.std(x, axis=0)
            else:
                # Will be computed by backend-specific code
                self.mean = None
                self.std = None
        else:
            # Single trajectory
            self.mean = x[0] if len(x.shape) == 3 else x
            self.std = None
        
        self.success = success
        self.message = message
        self.nfev = nfev
        self.nsteps = nsteps
        self.metadata = metadata
    
    def __repr__(self) -> str:
        return (
            f"SDEIntegrationResult(success={self.success}, "
            f"trajectories={self.num_trajectories}, "
            f"nsteps={self.nsteps}, nfev={self.nfev})"
        )


class SDEIntegratorBase(ABC):
    """
    Abstract base class for SDE numerical integrators.
    
    Provides a unified interface for integrating stochastic differential equations
    with multiple backends, noise generation strategies, and convergence modes.
    
    All SDE integrators must implement:
    - step(): Single integration step with Brownian increment
    - integrate(): Multi-step integration over interval with trajectory ensemble
    - name: Integrator name for display
    
    Subclasses handle backend-specific implementations for NumPy, PyTorch, JAX.
    
    Key Features
    -----------
    - Pseudo-random and quasi-random (QMC) noise generation
    - Multiple trajectory simulation for Monte Carlo estimation
    - Strong vs weak convergence modes
    - Reproducible noise with seeding
    - Automatic statistics computation (mean, std)
    - Support for additive and multiplicative noise
    
    Examples
    --------
    >>> # Create SDE integrator
    >>> integrator = EulerMaruyamaIntegrator(
    ...     system,
    ...     dt=0.01,
    ...     noise_type=NoiseType.PSEUDO,
    ...     backend='numpy'
    ... )
    >>> 
    >>> # Single step with provided noise
    >>> dW = np.random.randn(system.nw) * np.sqrt(0.01)
    >>> x_next = integrator.step(x, u, dt=0.01, dW=dW)
    >>> 
    >>> # Multi-trajectory Monte Carlo simulation
    >>> result = integrator.integrate(
    ...     x0=np.array([1.0, 0.0]),
    ...     u_func=lambda t, x: np.zeros(1),
    ...     t_span=(0.0, 10.0),
    ...     num_trajectories=1000,
    ...     seed=42
    ... )
    >>> print(f"Mean at t=10: {result.mean[-1]}")
    >>> print(f"Std at t=10: {result.std[-1]}")
    """
    
    def __init__(
        self,
        system: 'StochasticDynamicalSystem',
        dt: float,
        noise_type: Union[NoiseType, str] = NoiseType.PSEUDO,
        convergence_mode: Union[ConvergenceMode, str] = ConvergenceMode.STRONG,
        backend: str = 'numpy',
        qmc_sequence: Literal["sobol", "halton"] = "sobol",
        **options
    ):
        """
        Initialize SDE integrator.
        
        Parameters
        ----------
        system : StochasticDynamicalSystem
            Stochastic system to integrate
        dt : float
            Time step (must be > 0)
        noise_type : NoiseType or str
            'pseudo' for pseudo-random (standard MC)
            'quasi' for quasi-random (QMC with low-discrepancy sequences)
        convergence_mode : ConvergenceMode or str
            'strong' for path convergence (default)
            'weak' for distributional convergence
        backend : str
            Backend to use ('numpy', 'torch', 'jax')
        qmc_sequence : str
            Low-discrepancy sequence type ('sobol' or 'halton')
            Only used when noise_type='quasi'
        **options : dict
            Integrator-specific options:
            - rtol : float
                Relative tolerance (for adaptive schemes)
            - atol : float
                Absolute tolerance (for adaptive schemes)
            - max_steps : int
                Maximum number of steps (default: 100000)
        
        Raises
        ------
        ValueError
            If dt <= 0
            If backend is invalid
            If system is not StochasticDynamicalSystem
        RuntimeError
            If backend is not available
        
        Examples
        --------
        >>> # Standard pseudo-random integration
        >>> integrator = EulerMaruyamaIntegrator(
        ...     system,
        ...     dt=0.01,
        ...     noise_type='pseudo',
        ...     backend='numpy'
        ... )
        >>> 
        >>> # QMC for variance reduction
        >>> integrator = EulerMaruyamaIntegrator(
        ...     system,
        ...     dt=0.01,
        ...     noise_type='quasi',
        ...     qmc_sequence='sobol',
        ...     backend='jax'
        ... )
        """
        # Validate system type
        from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem
        if not isinstance(system, StochasticDynamicalSystem):
            raise ValueError(
                f"System must be StochasticDynamicalSystem, got {type(system).__name__}. "
                f"Use IntegratorBase for deterministic systems."
            )
        
        self.system = system
        
        # Validate time step
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        self.dt = dt
        
        # Parse enums from strings if needed
        if isinstance(noise_type, str):
            noise_type = NoiseType(noise_type.lower())
        if isinstance(convergence_mode, str):
            convergence_mode = ConvergenceMode(convergence_mode.lower())
        
        self.noise_type = noise_type
        self.convergence_mode = convergence_mode
        self.qmc_sequence = qmc_sequence
        
        # Backend validation
        valid_backends = ['numpy', 'torch', 'jax']
        if backend not in valid_backends:
            raise ValueError(
                f"Invalid backend '{backend}'. Must be one of {valid_backends}"
            )
        self.backend = backend
        
        # System properties
        self.nx = system.nx  # State dimension
        self.nu = system.nu  # Control dimension
        self.nw = system.nw  # Noise dimension (number of Wiener processes)
        
        # Validate QMC compatibility
        if noise_type == NoiseType.QUASI:
            if hasattr(self, 'step_mode') and self.step_mode == 'adaptive':
                warnings.warn(
                    "Quasi-random (QMC) noise with adaptive stepping may lose "
                    "low-discrepancy properties. Consider using fixed time grid.",
                    UserWarning
                )
        
        # Options
        self.options = options
        self.rtol = options.get('rtol', 1e-6)
        self.atol = options.get('atol', 1e-8)
        self.max_steps = options.get('max_steps', 100000)
        
        # QMC state (initialized on first use)
        self._qmc_generator = None
        self._qmc_dimension_index = 0  # Track which dimension we're on
        
        # Statistics
        self._stats = {
            'total_steps': 0,
            'total_fev': 0,  # Function evaluations
            'total_gev': 0,  # Diffusion evaluations
            'total_trajectories': 0,
            'total_time': 0.0,
        }
    
    # ========================================================================
    # Abstract Methods - Must be Implemented by Subclasses
    # ========================================================================
    
    @abstractmethod
    def step(
        self,
        x: ArrayLike,
        u: ArrayLike,
        dt: Optional[float] = None,
        dW: Optional[ArrayLike] = None
    ) -> ArrayLike:
        """
        Take one SDE integration step: x(t) → x(t + dt).
        
        Implements the numerical scheme for advancing the SDE by one time step
        using the provided (or generated) Brownian increment.
        
        Parameters
        ----------
        x : ArrayLike
            Current state (nx,) or (batch, nx)
        u : ArrayLike
            Control input (nu,) or (batch, nu)
        dt : Optional[float]
            Step size (uses self.dt if None)
        dW : Optional[ArrayLike]
            Brownian increment (nw,) or (batch, nw)
            If None, generated internally using noise_type strategy
            
        Returns
        -------
        ArrayLike
            Next state x(t + dt), same shape and type as input
            
        Notes
        -----
        The Brownian increment dW should satisfy:
        - E[dW] = 0
        - E[dW dW^T] = dt * I  (for independent Wiener processes)
        - For strong schemes: dW ~ N(0, dt * I)
        - For weak schemes: simplified distributions may be used
        
        Examples
        --------
        >>> # Automatic noise generation
        >>> x_next = integrator.step(x, u)
        >>> 
        >>> # Provide specific noise (e.g., for reproducibility)
        >>> rng = np.random.RandomState(42)
        >>> dW = rng.randn(system.nw) * np.sqrt(dt)
        >>> x_next = integrator.step(x, u, dW=dW)
        >>> 
        >>> # Batched stepping
        >>> x_batch = np.array([[1.0, 0.0], [2.0, 0.0]])
        >>> dW_batch = rng.randn(2, system.nw) * np.sqrt(dt)
        >>> x_next_batch = integrator.step(x_batch, u_batch, dW=dW_batch)
        """
        pass
    
    @abstractmethod
    def integrate(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], ArrayLike],
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
        return_full_ensemble: bool = True
    ) -> SDEIntegrationResult:
        """
        Integrate SDE over time interval with multiple trajectories.
        
        Simulates an ensemble of sample paths for Monte Carlo estimation
        of expectations, variances, and probability distributions.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,)
            Same initial condition used for all trajectories
        u_func : Callable[[float, ArrayLike], ArrayLike]
            Control policy: (t, x) → u
            Can be:
            - Constant control: lambda t, x: u_const
            - State feedback: lambda t, x: -K @ x  
            - Time-varying: lambda t, x: u(t)
        t_span : Tuple[float, float]
            Integration interval (t_start, t_end)
        t_eval : Optional[ArrayLike]
            Specific times at which to store solution
            If None, uses t = t_start + k*dt for k=0,1,2,...
        num_trajectories : int
            Number of sample paths to simulate (default: 1)
            For Monte Carlo: typically 100-10000
            For QMC: can use fewer (10-1000) due to better convergence
        seed : Optional[int]
            Random seed for reproducibility
            For pseudo-random: standard numpy/torch/jax seed
            For quasi-random: starting point in low-discrepancy sequence
        return_full_ensemble : bool
            If True, return all trajectories (default)
            If False, return only statistics (mean, std)
            Set False for memory efficiency with large num_trajectories
            
        Returns
        -------
        SDEIntegrationResult
            Object containing:
            - t: Time points (T,)
            - x: Trajectories (num_trajectories, T, nx) if return_full_ensemble
                 else only mean trajectory
            - mean: Ensemble mean (T, nx)
            - std: Ensemble std (T, nx)
            - success: Whether integration succeeded
            - nfev: Number of drift function evaluations
            - nsteps: Number of steps per trajectory
            
        Raises
        ------
        RuntimeError
            If integration fails (e.g., max steps exceeded, numerical instability)
        
        Examples
        --------
        >>> # Monte Carlo with 1000 paths
        >>> result = integrator.integrate(
        ...     x0=np.array([1.0, 0.0]),
        ...     u_func=lambda t, x: np.zeros(1),
        ...     t_span=(0.0, 10.0),
        ...     num_trajectories=1000,
        ...     seed=42
        ... )
        >>> print(f"Mean endpoint: {result.mean[-1]}")
        >>> print(f"Std endpoint: {result.std[-1]}")
        >>> 
        >>> # Quasi-Monte Carlo for variance reduction
        >>> qmc_integrator = EulerMaruyamaIntegrator(
        ...     system, dt=0.01, noise_type='quasi'
        ... )
        >>> result = qmc_integrator.integrate(
        ...     x0=x0,
        ...     u_func=u_func,
        ...     t_span=(0.0, 10.0),
        ...     num_trajectories=100,  # Fewer trajectories needed
        ...     seed=0
        ... )
        >>> 
        >>> # State feedback control with single trajectory
        >>> K = np.array([[1.0, 2.0]])
        >>> result = integrator.integrate(
        ...     x0=x0,
        ...     u_func=lambda t, x: -K @ x,
        ...     t_span=(0.0, 10.0),
        ...     num_trajectories=1
        ... )
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get integrator name for display and logging.
        
        Returns
        -------
        str
            Human-readable integrator name
            
        Examples
        --------
        >>> integrator.name
        'Euler-Maruyama (Strong, Pseudo-random)'
        >>> qmc_integrator.name
        'Euler-Maruyama (Strong, QMC-Sobol)'
        """
        pass
    
    # ========================================================================
    # Noise Generation - Pseudo-random and Quasi-random
    # ========================================================================
    
    def _generate_noise(
        self,
        dt: float,
        shape: Tuple[int, ...],
        seed: Optional[int] = None
    ) -> ArrayLike:
        """
        Generate Brownian increments using configured noise type.
        
        Dispatches to pseudo-random or quasi-random generation based on
        self.noise_type setting.
        
        Parameters
        ----------
        dt : float
            Time step (for scaling: dW ~ N(0, dt))
        shape : Tuple[int, ...]
            Shape of noise array
            - (nw,) for single step
            - (num_steps, nw) for trajectory
            - (num_trajectories, num_steps, nw) for ensemble
        seed : Optional[int]
            Random seed
            
        Returns
        -------
        ArrayLike
            Brownian increments, scaled by sqrt(dt)
        """
        if self.noise_type == NoiseType.PSEUDO:
            return self._generate_pseudo_noise(dt, shape, seed)
        elif self.noise_type == NoiseType.QUASI:
            return self._generate_quasi_noise(dt, shape, seed)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
    
    def _generate_pseudo_noise(
        self,
        dt: float,
        shape: Tuple[int, ...],
        seed: Optional[int] = None
    ) -> ArrayLike:
        """
        Generate pseudo-random Brownian increments.
        
        Uses backend-specific random number generation:
        - NumPy: numpy.random.randn
        - PyTorch: torch.randn
        - JAX: jax.random.normal
        
        Parameters
        ----------
        dt : float
            Time step
        shape : Tuple[int, ...]
            Output shape
        seed : Optional[int]
            Random seed for reproducibility
            
        Returns
        -------
        ArrayLike
            dW ~ N(0, dt * I)
        """
        sqrt_dt = np.sqrt(dt)
        
        if self.backend == 'numpy':
            if seed is not None:
                np.random.seed(seed)
            return np.random.randn(*shape) * sqrt_dt
            
        elif self.backend == 'torch':
            import torch
            if seed is not None:
                torch.manual_seed(seed)
            return torch.randn(*shape) * sqrt_dt
            
        elif self.backend == 'jax':
            import jax
            import jax.numpy as jnp
            if seed is None:
                seed = 0
            key = jax.random.PRNGKey(seed)
            return jax.random.normal(key, shape) * sqrt_dt
            
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _generate_quasi_noise(
        self,
        dt: float,
        shape: Tuple[int, ...],
        seed: Optional[int] = None
    ) -> ArrayLike:
        """
        Generate quasi-random Brownian increments using low-discrepancy sequences.
        
        Uses Sobol or Halton sequences for better space-filling properties.
        Applies inverse normal CDF to map uniform [0,1] samples to Gaussian.
        
        Parameters
        ----------
        dt : float
            Time step
        shape : Tuple[int, ...]
            Output shape
        seed : Optional[int]
            Starting index in sequence (not a random seed!)
            
        Returns
        -------
        ArrayLike
            dW with low-discrepancy structure, scaled by sqrt(dt)
            
        Notes
        -----
        - QMC effectiveness decreases in high dimensions (d > 20-30)
        - Total dimensions = nw * num_steps, so keep time horizons reasonable
        - For very long integrations, consider randomized QMC (RQMC)
        
        Warnings
        --------
        - QMC requires fixed time grid (incompatible with adaptive stepping)
        - Samples are negatively correlated (not independent like pseudo-random)
        """
        sqrt_dt = np.sqrt(dt)
        
        # Initialize QMC generator on first use
        if self._qmc_generator is None:
            self._initialize_qmc_generator(seed)
        
        # Total number of samples needed
        total_samples = int(np.prod(shape))
        
        # Generate uniform [0,1] samples from low-discrepancy sequence
        if self.backend == 'numpy':
            from scipy.stats import qmc
            uniform_samples = self._qmc_generator.random(total_samples)
            
            # Box-Muller transform: U[0,1] → N(0,1)
            # For QMC, use inverse CDF (more stable)
            from scipy.special import ndtri  # Inverse normal CDF
            gaussian_samples = ndtri(uniform_samples)
            
            # Reshape and scale
            noise = gaussian_samples.reshape(shape) * sqrt_dt
            return noise
            
        elif self.backend == 'jax':
            import jax.numpy as jnp
            # JAX has native Sobol support
            # (Implementation would go here - placeholder for now)
            warnings.warn(
                "JAX QMC implementation pending. Falling back to pseudo-random.",
                UserWarning
            )
            return self._generate_pseudo_noise(dt, shape, seed)
            
        elif self.backend == 'torch':
            import torch
            # PyTorch doesn't have native QMC - use NumPy then convert
            noise_np = self._generate_quasi_noise(dt, shape, seed)
            return torch.from_numpy(noise_np).float()
            
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _initialize_qmc_generator(self, seed: Optional[int] = None):
        """
        Initialize quasi-Monte Carlo generator.
        
        Parameters
        ----------
        seed : Optional[int]
            Starting index in sequence
        """
        if self.backend in ['numpy', 'torch']:
            from scipy.stats import qmc
            
            # Dimension is number of Wiener processes
            # (Will be used repeatedly for each time step)
            d = self.nw
            
            if self.qmc_sequence == 'sobol':
                self._qmc_generator = qmc.Sobol(d=d, scramble=True, seed=seed)
            elif self.qmc_sequence == 'halton':
                self._qmc_generator = qmc.Halton(d=d, scramble=True, seed=seed)
            else:
                raise ValueError(
                    f"Unknown QMC sequence: {self.qmc_sequence}. "
                    f"Use 'sobol' or 'halton'."
                )
                
            # Skip to seed position if provided
            if seed is not None and seed > 0:
                self._qmc_generator.fast_forward(seed)
                
        elif self.backend == 'jax':
            # JAX QMC would be initialized here
            pass
    
    # ========================================================================
    # Common Utilities
    # ========================================================================
    
    def _evaluate_drift(self, x: ArrayLike, u: ArrayLike) -> ArrayLike:
        """
        Evaluate drift function with statistics tracking.
        
        Parameters
        ----------
        x : ArrayLike
            State
        u : ArrayLike
            Control
            
        Returns
        -------
        ArrayLike
            Drift f(x, u, t)
        """
        self._stats['total_fev'] += 1
        return self.system.drift(x, u, backend=self.backend)
    
    def _evaluate_diffusion(self, x: ArrayLike, u: ArrayLike) -> ArrayLike:
        """
        Evaluate diffusion matrix with statistics tracking.
        
        Parameters
        ----------
        x : ArrayLike
            State
        u : ArrayLike
            Control
            
        Returns
        -------
        ArrayLike
            Diffusion g(x, u, t), shape (nx, nw)
        """
        self._stats['total_gev'] += 1
        return self.system.diffusion(x, u, backend=self.backend)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get integration statistics.
        
        Returns
        -------
        dict
            Statistics with keys:
            - 'total_steps': Total integration steps taken
            - 'total_fev': Total drift function evaluations
            - 'total_gev': Total diffusion evaluations
            - 'total_trajectories': Total trajectories simulated
            - 'total_time': Total integration time
            - 'avg_fev_per_step': Average drift evals per step
            - 'avg_gev_per_step': Average diffusion evals per step
            
        Examples
        --------
        >>> result = integrator.integrate(x0, u_func, (0, 10), num_trajectories=100)
        >>> stats = integrator.get_stats()
        >>> print(f"Trajectories: {stats['total_trajectories']}")
        >>> print(f"Steps per trajectory: {stats['total_steps'] / 100}")
        >>> print(f"Drift evals: {stats['total_fev']}")
        """
        avg_fev = self._stats['total_fev'] / max(1, self._stats['total_steps'])
        avg_gev = self._stats['total_gev'] / max(1, self._stats['total_steps'])
        
        return {
            **self._stats,
            'avg_fev_per_step': avg_fev,
            'avg_gev_per_step': avg_gev,
        }
    
    def reset_stats(self):
        """
        Reset integration statistics to zero.
        
        Examples
        --------
        >>> integrator.reset_stats()
        >>> integrator.get_stats()['total_steps']
        0
        """
        self._stats['total_steps'] = 0
        self._stats['total_fev'] = 0
        self._stats['total_gev'] = 0
        self._stats['total_trajectories'] = 0
        self._stats['total_time'] = 0.0
    
    def _check_noise_characteristics(self) -> Dict[str, bool]:
        """
        Check system noise characteristics for optimization hints.
        
        Returns
        -------
        dict
            Noise properties that can enable faster integration:
            - 'is_additive': Constant diffusion (no state/control dependence)
            - 'is_diagonal': Independent noise sources
            - 'is_scalar': Single noise source
        """
        return {
            'is_additive': self.system.is_additive_noise(),
            'is_diagonal': self.system.is_diagonal_noise(),
            'is_scalar': self.system.is_scalar_noise(),
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"dt={self.dt}, noise={self.noise_type.value}, "
            f"mode={self.convergence_mode.value}, backend={self.backend})"
        )
    
    def __str__(self) -> str:
        """Human-readable string."""
        noise_str = (
            f"QMC-{self.qmc_sequence}" 
            if self.noise_type == NoiseType.QUASI 
            else "Pseudo"
        )
        return f"{self.name} (dt={self.dt:.4f}, {noise_str}, {self.backend})"