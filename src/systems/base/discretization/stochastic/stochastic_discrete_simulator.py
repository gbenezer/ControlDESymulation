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
Stochastic Discrete-Time Trajectory Simulator

Simulates stochastic discrete-time trajectories with Monte Carlo support.

This simulator handles:
- Pure discrete stochastic systems (DiscreteStochasticSystem)
- Discretized continuous SDEs (StochasticDynamicalSystem + StochasticDiscretizer)
- Controller types (sequence, function, nn.Module, None)
- Output feedback with observers
- Single and multiple trajectory simulation (Monte Carlo)
- Batched simulation for efficiency
- Noise seeding and reproducibility
- Statistical analysis of trajectories

Architecture
-----------
Extends DiscreteSimulator to add:
- Noise handling (custom or auto-generated)
- Monte Carlo simulation (n_paths trajectories)
- Statistical analysis (mean, variance, quantiles)
- Reproducible random number generation

Delegates to:
- DiscreteStochasticSystem.step_stochastic() for pure discrete stochastic
- StochasticDiscretizer.step() for discretized SDEs
- Observer.update() for output feedback (when provided)
"""

from typing import Optional, Union, Callable, Tuple, Dict, Any
import numpy as np

# Conditional imports for optional backends
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module:
            pass

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.systems.base.discrete_stochastic_system import DiscreteStochasticSystem
    from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem
    from src.discretization.stochastic_discretizer import StochasticDiscretizer
    from src.observers.observer_base import Observer

# Import base simulator
from src.systems.base.discretization.discrete_simulator import DiscreteSimulator

# Type alias
ArrayLike = Union[np.ndarray, "torch.Tensor", "jnp.ndarray"]


class MonteCarloResult:
    """
    Container for Monte Carlo simulation results.
    
    Stores multiple trajectories and provides statistical analysis.
    
    Attributes
    ----------
    states : ArrayLike
        All trajectories, shape (n_paths, steps+1, nx)
    controls : Optional[ArrayLike]
        All control sequences, shape (n_paths, steps, nu)
    noise : Optional[ArrayLike]
        All noise samples, shape (n_paths, steps, nw)
    n_paths : int
        Number of trajectories
    steps : int
        Number of time steps per trajectory
    """
    
    def __init__(
        self,
        states: ArrayLike,
        controls: Optional[ArrayLike] = None,
        noise: Optional[ArrayLike] = None,
        n_paths: int = 0,
        steps: int = 0
    ):
        self.states = states
        self.controls = controls
        self.noise = noise
        self.n_paths = n_paths
        self.steps = steps
    
    def get_statistics(self) -> Dict[str, ArrayLike]:
        """
        Compute trajectory statistics across paths.
        
        Returns
        -------
        dict
            Statistics with keys:
            - 'mean': Mean trajectory (steps+1, nx)
            - 'std': Standard deviation (steps+1, nx)
            - 'min': Minimum values (steps+1, nx)
            - 'max': Maximum values (steps+1, nx)
            - 'median': Median trajectory (steps+1, nx)
            - 'q25': 25th percentile (steps+1, nx)
            - 'q75': 75th percentile (steps+1, nx)
        
        Examples
        --------
        >>> result = sim.simulate_monte_carlo(x0, steps=100, n_paths=1000)
        >>> stats = result.get_statistics()
        >>> 
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(stats['mean'], label='Mean')
        >>> plt.fill_between(range(101), 
        ...                  stats['mean'] - stats['std'],
        ...                  stats['mean'] + stats['std'],
        ...                  alpha=0.3, label='±1σ')
        """
        if isinstance(self.states, np.ndarray):
            return {
                'mean': np.mean(self.states, axis=0),
                'std': np.std(self.states, axis=0),
                'min': np.min(self.states, axis=0),
                'max': np.max(self.states, axis=0),
                'median': np.median(self.states, axis=0),
                'q25': np.quantile(self.states, 0.25, axis=0),
                'q75': np.quantile(self.states, 0.75, axis=0),
            }
        elif TORCH_AVAILABLE and isinstance(self.states, torch.Tensor):
            return {
                'mean': torch.mean(self.states, dim=0),
                'std': torch.std(self.states, dim=0),
                'min': torch.min(self.states, dim=0).values,
                'max': torch.max(self.states, dim=0).values,
                'median': torch.median(self.states, dim=0).values,
                'q25': torch.quantile(self.states, 0.25, dim=0),
                'q75': torch.quantile(self.states, 0.75, dim=0),
            }
        elif JAX_AVAILABLE and isinstance(self.states, jnp.ndarray):
            return {
                'mean': jnp.mean(self.states, axis=0),
                'std': jnp.std(self.states, axis=0),
                'min': jnp.min(self.states, axis=0),
                'max': jnp.max(self.states, axis=0),
                'median': jnp.median(self.states, axis=0),
                'q25': jnp.quantile(self.states, 0.25, axis=0),
                'q75': jnp.quantile(self.states, 0.75, axis=0),
            }
    
    def __repr__(self) -> str:
        return f"MonteCarloResult(n_paths={self.n_paths}, steps={self.steps})"


class StochasticDiscreteSimulator(DiscreteSimulator):
    """
    Simulates stochastic discrete-time trajectories.
    
    Extends DiscreteSimulator to handle stochastic systems with:
    - Noise generation and seeding
    - Monte Carlo simulation (multiple paths)
    - Statistical analysis
    - Custom or automatic noise
    
    Handles both:
    - Pure discrete stochastic (DiscreteStochasticSystem)
    - Discretized continuous SDEs (StochasticDynamicalSystem + StochasticDiscretizer)
    
    Key Features
    -----------
    - Single and multiple trajectory simulation
    - Reproducible noise generation via seeding
    - Custom noise sequences (for quasi-MC, antithetic variates, etc.)
    - Automatic statistical analysis across paths
    - Batched evaluation for efficiency
    - All controller types from parent (sequence, function, nn.Module, None)
    
    Parameters
    ----------
    system : Union[DiscreteStochasticSystem, StochasticDynamicalSystem]
        Stochastic system to simulate
    discretizer : Optional[StochasticDiscretizer]
        Required for continuous SDEs, None for pure discrete stochastic
    observer : Optional[Observer]
        Optional observer for output feedback
    seed : Optional[int]
        Random seed for reproducibility
    
    Examples
    --------
    Pure discrete stochastic:
    >>> from src.systems.builtin.stochastic.discrete_ar1 import DiscreteAR1
    >>> system = DiscreteAR1(phi=0.9, sigma=0.2)
    >>> sim = StochasticDiscreteSimulator(system, seed=42)
    >>> 
    >>> # Single trajectory
    >>> states = sim.simulate(x0, steps=100)
    >>> 
    >>> # Monte Carlo (1000 paths)
    >>> result = sim.simulate_monte_carlo(x0, steps=100, n_paths=1000)
    >>> stats = result.get_statistics()
    >>> print(f"Mean at final time: {stats['mean'][-1]}")
    
    Discretized SDE:
    >>> from src.discretization.stochastic_discretizer import StochasticDiscretizer
    >>> sde = OrnsteinUhlenbeck(alpha=2.0, sigma=0.5)
    >>> discretizer = StochasticDiscretizer(sde, dt=0.01, method='euler')
    >>> sim = StochasticDiscreteSimulator(sde, discretizer=discretizer, seed=42)
    >>> 
    >>> result = sim.simulate_monte_carlo(x0, steps=1000, n_paths=500, dt=0.01)
    
    Custom noise (quasi-Monte Carlo):
    >>> # Sobol sequence for low-discrepancy sampling
    >>> from scipy.stats.qmc import Sobol
    >>> sobol = Sobol(d=1, scramble=True, seed=42)
    >>> noise_sequence = sobol.random(100)  # (100, 1)
    >>> 
    >>> states = sim.simulate(x0, steps=100, noise=noise_sequence)
    """
    
    def __init__(
        self,
        system: Union['DiscreteStochasticSystem', 'StochasticDynamicalSystem'],
        discretizer: Optional['StochasticDiscretizer'] = None,
        observer: Optional['Observer'] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize stochastic discrete simulator.
        
        Parameters
        ----------
        system : Union[DiscreteStochasticSystem, StochasticDynamicalSystem]
            Stochastic system to simulate
        discretizer : Optional[StochasticDiscretizer]
            Required for continuous SDEs
        observer : Optional[Observer]
            Observer for output feedback
        seed : Optional[int]
            Random seed for reproducibility
        
        Raises
        ------
        TypeError
            If system is not stochastic
            If continuous SDE without discretizer
        """
        # Call parent initialization
        super().__init__(system, discretizer, observer)
        
        # Validate system is stochastic
        if not hasattr(system, 'is_stochastic') or not system.is_stochastic:
            raise TypeError(
                f"System {system.__class__.__name__} is not stochastic. "
                f"Use DiscreteSimulator for deterministic systems."
            )
        
        # Store noise dimension
        self.nw = system.nw
        
        # Random seed management
        self.seed = seed
        self._rng_state = self._initialize_rng(seed)
    
    def _initialize_rng(self, seed: Optional[int]):
        """Initialize random number generator."""
        if seed is not None:
            np.random.seed(seed)
            if TORCH_AVAILABLE:
                torch.manual_seed(seed)
        
        if self.backend == 'numpy':
            return np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        elif self.backend == 'torch':
            # PyTorch uses global RNG
            return None
        elif self.backend == 'jax':
            # JAX uses explicit keys
            import jax
            return jax.random.PRNGKey(seed if seed is not None else 0)
    
    def set_seed(self, seed: int):
        """
        Set random seed for reproducibility.
        
        Parameters
        ----------
        seed : int
            Random seed
        
        Examples
        --------
        >>> sim.set_seed(42)
        >>> result1 = sim.simulate(x0, steps=100)
        >>> sim.set_seed(42)
        >>> result2 = sim.simulate(x0, steps=100)
        >>> # result1 and result2 are identical
        """
        self.seed = seed
        self._rng_state = self._initialize_rng(seed)
    
    # ========================================================================
    # Single Trajectory Simulation
    # ========================================================================
    
    def simulate(
        self,
        x0: ArrayLike,
        steps: int,
        controller: Optional[Union[ArrayLike, Callable, 'nn.Module']] = None,
        noise: Optional[ArrayLike] = None,
        dt: Optional[float] = None,
        return_controls: bool = False,
        return_noise: bool = False,
        return_final_only: bool = False,
    ) -> Union[ArrayLike, Tuple[ArrayLike, ...]]:
        """
        Simulate single stochastic trajectory.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,) or (batch, nx)
        steps : int
            Number of discrete time steps
        controller : Optional[Union[ArrayLike, Callable, nn.Module]]
            Controller (same types as parent DiscreteSimulator)
        noise : Optional[ArrayLike]
            Custom noise sequence, shape (steps, nw) or (batch, steps, nw)
            If None, generated automatically using RNG
        dt : Optional[float]
            Time step (required for discretized SDEs)
        return_controls : bool
            If True, return controls
        return_noise : bool
            If True, return noise samples used
        return_final_only : bool
            If True, return only final state
        
        Returns
        -------
        states : ArrayLike
            State trajectory
        controls : ArrayLike (optional)
            Control sequence (if return_controls=True)
        noise_samples : ArrayLike (optional)
            Noise samples used (if return_noise=True)
        
        Examples
        --------
        >>> # Single trajectory with auto noise
        >>> states = sim.simulate(x0, steps=100)
        >>> 
        >>> # With custom noise (reproducibility)
        >>> noise_seq = np.random.randn(100, 1)
        >>> states = sim.simulate(x0, steps=100, noise=noise_seq)
        >>> 
        >>> # Return everything
        >>> states, controls, noise = sim.simulate(
        ...     x0, steps=100, controller=controller,
        ...     return_controls=True, return_noise=True
        ... )
        
        >>> # Deterministic (w=0)
        >>> noise_zero = np.zeros((100, 1))
        >>> states = sim.simulate(x0, steps=100, noise=noise_zero)
        """
        # Validate
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        
        if self.discretizer is not None and dt is None:
            raise ValueError("dt must be provided for discretized SDEs")
        
        # Handle batched vs single trajectory
        if self._get_ndim(x0) == 1:
            x0 = self._expand_batch(x0, 0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        batch_size, state_dim = self._get_shape(x0)
        
        if state_dim != self.nx:
            raise ValueError(
                f"Initial state dimension {state_dim} doesn't match system dimension {self.nx}"
            )
        
        # Initialize storage
        if not return_final_only:
            states = self._zeros(batch_size, steps + 1, state_dim)
            states = self._set_initial(states, x0)
        else:
            x_current = self._clone(x0)
        
        if return_controls:
            controls = self._zeros(batch_size, steps, self.nu)
        
        if return_noise:
            noise_samples = self._zeros(batch_size, steps, self.nw)
        
        # Initialize observer if needed
        if self.observer is not None:
            x_hat = self.observer.initialize(x0)
        else:
            x_hat = self._clone(x0) if return_final_only else self._get_state(states, 0)
        
        # Main simulation loop
        for k in range(steps):
            # Get current state for control computation
            if return_final_only:
                x_ctrl = x_hat if self.observer is not None else x_current
            else:
                x_ctrl = x_hat if self.observer is not None else self._get_state(states, k)
            
            # Compute control
            u = self._compute_control(controller, x_ctrl, k, batch_size)
            
            if return_controls:
                controls = self._set_control(controls, k, u)
            
            # Get or generate noise
            if noise is not None:
                # Use provided noise
                if self._get_ndim(noise) == 2:
                    # Single trajectory noise: (steps, nw)
                    w_k = self._get_item(noise, k)
                    if batch_size > 1:
                        w_k = self._expand_to_batch(w_k, batch_size)
                    else:
                        w_k = self._expand_batch(w_k, 0)
                else:
                    # Batched noise: (batch, steps, nw)
                    w_k = self._get_batch_item(noise, k)
            else:
                # Generate noise
                w_k = self._generate_noise((batch_size, self.nw))
            
            if return_noise:
                noise_samples = self._set_control(noise_samples, k, w_k)
            
            # Propagate stochastic dynamics
            if return_final_only:
                x_next = self._step_stochastic(x_current, u, w_k, dt)
                x_current = x_next
            else:
                x_current = self._get_state(states, k)
                x_next = self._step_stochastic(x_current, u, w_k, dt)
                states = self._set_state(states, k + 1, x_next)
            
            # Update observer
            if self.observer is not None:
                y = self.system.h(x_next, backend=self.backend)
                x_hat = self.observer.update(x_hat, u, y)
        
        # Prepare outputs
        outputs = []
        
        if return_final_only:
            outputs.append(self._squeeze_if_needed(x_current, squeeze_batch))
        else:
            outputs.append(self._squeeze_if_needed(states, squeeze_batch))
        
        if return_controls:
            outputs.append(self._squeeze_if_needed(controls, squeeze_batch))
        
        if return_noise:
            outputs.append(self._squeeze_if_needed(noise_samples, squeeze_batch))
        
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
    
    # ========================================================================
    # Monte Carlo Simulation
    # ========================================================================
    
    def simulate_monte_carlo(
        self,
        x0: ArrayLike,
        steps: int,
        n_paths: int,
        controller: Optional[Union[ArrayLike, Callable, 'nn.Module']] = None,
        dt: Optional[float] = None,
        return_controls: bool = False,
        return_noise: bool = False,
        parallel: bool = True,
    ) -> MonteCarloResult:
        """
        Simulate multiple trajectories for Monte Carlo analysis.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,) - same for all paths
        steps : int
            Number of time steps
        n_paths : int
            Number of independent trajectories
        controller : Optional
            Controller (same for all paths)
        dt : Optional[float]
            Time step (for discretized SDEs)
        return_controls : bool
            Store control sequences
        return_noise : bool
            Store noise samples
        parallel : bool
            Use batched evaluation if True (faster)
            If False, loop over paths (more memory efficient)
        
        Returns
        -------
        MonteCarloResult
            Container with trajectories and statistics
        
        Examples
        --------
        >>> # Monte Carlo with 1000 paths
        >>> result = sim.simulate_monte_carlo(
        ...     x0=np.array([1.0]),
        ...     steps=100,
        ...     n_paths=1000,
        ...     controller=lambda x, k: -0.5*x
        ... )
        >>> 
        >>> # Analyze statistics
        >>> stats = result.get_statistics()
        >>> print(f"Final mean: {stats['mean'][-1]}")
        >>> print(f"Final std: {stats['std'][-1]}")
        >>> 
        >>> # Access individual paths
        >>> path_0 = result.states[0]  # First trajectory
        >>> 
        >>> # Confidence intervals
        >>> lower = stats['mean'] - 1.96 * stats['std']
        >>> upper = stats['mean'] + 1.96 * stats['std']
        """
        if n_paths <= 0:
            raise ValueError(f"n_paths must be positive, got {n_paths}")
        
        # Ensure x0 is 1D
        if self._get_ndim(x0) > 1:
            if x0.shape[0] == 1:
                x0 = self._squeeze_if_needed(x0, True)
            else:
                raise ValueError(
                    f"For Monte Carlo, x0 must be single state (nx,), "
                    f"got shape {x0.shape}. All paths start from same x0."
                )
        
        if parallel and n_paths > 1:
            # Batched evaluation (efficient)
            # Create batched initial condition
            if self.backend == 'numpy':
                x0_batch = np.tile(x0, (n_paths, 1))
            elif self.backend == 'torch':
                import torch
                x0_batch = x0.unsqueeze(0).expand(n_paths, -1)
            elif self.backend == 'jax':
                import jax.numpy as jnp
                x0_batch = jnp.tile(x0, (n_paths, 1))
            
            # Simulate all paths at once
            result = self.simulate(
                x0_batch,
                steps=steps,
                controller=controller,
                noise=None,  # Auto-generate per path
                dt=dt,
                return_controls=return_controls,
                return_noise=return_noise,
                return_final_only=False,
            )
            
            # Unpack results
            if return_controls and return_noise:
                states, controls, noise_samples = result
            elif return_controls:
                states, controls = result
                noise_samples = None
            elif return_noise:
                states, noise_samples = result
                controls = None
            else:
                states = result
                controls = None
                noise_samples = None
            
            return MonteCarloResult(
                states=states,
                controls=controls,
                noise=noise_samples,
                n_paths=n_paths,
                steps=steps
            )
        
        else:
            # Sequential evaluation (memory efficient)
            all_states = []
            all_controls = [] if return_controls else None
            all_noise = [] if return_noise else None
            
            for i in range(n_paths):
                # Simulate single path
                result = self.simulate(
                    x0,
                    steps=steps,
                    controller=controller,
                    noise=None,
                    dt=dt,
                    return_controls=return_controls,
                    return_noise=return_noise,
                    return_final_only=False,
                )
                
                # Unpack
                if return_controls and return_noise:
                    states_i, controls_i, noise_i = result
                    all_controls.append(controls_i)
                    all_noise.append(noise_i)
                elif return_controls:
                    states_i, controls_i = result
                    all_controls.append(controls_i)
                elif return_noise:
                    states_i, noise_i = result
                    all_noise.append(noise_i)
                else:
                    states_i = result
                
                all_states.append(states_i)
            
            # Stack results
            if self.backend == 'numpy':
                states = np.stack(all_states, axis=0)
                controls = np.stack(all_controls, axis=0) if all_controls else None
                noise_samples = np.stack(all_noise, axis=0) if all_noise else None
            elif self.backend == 'torch':
                import torch
                states = torch.stack(all_states, dim=0)
                controls = torch.stack(all_controls, dim=0) if all_controls else None
                noise_samples = torch.stack(all_noise, dim=0) if all_noise else None
            elif self.backend == 'jax':
                import jax.numpy as jnp
                states = jnp.stack(all_states, axis=0)
                controls = jnp.stack(all_controls, axis=0) if all_controls else None
                noise_samples = jnp.stack(all_noise, axis=0) if all_noise else None
            
            return MonteCarloResult(
                states=states,
                controls=controls,
                noise=noise_samples,
                n_paths=n_paths,
                steps=steps
            )
    
    # ========================================================================
    # Stochastic Dynamics Propagation
    # ========================================================================
    
    def _step_stochastic(
        self,
        x: ArrayLike,
        u: ArrayLike,
        w: ArrayLike,
        dt: Optional[float],
    ) -> ArrayLike:
        """
        Single stochastic discrete-time step.
        
        Parameters
        ----------
        x : ArrayLike
            Current state (batch_size, nx)
        u : ArrayLike
            Control input (batch_size, nu)
        w : ArrayLike
            Noise sample (batch_size, nw)
        dt : Optional[float]
            Time step (only for discretized SDEs)
        
        Returns
        -------
        ArrayLike
            Next state (batch_size, nx)
        """
        # Check if system has step_stochastic method (DiscreteStochasticSystem)
        if hasattr(self.system, 'step_stochastic'):
            # Pure discrete stochastic
            if self.nu == 0:
                return self.system.step_stochastic(x, u_k=None, w_k=w, backend=self.backend)
            else:
                return self.system.step_stochastic(x, u, w, backend=self.backend)
        
        elif self.discretizer is not None:
            # Discretized continuous SDE
            if self.nu == 0:
                return self.discretizer.step(x, u=None, w=w, dt=dt)
            else:
                return self.discretizer.step(x, u, w=w, dt=dt)
        
        else:
            raise RuntimeError(
                "System is neither DiscreteStochasticSystem nor has StochasticDiscretizer. "
                "This should have been caught in __init__."
            )
    
    def _generate_noise(self, shape):
        """Generate standard normal noise using RNG."""
        if self.backend == 'numpy':
            if self._rng_state is not None:
                return self._rng_state.randn(*shape)
            else:
                return np.random.randn(*shape)
        
        elif self.backend == 'torch':
            import torch
            return torch.randn(*shape)
        
        elif self.backend == 'jax':
            import jax
            # Split key for next use
            key, subkey = jax.random.split(self._rng_state)
            self._rng_state = key
            return jax.random.normal(subkey, shape)
    
    # ========================================================================
    # Statistical Analysis Utilities
    # ========================================================================
    
    def estimate_statistics(
        self,
        x0: ArrayLike,
        steps: int,
        n_paths: int,
        controller: Optional[Union[ArrayLike, Callable, 'nn.Module']] = None,
        dt: Optional[float] = None,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Estimate statistics with confidence intervals.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state
        steps : int
            Number of steps
        n_paths : int
            Number of Monte Carlo paths
        controller : Optional
            Controller
        dt : Optional[float]
            Time step
        confidence_level : float
            Confidence level for intervals (default: 0.95)
        
        Returns
        -------
        dict
            Statistics with confidence intervals
        
        Examples
        --------
        >>> stats = sim.estimate_statistics(x0, steps=100, n_paths=1000)
        >>> print(f"Mean: {stats['mean'][-1]}")
        >>> print(f"95% CI: [{stats['ci_lower'][-1]}, {stats['ci_upper'][-1]}]")
        """
        result = self.simulate_monte_carlo(
            x0, steps=steps, n_paths=n_paths, controller=controller, dt=dt
        )
        
        base_stats = result.get_statistics()
        
        # Add confidence intervals
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        std_error = base_stats['std'] / np.sqrt(n_paths)
        
        base_stats['ci_lower'] = base_stats['mean'] - z_score * std_error
        base_stats['ci_upper'] = base_stats['mean'] + z_score * std_error
        base_stats['confidence_level'] = confidence_level
        base_stats['n_paths'] = n_paths
        
        return base_stats
    
    # ========================================================================
    # Variance Reduction Techniques
    # ========================================================================
    
    def simulate_antithetic(
        self,
        x0: ArrayLike,
        steps: int,
        n_pairs: int,
        controller: Optional[Union[ArrayLike, Callable, 'nn.Module']] = None,
        dt: Optional[float] = None,
    ) -> MonteCarloResult:
        """
        Simulate using antithetic variates for variance reduction.
        
        Generates n_pairs pairs of trajectories using (w, -w) noise pairs.
        This reduces variance in Monte Carlo estimates.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state
        steps : int
            Number of steps
        n_pairs : int
            Number of antithetic pairs (total paths = 2*n_pairs)
        controller : Optional
            Controller
        dt : Optional[float]
            Time step
        
        Returns
        -------
        MonteCarloResult
            Results with 2*n_pairs trajectories
        
        Examples
        --------
        >>> # Antithetic variates (variance reduction)
        >>> result = sim.simulate_antithetic(x0, steps=100, n_pairs=500)
        >>> # 1000 paths total (500 pairs)
        >>> result.n_paths
        1000
        >>> 
        >>> # Compare variance with standard Monte Carlo
        >>> stats_anti = result.get_statistics()
        >>> result_std = sim.simulate_monte_carlo(x0, steps=100, n_paths=1000)
        >>> stats_std = result_std.get_statistics()
        >>> # stats_anti['std'] should be smaller
        
        Notes
        -----
        Antithetic variates reduce variance for linear functionals.
        Effectiveness depends on system linearity.
        """
        all_states = []
        all_controls = [] if controller is not None else None
        
        for i in range(n_pairs):
            # Generate noise
            if self.backend == 'numpy':
                w_pos = self._rng_state.randn(steps, self.nw)
            else:
                w_pos = self._generate_noise((steps, self.nw))
            
            # Positive noise path
            result_pos = self.simulate(
                x0, steps=steps, controller=controller,
                noise=w_pos, dt=dt,
                return_controls=(controller is not None)
            )
            
            if controller is not None:
                states_pos, controls_pos = result_pos
                all_controls.append(controls_pos)
            else:
                states_pos = result_pos
            
            all_states.append(states_pos)
            
            # Negative noise path (antithetic)
            w_neg = -w_pos
            result_neg = self.simulate(
                x0, steps=steps, controller=controller,
                noise=w_neg, dt=dt,
                return_controls=(controller is not None)
            )
            
            if controller is not None:
                states_neg, controls_neg = result_neg
                all_controls.append(controls_neg)
            else:
                states_neg = result_neg
            
            all_states.append(states_neg)
        
        # Stack all paths
        if self.backend == 'numpy':
            states = np.stack(all_states, axis=0)
            controls = np.stack(all_controls, axis=0) if all_controls else None
        elif self.backend == 'torch':
            import torch
            states = torch.stack(all_states, dim=0)
            controls = torch.stack(all_controls, dim=0) if all_controls else None
        elif self.backend == 'jax':
            import jax.numpy as jnp
            states = jnp.stack(all_states, axis=0)
            controls = jnp.stack(all_controls, axis=0) if all_controls else None
        
        return MonteCarloResult(
            states=states,
            controls=controls,
            noise=None,
            n_paths=2*n_pairs,
            steps=steps
        )
    
    # ========================================================================
    # Utility and Information
    # ========================================================================
    
    def get_info(self) -> dict:
        """
        Get simulator configuration including noise info.
        
        Returns
        -------
        dict
            Configuration with stochastic properties
        """
        base_info = super().get_info()
        
        base_info['is_stochastic'] = True
        base_info['nw'] = self.nw
        base_info['seed'] = self.seed
        base_info['noise_type'] = self.system.get_noise_type().value if hasattr(self.system, 'get_noise_type') else 'unknown'
        
        return base_info
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        observer_str = f", observer={self.observer.__class__.__name__}" if self.observer else ""
        discretizer_str = f", discretizer={self.discretizer.method}" if self.discretizer else ""
        seed_str = f", seed={self.seed}" if self.seed is not None else ""
        
        return (
            f"StochasticDiscreteSimulator("
            f"system={self.system.__class__.__name__}"
            f"{discretizer_str}{observer_str}{seed_str})"
        )
    
    def __str__(self) -> str:
        """Human-readable string."""
        sys_type = "discrete-stochastic" if self.discretizer is None else "discretized-SDE"
        obs_str = " with observer" if self.observer else ""
        seed_str = f" (seed={self.seed})" if self.seed is not None else ""
        
        return f"StochasticDiscreteSimulator({self.system.__class__.__name__}, {sys_type}{obs_str}{seed_str})"


# ============================================================================
# Convenience Functions
# ============================================================================

def simulate_discrete_stochastic(
    system: 'DiscreteStochasticSystem',
    x0: ArrayLike,
    steps: int,
    controller: Optional[Union[ArrayLike, Callable, 'nn.Module']] = None,
    seed: Optional[int] = None,
    **kwargs
) -> ArrayLike:
    """
    Convenience function for quick discrete stochastic simulation.
    
    Examples
    --------
    >>> states = simulate_discrete_stochastic(ar1_system, x0, steps=100, seed=42)
    """
    sim = StochasticDiscreteSimulator(system, seed=seed)
    return sim.simulate(x0, steps, controller=controller, **kwargs)


def monte_carlo_discrete(
    system: 'DiscreteStochasticSystem',
    x0: ArrayLike,
    steps: int,
    n_paths: int,
    seed: Optional[int] = None,
    **kwargs
) -> MonteCarloResult:
    """
    Convenience function for Monte Carlo simulation.
    
    Examples
    --------
    >>> result = monte_carlo_discrete(ar1_system, x0, steps=100, n_paths=1000, seed=42)
    >>> stats = result.get_statistics()
    """
    sim = StochasticDiscreteSimulator(system, seed=seed)
    return sim.simulate_monte_carlo(x0, steps, n_paths, **kwargs)