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
Monte Carlo Result Container

Generic container for Monte Carlo simulation results with statistical analysis.

This class provides a unified interface for storing and analyzing multiple
stochastic trajectories, regardless of whether they come from:
- Discrete stochastic systems
- Discretized continuous SDEs
- Uncertainty propagation
- Risk analysis
- Other Monte Carlo applications

Future Refactoring
------------------
TODO: Consider refactoring SDEIntegrationResult to inherit from a common base:

Current structure:
    - SDEIntegrationResult (in sde_integrator_base.py) - for continuous SDEs
    - MonteCarloResult (this file) - for discrete stochastic

Proposed refactoring:

    class MonteCarloResultBase(ABC):
        '''Common base for all Monte Carlo results.'''

        def __init__(self, states, n_paths):
            self.states = states
            self.n_paths = n_paths

        @abstractmethod
        def get_statistics(self):
            '''Compute statistics - implemented by subclasses.'''
            pass

    class SDEIntegrationResult(MonteCarloResultBase):
        '''Results from continuous SDE integration.'''

        def __init__(self, t, x, n_paths, ...):
            super().__init__(x, n_paths)
            self.t = t  # Continuous time points
            self.sde_type = sde_type
            self.diffusion_evals = diffusion_evals
            # SDE-specific metadata

    class MonteCarloResult(MonteCarloResultBase):
        '''Results from discrete stochastic simulation.'''

        def __init__(self, states, n_paths, ...):
            super().__init__(states, n_paths)
            self.controls = controls
            self.noise = noise
            # Discrete-specific metadata

Benefits of refactoring:
- Shared statistics implementation (DRY)
- Common interface for all MC results
- Easier to add new result types
- Type hints work better

Implementation effort: 1-2 hours
Priority: Low (current duplication is minimal)
Trigger: When adding 3rd type of MC result
"""

from typing import Callable, Dict, Optional, Union

import numpy as np

# Conditional imports
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Type alias
from src.types import ArrayLike


class MonteCarloResult:
    """
    Container for Monte Carlo simulation results.

    Stores multiple stochastic trajectories and provides statistical analysis
    across paths. Supports NumPy, PyTorch, and JAX backends.

    Attributes
    ----------
    states : ArrayLike
        All trajectories, shape (n_paths, steps+1, nx)
        First dimension: independent Monte Carlo paths
        Second dimension: time steps (includes initial state)
        Third dimension: state variables
    controls : Optional[ArrayLike]
        Control sequences for all paths, shape (n_paths, steps, nu)
        None if controls were not recorded
    noise : Optional[ArrayLike]
        Noise samples for all paths, shape (n_paths, steps, nw)
        None if noise was not recorded
    n_paths : int
        Number of Monte Carlo trajectories
    steps : int
        Number of time steps per trajectory

    Examples
    --------
    >>> # After Monte Carlo simulation
    >>> result = sim.simulate_monte_carlo(x0, steps=100, n_paths=1000)
    >>>
    >>> # Get statistics
    >>> stats = result.get_statistics()
    >>> mean_traj = stats['mean']  # (101, nx)
    >>> std_traj = stats['std']    # (101, nx)
    >>>
    >>> # Access individual trajectories
    >>> first_path = result.states[0]  # (101, nx)
    >>> last_path = result.states[-1]
    >>>
    >>> # Plot with confidence bands
    >>> import matplotlib.pyplot as plt
    >>> t = np.arange(101)
    >>> plt.plot(t, stats['mean'][:, 0], label='Mean')
    >>> plt.fill_between(t,
    ...                  stats['q25'][:, 0],
    ...                  stats['q75'][:, 0],
    ...                  alpha=0.3, label='IQR')
    """

    def __init__(
        self,
        states: ArrayLike,
        controls: Optional[ArrayLike] = None,
        noise: Optional[ArrayLike] = None,
        n_paths: int = 0,
        steps: int = 0,
        **metadata,
    ):
        """
        Initialize Monte Carlo result container.

        Parameters
        ----------
        states : ArrayLike
            Trajectory array (n_paths, steps+1, nx)
        controls : Optional[ArrayLike]
            Control array (n_paths, steps, nu)
        noise : Optional[ArrayLike]
            Noise array (n_paths, steps, nw)
        n_paths : int
            Number of paths
        steps : int
            Number of steps per path
        **metadata : dict
            Additional metadata (system name, parameters, etc.)

        Examples
        --------
        >>> result = MonteCarloResult(
        ...     states=states_array,
        ...     controls=controls_array,
        ...     n_paths=1000,
        ...     steps=100,
        ...     system_name='AR1',
        ...     seed=42
        ... )
        """
        self.states = states
        self.controls = controls
        self.noise = noise
        self.n_paths = n_paths
        self.steps = steps
        self.metadata = metadata

    def get_statistics(self, axis: int = 0) -> Dict[str, ArrayLike]:
        """
        Compute trajectory statistics across Monte Carlo paths.

        Computes mean, standard deviation, and quantiles across all
        trajectories at each time step.

        Parameters
        ----------
        axis : int
            Axis to compute statistics over (default: 0 = paths)

        Returns
        -------
        dict
            Statistics dictionary with keys:
            - 'mean': Mean trajectory, shape (steps+1, nx)
            - 'std': Standard deviation, shape (steps+1, nx)
            - 'var': Variance, shape (steps+1, nx)
            - 'min': Minimum values, shape (steps+1, nx)
            - 'max': Maximum values, shape (steps+1, nx)
            - 'median': Median trajectory, shape (steps+1, nx)
            - 'q25': 25th percentile, shape (steps+1, nx)
            - 'q75': 75th percentile, shape (steps+1, nx)
            - 'q05': 5th percentile, shape (steps+1, nx)
            - 'q95': 95th percentile, shape (steps+1, nx)

        Examples
        --------
        >>> stats = result.get_statistics()
        >>>
        >>> # Mean trajectory
        >>> mean_final = stats['mean'][-1]  # Mean at final time
        >>>
        >>> # Confidence intervals (95%)
        >>> ci_lower = stats['q05']
        >>> ci_upper = stats['q95']
        >>>
        >>> # Interquartile range
        >>> iqr = stats['q75'] - stats['q25']
        """
        if isinstance(self.states, np.ndarray):
            return {
                "mean": np.mean(self.states, axis=axis),
                "std": np.std(self.states, axis=axis),
                "var": np.var(self.states, axis=axis),
                "min": np.min(self.states, axis=axis),
                "max": np.max(self.states, axis=axis),
                "median": np.median(self.states, axis=axis),
                "q05": np.quantile(self.states, 0.05, axis=axis),
                "q25": np.quantile(self.states, 0.25, axis=axis),
                "q75": np.quantile(self.states, 0.75, axis=axis),
                "q95": np.quantile(self.states, 0.95, axis=axis),
            }

        elif TORCH_AVAILABLE and isinstance(self.states, torch.Tensor):
            return {
                "mean": torch.mean(self.states, dim=axis),
                "std": torch.std(self.states, dim=axis),
                "var": torch.var(self.states, dim=axis),
                "min": torch.min(self.states, dim=axis).values,
                "max": torch.max(self.states, dim=axis).values,
                "median": torch.median(self.states, dim=axis).values,
                "q05": torch.quantile(self.states, 0.05, dim=axis),
                "q25": torch.quantile(self.states, 0.25, dim=axis),
                "q75": torch.quantile(self.states, 0.75, dim=axis),
                "q95": torch.quantile(self.states, 0.95, dim=axis),
            }

        elif JAX_AVAILABLE and isinstance(self.states, jnp.ndarray):
            return {
                "mean": jnp.mean(self.states, axis=axis),
                "std": jnp.std(self.states, axis=axis),
                "var": jnp.var(self.states, axis=axis),
                "min": jnp.min(self.states, axis=axis),
                "max": jnp.max(self.states, axis=axis),
                "median": jnp.median(self.states, axis=axis),
                "q05": jnp.quantile(self.states, 0.05, axis=axis),
                "q25": jnp.quantile(self.states, 0.25, axis=axis),
                "q75": jnp.quantile(self.states, 0.75, axis=axis),
                "q95": jnp.quantile(self.states, 0.95, axis=axis),
            }

        else:
            raise TypeError(f"Unsupported array type: {type(self.states)}")

    def get_final_statistics(self) -> Dict[str, ArrayLike]:
        """
        Get statistics at final time only.

        Useful for terminal cost analysis or final state distribution.

        Returns
        -------
        dict
            Statistics at final time step, each with shape (nx,)

        Examples
        --------
        >>> final_stats = result.get_final_statistics()
        >>> print(f"Final mean: {final_stats['mean']}")
        >>> print(f"Final std: {final_stats['std']}")
        """
        all_stats = self.get_statistics()

        return {key: value[-1] for key, value in all_stats.items()}

    def get_path(self, path_idx: int) -> ArrayLike:
        """
        Get a single trajectory by index.

        Parameters
        ----------
        path_idx : int
            Path index (0 to n_paths-1)

        Returns
        -------
        ArrayLike
            Single trajectory, shape (steps+1, nx)

        Examples
        --------
        >>> # Get first trajectory
        >>> path_0 = result.get_path(0)
        >>>
        >>> # Get last trajectory
        >>> path_last = result.get_path(-1)
        """
        return self.states[path_idx]

    def get_paths_slice(self, start: int, end: int) -> ArrayLike:
        """
        Get a slice of trajectories.

        Parameters
        ----------
        start : int
            Start index
        end : int
            End index (exclusive)

        Returns
        -------
        ArrayLike
            Trajectories, shape (end-start, steps+1, nx)

        Examples
        --------
        >>> # Get first 100 paths
        >>> subset = result.get_paths_slice(0, 100)
        """
        return self.states[start:end]

    def compute_probability(
        self, condition: Callable[[ArrayLike], ArrayLike], time_step: Optional[int] = None
    ) -> float:
        """
        Estimate probability of an event via Monte Carlo.

        Parameters
        ----------
        condition : Callable
            Function that returns True/False for each path
            Signature: condition(state) -> bool or array of bools
        time_step : Optional[int]
            Time step to check (None = final time)

        Returns
        -------
        float
            Estimated probability (fraction of paths satisfying condition)

        Examples
        --------
        >>> # Probability of being in region at final time
        >>> def in_region(x):
        ...     return np.linalg.norm(x) < 1.0
        >>>
        >>> prob = result.compute_probability(in_region)
        >>> print(f"P(||x|| < 1) ≈ {prob:.3f}")
        >>>
        >>> # Probability at specific time
        >>> prob_t50 = result.compute_probability(in_region, time_step=50)
        """
        if time_step is None:
            time_step = -1  # Final time

        # Get states at specified time
        states_at_t = self.states[:, time_step, :]

        # Evaluate condition for each path
        satisfies = np.array([condition(states_at_t[i]) for i in range(self.n_paths)])

        # Return fraction
        return np.mean(satisfies)

    def compute_expectation(
        self, function: Callable[[ArrayLike], float], time_step: Optional[int] = None
    ) -> float:
        """
        Compute expected value of a function via Monte Carlo.

        Estimates E[f(x)] by averaging f(x) over all paths.

        Parameters
        ----------
        function : Callable
            Function to compute expectation of
            Signature: function(state) -> scalar
        time_step : Optional[int]
            Time step to evaluate (None = final time)

        Returns
        -------
        float
            Estimated E[f(x)]

        Examples
        --------
        >>> # Expected energy at final time
        >>> def energy(x):
        ...     return 0.5 * np.sum(x**2)
        >>>
        >>> expected_energy = result.compute_expectation(energy)
        >>> print(f"E[energy] ≈ {expected_energy:.3f}")
        >>>
        >>> # Expected cost over time
        >>> costs = [result.compute_expectation(energy, t) for t in range(101)]
        """
        if time_step is None:
            time_step = -1

        states_at_t = self.states[:, time_step, :]

        # Evaluate function for each path
        values = np.array([function(states_at_t[i]) for i in range(self.n_paths)])

        return np.mean(values)

    def __repr__(self) -> str:
        """String representation for debugging."""
        has_controls = self.controls is not None
        has_noise = self.noise is not None

        return (
            f"MonteCarloResult("
            f"n_paths={self.n_paths}, "
            f"steps={self.steps}, "
            f"controls={has_controls}, "
            f"noise={has_noise})"
        )

    def __str__(self) -> str:
        """Human-readable string."""
        return f"MonteCarloResult: {self.n_paths} paths, {self.steps} steps each"

    def __len__(self) -> int:
        """Return number of paths."""
        return self.n_paths

    def __getitem__(self, idx: int) -> ArrayLike:
        """
        Get trajectory by index (allows result[i] syntax).

        Examples
        --------
        >>> path_0 = result[0]
        >>> path_5 = result[5]
        """
        return self.get_path(idx)


# ============================================================================
# Utility Functions
# ============================================================================


def combine_results(results: list) -> MonteCarloResult:
    """
    Combine multiple MonteCarloResult objects.

    Useful for distributed Monte Carlo or combining results from
    multiple simulation runs.

    Parameters
    ----------
    results : list
        List of MonteCarloResult objects to combine

    Returns
    -------
    MonteCarloResult
        Combined result with all paths

    Raises
    ------
    ValueError
        If results have incompatible shapes or backends

    Examples
    --------
    >>> # Run MC on multiple machines
    >>> result1 = sim.simulate_monte_carlo(x0, 100, n_paths=500)
    >>> result2 = sim.simulate_monte_carlo(x0, 100, n_paths=500)
    >>>
    >>> # Combine
    >>> combined = combine_results([result1, result2])
    >>> combined.n_paths
    1000
    """
    if not results:
        raise ValueError("Cannot combine empty list of results")

    # Validate compatibility
    first = results[0]
    steps = first.steps

    for r in results[1:]:
        if r.steps != steps:
            raise ValueError(f"Incompatible steps: {r.steps} vs {steps}")
        if r.states.shape[1:] != first.states.shape[1:]:
            raise ValueError(f"Incompatible state shapes: {r.states.shape} vs {first.states.shape}")

    # Concatenate all paths
    if isinstance(first.states, np.ndarray):
        all_states = np.concatenate([r.states for r in results], axis=0)
        all_controls = (
            np.concatenate([r.controls for r in results], axis=0)
            if first.controls is not None
            else None
        )
        all_noise = (
            np.concatenate([r.noise for r in results], axis=0) if first.noise is not None else None
        )

    elif TORCH_AVAILABLE and isinstance(first.states, torch.Tensor):
        import torch

        all_states = torch.cat([r.states for r in results], dim=0)
        all_controls = (
            torch.cat([r.controls for r in results], dim=0) if first.controls is not None else None
        )
        all_noise = (
            torch.cat([r.noise for r in results], dim=0) if first.noise is not None else None
        )

    elif JAX_AVAILABLE and isinstance(first.states, jnp.ndarray):
        import jax.numpy as jnp

        all_states = jnp.concatenate([r.states for r in results], axis=0)
        all_controls = (
            jnp.concatenate([r.controls for r in results], axis=0)
            if first.controls is not None
            else None
        )
        all_noise = (
            jnp.concatenate([r.noise for r in results], axis=0) if first.noise is not None else None
        )

    total_paths = sum(r.n_paths for r in results)

    return MonteCarloResult(
        states=all_states,
        controls=all_controls,
        noise=all_noise,
        n_paths=total_paths,
        steps=steps,
        **first.metadata,
    )


def save_monte_carlo_result(result: MonteCarloResult, filename: str):
    """
    Save Monte Carlo result to file.

    Parameters
    ----------
    result : MonteCarloResult
        Result to save
    filename : str
        Output filename (.npz for NumPy, .pt for PyTorch)

    Examples
    --------
    >>> save_monte_carlo_result(result, 'mc_simulation.npz')
    >>>
    >>> # Load later
    >>> loaded = load_monte_carlo_result('mc_simulation.npz')
    """
    if filename.endswith(".npz"):
        # NumPy format
        save_dict = {
            "states": np.asarray(result.states),
            "n_paths": result.n_paths,
            "steps": result.steps,
        }

        if result.controls is not None:
            save_dict["controls"] = np.asarray(result.controls)

        if result.noise is not None:
            save_dict["noise"] = np.asarray(result.noise)

        np.savez(filename, **save_dict)

    elif filename.endswith(".pt"):
        # PyTorch format
        import torch

        save_dict = {
            "states": torch.as_tensor(result.states),
            "n_paths": result.n_paths,
            "steps": result.steps,
        }

        if result.controls is not None:
            save_dict["controls"] = torch.as_tensor(result.controls)

        if result.noise is not None:
            save_dict["noise"] = torch.as_tensor(result.noise)

        torch.save(save_dict, filename)

    else:
        raise ValueError(f"Unsupported format: {filename}. Use .npz or .pt")


def load_monte_carlo_result(filename: str) -> MonteCarloResult:
    """
    Load Monte Carlo result from file.

    Parameters
    ----------
    filename : str
        Input filename

    Returns
    -------
    MonteCarloResult
        Loaded result

    Examples
    --------
    >>> result = load_monte_carlo_result('mc_simulation.npz')
    >>> stats = result.get_statistics()
    """
    if filename.endswith(".npz"):
        data = np.load(filename)

        return MonteCarloResult(
            states=data["states"],
            controls=data["controls"] if "controls" in data else None,
            noise=data["noise"] if "noise" in data else None,
            n_paths=int(data["n_paths"]),
            steps=int(data["steps"]),
        )

    elif filename.endswith(".pt"):
        import torch

        data = torch.load(filename)

        return MonteCarloResult(
            states=data["states"],
            controls=data.get("controls"),
            noise=data.get("noise"),
            n_paths=data["n_paths"],
            steps=data["steps"],
        )

    else:
        raise ValueError(f"Unsupported format: {filename}")
