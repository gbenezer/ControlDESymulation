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
Discrete Linearization - Cached Linearization for Discrete Systems

Provides caching layer for numerical linearization of discrete-time systems.

This class computes and caches discrete-time Jacobian matrices (Ad, Bd)
at equilibrium or operating points, avoiding expensive recomputation.

Use Cases
---------
- LQR controller design (need A, B at equilibrium)
- Gain scheduling (linearize at multiple operating points)
- Stability analysis (eigenvalues of Ad)
- Model predictive control (repeated linearization)
- Sequential linearization in optimization loops

Architecture
-----------
Thin caching layer that delegates to:
- DiscreteSymbolicSystem.linearized_dynamics() for pure discrete
- Discretizer.linearize() for discretized continuous systems

The caching strategy uses equilibrium name or state/control hash as key.

Examples
--------
>>> # Pure discrete system
>>> discrete_system = DiscreteLinearSystem()
>>> lin = DiscreteLinearization(discrete_system)
>>>
>>> # First call computes
>>> Ad, Bd = lin.compute(x_eq, u_eq)
>>>
>>> # Second call uses cache
>>> Ad2, Bd2 = lin.compute(x_eq, u_eq)
>>> # Returns same objects (cached)
>>>
>>> # Discretized continuous system
>>> continuous_system = Pendulum()
>>> discretizer = Discretizer(continuous_system, dt=0.01, method='rk4')
>>> lin = DiscreteLinearization(continuous_system, discretizer=discretizer)
>>>
>>> # Linearize at equilibrium
>>> discrete_system.add_equilibrium('upright', x_eq, u_eq)
>>> Ad, Bd = lin.compute('upright')  # Uses equilibrium name
>>>
>>> # Multiple operating points (gain scheduling)
>>> operating_points = [
...     (np.array([0.1, 0.0]), np.array([0.0])),
...     (np.array([0.5, 0.0]), np.array([0.0])),
...     (np.array([1.0, 0.0]), np.array([0.0])),
... ]
>>>
>>> linearizations = {}
>>> for x, u in operating_points:
...     Ad, Bd = lin.compute(x, u)
...     linearizations[(tuple(x), tuple(u))] = (Ad, Bd)
"""

import hashlib
from typing import Any, Dict, Optional, Tuple, Union

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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.systems.base.discrete_symbolic_system import DiscreteSymbolicSystem
    from src.systems.base.discretization.discretizer import Discretizer
    from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem

# Type alias
ArrayLike = Union[np.ndarray, "torch.Tensor", "jnp.ndarray"]


class DiscreteLinearization:
    """
    Caches numerical linearization for discrete-time systems.

    Computes and caches discrete-time Jacobian matrices (Ad, Bd) at
    equilibrium or operating points to avoid expensive recomputation.

    Works with both:
    - Pure discrete systems (DiscreteSymbolicSystem)
    - Discretized continuous systems (via Discretizer)

    Attributes
    ----------
    system : Union[DiscreteSymbolicSystem, SymbolicDynamicalSystem]
        System to linearize
    discretizer : Optional[Discretizer]
        Discretizer for continuous systems
    backend : str
        Backend for numerical arrays

    Examples
    --------
    >>> system = DiscreteLinearSystem()
    >>> lin = DiscreteLinearization(system)
    >>>
    >>> # Compute at equilibrium
    >>> Ad, Bd = lin.compute(x_eq, u_eq)
    >>>
    >>> # Second call returns cached
    >>> Ad2, Bd2 = lin.compute(x_eq, u_eq)
    >>> assert Ad is Ad2  # Same object (cached)
    >>>
    >>> # Clear cache
    >>> lin.reset_cache()
    """

    def __init__(
        self,
        system: Union["DiscreteSymbolicSystem", "SymbolicDynamicalSystem"],
        discretizer: Optional["Discretizer"] = None,
    ):
        """
        Initialize discrete linearization cache.

        Parameters
        ----------
        system : Union[DiscreteSymbolicSystem, SymbolicDynamicalSystem]
            System to linearize
        discretizer : Optional[Discretizer]
            Required for continuous systems, None for discrete

        Raises
        ------
        TypeError
            If continuous system without discretizer
        """
        self.system = system
        self.discretizer = discretizer

        # Validate system/discretizer compatibility
        from src.systems.base.discrete_symbolic_system import DiscreteSymbolicSystem

        is_discrete = isinstance(system, DiscreteSymbolicSystem) or getattr(
            system, "_is_discrete", False
        )

        if not is_discrete and discretizer is None:
            raise TypeError(
                f"Continuous system {system.__class__.__name__} requires a Discretizer. "
                f"Provide discretizer parameter."
            )

        # Check if stochastic system and warn
        is_stochastic = False
        try:
            from src.systems.base.discrete_stochastic_system import DiscreteStochasticSystem

            if isinstance(system, DiscreteStochasticSystem):
                is_stochastic = True
        except ImportError:
            pass

        try:
            from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem

            if isinstance(system, StochasticDynamicalSystem):
                is_stochastic = True
        except ImportError:
            pass

        if is_stochastic:
            import warnings

            warnings.warn(
                f"Using DiscreteLinearization with stochastic system {system.__class__.__name__}. "
                f"Consider using StochasticDiscreteLinearization for proper handling of stochastic dynamics.",
                UserWarning,
                stacklevel=2,
            )

        # Cache storage
        self._cache: Dict[str, Tuple[ArrayLike, ArrayLike]] = {}

        # Cache statistics
        self._stats = {
            "computes": 0,
            "cache_hits": 0,
        }

        # Backend
        self.backend = system._default_backend

    # ========================================================================
    # Main Linearization API
    # ========================================================================

    def compute(
        self,
        x_eq: Union[ArrayLike, str],
        u_eq: Optional[ArrayLike] = None,
        method: str = "euler",
        use_cache: bool = True,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Compute discrete-time linearization: (Ad, Bd).

        Parameters
        ----------
        x_eq : Union[ArrayLike, str]
            Equilibrium state (nx,) or equilibrium name
        u_eq : Optional[ArrayLike]
            Equilibrium control (nu,)
            Ignored if x_eq is string (equilibrium name)
        method : str
            Discretization method for continuous systems
            ('euler', 'exact', 'tustin', 'matched')
            Ignored for pure discrete systems
        use_cache : bool
            If True, use cached result if available

        Returns
        -------
        Ad : ArrayLike
            Discrete-time state matrix (nx, nx)
        Bd : ArrayLike
            Discrete-time control matrix (nx, nu)

        Examples
        --------
        >>> # At specific point
        >>> Ad, Bd = lin.compute(
        ...     x_eq=np.array([0.0, 0.0]),
        ...     u_eq=np.array([0.0])
        ... )
        >>>
        >>> # At named equilibrium
        >>> system.add_equilibrium('origin', x_eq, u_eq)
        >>> Ad, Bd = lin.compute('origin')
        >>>
        >>> # Force recomputation
        >>> Ad, Bd = lin.compute(x_eq, u_eq, use_cache=False)

        Notes
        -----
        For pure discrete systems, method parameter is ignored.
        For discretized continuous systems, method determines how
        continuous-time Jacobians are converted to discrete-time.
        """
        # Handle equilibrium name
        if isinstance(x_eq, str):
            equilibrium_name = x_eq
            x_eq, u_eq = self.system.equilibria.get_both(equilibrium_name, backend=self.backend)
            cache_key = equilibrium_name
        else:
            # Generate cache key from state/control
            cache_key = self._generate_cache_key(x_eq, u_eq, method)

        # Check cache
        if use_cache and cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]

        # Compute linearization
        if self.discretizer is not None:
            # Discretized continuous system
            Ad, Bd = self.discretizer.linearize(x_eq, u_eq, method=method)
        else:
            # Pure discrete system - linearization is direct
            Ad, Bd = self.system.linearized_dynamics(x_eq, u_eq, backend=self.backend)

        # Cache it
        self._cache[cache_key] = (Ad, Bd)
        self._stats["computes"] += 1

        return Ad, Bd

    def compute_at_equilibria(
        self,
        equilibrium_names: Optional[list] = None,
        method: str = "euler",
    ) -> Dict[str, Tuple[ArrayLike, ArrayLike]]:
        """
        Compute linearization at all (or specified) equilibria.

        Useful for:
        - Gain scheduling
        - Stability analysis at multiple points
        - Precomputing linearizations for control design

        Parameters
        ----------
        equilibrium_names : Optional[list]
            List of equilibrium names to linearize at
            If None, uses all equilibria
        method : str
            Discretization method

        Returns
        -------
        dict
            Mapping equilibrium_name -> (Ad, Bd)

        Examples
        --------
        >>> # Add multiple equilibria
        >>> system.add_equilibrium('down', x_down, u_down)
        >>> system.add_equilibrium('up', x_up, u_up)
        >>>
        >>> # Linearize at all
        >>> linearizations = lin.compute_at_equilibria()
        >>> Ad_down, Bd_down = linearizations['down']
        >>> Ad_up, Bd_up = linearizations['up']
        >>>
        >>> # Check stability
        >>> eigs_down = np.linalg.eigvals(Ad_down)
        >>> eigs_up = np.linalg.eigvals(Ad_up)
        """
        if equilibrium_names is None:
            equilibrium_names = self.system.list_equilibria()

        linearizations = {}

        for name in equilibrium_names:
            Ad, Bd = self.compute(name, method=method)
            linearizations[name] = (Ad, Bd)

        return linearizations

    # ========================================================================
    # Cache Management
    # ========================================================================

    def _generate_cache_key(self, x_eq: ArrayLike, u_eq: Optional[ArrayLike], method: str) -> str:
        """
        Generate cache key from state, control, and method.

        Uses hash of array values for consistent key generation.

        Parameters
        ----------
        x_eq : ArrayLike
            Equilibrium state
        u_eq : Optional[ArrayLike]
            Equilibrium control
        method : str
            Discretization method

        Returns
        -------
        str
            Cache key
        """
        # Convert to numpy for hashing
        x_np = np.asarray(x_eq)
        u_np = np.asarray(u_eq) if u_eq is not None else np.array([])

        # Create hash from values
        combined = np.concatenate([x_np.flatten(), u_np.flatten()])
        value_hash = hashlib.md5(combined.tobytes()).hexdigest()[:8]

        return f"{value_hash}_{method}"

    def reset_cache(self, equilibrium_name: Optional[str] = None):
        """
        Clear linearization cache.

        Parameters
        ----------
        equilibrium_name : Optional[str]
            If provided, clear only this equilibrium
            If None, clear entire cache

        Examples
        --------
        >>> # Clear everything
        >>> lin.reset_cache()
        >>>
        >>> # Clear specific equilibrium
        >>> lin.reset_cache('upright')
        """
        if equilibrium_name is not None:
            # Clear specific equilibrium
            if equilibrium_name in self._cache:
                del self._cache[equilibrium_name]
        else:
            # Clear all
            self._cache.clear()

    def is_cached(
        self, x_eq: Union[ArrayLike, str], u_eq: Optional[ArrayLike] = None, method: str = "euler"
    ) -> bool:
        """
        Check if linearization is cached.

        Parameters
        ----------
        x_eq : Union[ArrayLike, str]
            State or equilibrium name
        u_eq : Optional[ArrayLike]
            Control
        method : str
            Discretization method

        Returns
        -------
        bool
            True if cached

        Examples
        --------
        >>> if not lin.is_cached('origin'):
        ...     Ad, Bd = lin.compute('origin')
        """
        if isinstance(x_eq, str):
            cache_key = x_eq
        else:
            cache_key = self._generate_cache_key(x_eq, u_eq, method)

        return cache_key in self._cache

    def get_cached(
        self, x_eq: Union[ArrayLike, str], u_eq: Optional[ArrayLike] = None, method: str = "euler"
    ) -> Optional[Tuple[ArrayLike, ArrayLike]]:
        """
        Get cached linearization without computing.

        Parameters
        ----------
        x_eq : Union[ArrayLike, str]
            State or equilibrium name
        u_eq : Optional[ArrayLike]
            Control
        method : str
            Method

        Returns
        -------
        Optional[Tuple[ArrayLike, ArrayLike]]
            (Ad, Bd) if cached, None otherwise

        Examples
        --------
        >>> cached = lin.get_cached('origin')
        >>> if cached is not None:
        ...     Ad, Bd = cached
        ... else:
        ...     Ad, Bd = lin.compute('origin')
        """
        if isinstance(x_eq, str):
            cache_key = x_eq
        else:
            cache_key = self._generate_cache_key(x_eq, u_eq, method)

        return self._cache.get(cache_key)

    # ========================================================================
    # Stability Analysis
    # ========================================================================

    def check_stability(
        self, x_eq: Union[ArrayLike, str], u_eq: Optional[ArrayLike] = None, method: str = "euler"
    ) -> Dict[str, Any]:
        """
        Check stability of linearized system.

        Computes eigenvalues of Ad and determines stability.

        Parameters
        ----------
        x_eq : Union[ArrayLike, str]
            Equilibrium state or name
        u_eq : Optional[ArrayLike]
            Equilibrium control
        method : str
            Discretization method

        Returns
        -------
        dict
            Stability analysis with keys:
            - 'eigenvalues': Eigenvalues of Ad
            - 'max_magnitude': Maximum eigenvalue magnitude
            - 'is_stable': True if all |λ| < 1
            - 'is_marginally_stable': True if max|λ| ≈ 1
            - 'is_unstable': True if any |λ| > 1

        Examples
        --------
        >>> stability = lin.check_stability('origin')
        >>>
        >>> if stability['is_stable']:
        ...     print("Equilibrium is stable!")
        ... else:
        ...     print(f"Unstable: max|λ|={stability['max_magnitude']:.3f}")
        >>>
        >>> # Visualize eigenvalues
        >>> eigs = stability['eigenvalues']
        >>> import matplotlib.pyplot as plt
        >>> plt.scatter(eigs.real, eigs.imag)
        >>> plt.gca().add_patch(plt.Circle((0,0), 1, fill=False))
        """
        Ad, Bd = self.compute(x_eq, u_eq, method)

        # Convert to numpy for eigenvalue computation
        if TORCH_AVAILABLE and isinstance(Ad, torch.Tensor):
            Ad_np = Ad.detach().cpu().numpy()
        elif JAX_AVAILABLE and isinstance(Ad, jnp.ndarray):
            Ad_np = np.array(Ad)
        else:
            Ad_np = Ad

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(Ad_np)
        magnitudes = np.abs(eigenvalues)
        max_magnitude = np.max(magnitudes)

        # Stability determination
        # Use bool() to convert numpy bool to Python bool
        is_stable = bool(max_magnitude < 1.0)
        is_marginally_stable = bool(np.abs(max_magnitude - 1.0) < 1e-6)
        is_unstable = bool(max_magnitude > 1.0)

        return {
            "eigenvalues": eigenvalues,
            "magnitudes": magnitudes,
            "max_magnitude": float(max_magnitude),
            "is_stable": is_stable,
            "is_marginally_stable": is_marginally_stable,
            "is_unstable": is_unstable,
            "spectral_radius": float(max_magnitude),
        }

    def compute_controllability_matrix(
        self, x_eq: Union[ArrayLike, str], u_eq: Optional[ArrayLike] = None, method: str = "euler"
    ) -> ArrayLike:
        """
        Compute discrete-time controllability matrix.

        C = [Bd, Ad*Bd, Ad²*Bd, ..., Ad^(n-1)*Bd]

        Parameters
        ----------
        x_eq : Union[ArrayLike, str]
            Equilibrium
        u_eq : Optional[ArrayLike]
            Control
        method : str
            Method

        Returns
        -------
        ArrayLike
            Controllability matrix (nx, nx*nu)

        Examples
        --------
        >>> C = lin.compute_controllability_matrix('origin')
        >>> rank = np.linalg.matrix_rank(C)
        >>>
        >>> if rank == system.nx:
        ...     print("System is controllable!")
        """
        Ad, Bd = self.compute(x_eq, u_eq, method)

        # Convert to numpy for matrix operations
        if TORCH_AVAILABLE and isinstance(Ad, torch.Tensor):
            Ad_np = Ad.detach().cpu().numpy()
            Bd_np = Bd.detach().cpu().numpy()
        elif JAX_AVAILABLE and isinstance(Ad, jnp.ndarray):
            Ad_np = np.array(Ad)
            Bd_np = np.array(Bd)
        else:
            Ad_np = Ad
            Bd_np = Bd

        nx = Ad_np.shape[0]
        nu = Bd_np.shape[1]

        # Build controllability matrix
        C = np.zeros((nx, nx * nu))

        Ad_power = np.eye(nx)
        for i in range(nx):
            C[:, i * nu : (i + 1) * nu] = Ad_power @ Bd_np
            Ad_power = Ad_power @ Ad_np

        # Convert back to original backend
        if TORCH_AVAILABLE and isinstance(Ad, torch.Tensor):
            return torch.tensor(C, dtype=Ad.dtype, device=Ad.device)
        elif JAX_AVAILABLE and isinstance(Ad, jnp.ndarray):
            return jnp.array(C)
        else:
            return C

    def compute_observability_matrix(
        self, x_eq: Union[ArrayLike, str], u_eq: Optional[ArrayLike] = None, method: str = "euler"
    ) -> ArrayLike:
        """
        Compute discrete-time observability matrix.

        O = [Cd; Cd*Ad; Cd*Ad²; ...; Cd*Ad^(n-1)]

        Parameters
        ----------
        x_eq : Union[ArrayLike, str]
            Equilibrium
        u_eq : Optional[ArrayLike]
            Control
        method : str
            Method

        Returns
        -------
        ArrayLike
            Observability matrix (nx*ny, nx)

        Examples
        --------
        >>> O = lin.compute_observability_matrix('origin')
        >>> rank = np.linalg.matrix_rank(O)
        >>>
        >>> if rank == system.nx:
        ...     print("System is observable!")
        """
        Ad, Bd = self.compute(x_eq, u_eq, method)

        # Get observation Jacobian
        if self.discretizer is not None:
            Cd = self.discretizer.linearized_observation(x_eq, backend=self.backend)
        else:
            Cd = self.system.linearized_observation(x_eq, backend=self.backend)

        # Convert to numpy
        if TORCH_AVAILABLE and isinstance(Ad, torch.Tensor):
            Ad_np = Ad.detach().cpu().numpy()
            Cd_np = Cd.detach().cpu().numpy()
        elif JAX_AVAILABLE and isinstance(Ad, jnp.ndarray):
            Ad_np = np.array(Ad)
            Cd_np = np.array(Cd)
        else:
            Ad_np = Ad
            Cd_np = Cd

        nx = Ad_np.shape[0]
        ny = Cd_np.shape[0]

        # Build observability matrix
        O = np.zeros((nx * ny, nx))

        Ad_power = np.eye(nx)
        for i in range(nx):
            O[i * ny : (i + 1) * ny, :] = Cd_np @ Ad_power
            Ad_power = Ad_power @ Ad_np

        # Convert back
        if TORCH_AVAILABLE and isinstance(Ad, torch.Tensor):
            return torch.tensor(O, dtype=Ad.dtype, device=Ad.device)
        elif JAX_AVAILABLE and isinstance(Ad, jnp.ndarray):
            return jnp.array(O)
        else:
            return O

    # ========================================================================
    # Cache Statistics and Management
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
        dict
            Statistics including:
            - 'computes': Number of linearizations computed
            - 'cache_hits': Number of cache hits
            - 'cache_size': Number of cached linearizations
            - 'hit_rate': Fraction of requests served from cache

        Examples
        --------
        >>> stats = lin.get_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
        >>> print(f"Cache size: {stats['cache_size']}")
        """
        total_requests = self._stats["computes"] + self._stats["cache_hits"]
        hit_rate = self._stats["cache_hits"] / max(1, total_requests)

        return {
            "computes": self._stats["computes"],
            "cache_hits": self._stats["cache_hits"],
            "total_requests": total_requests,
            "cache_size": len(self._cache),
            "hit_rate": hit_rate,
        }

    def reset_stats(self):
        """Reset cache statistics (but keep cached values)."""
        self._stats["computes"] = 0
        self._stats["cache_hits"] = 0

    def list_cached(self) -> list:
        """
        List all cached equilibrium/point identifiers.

        Returns
        -------
        list
            Cache keys (equilibrium names or hashes)

        Examples
        --------
        >>> cached = lin.list_cached()
        >>> print(f"Cached: {cached}")
        ['origin', 'upright', 'a3f5b2c1_euler']
        """
        return list(self._cache.keys())

    def clear_cache(self):
        """
        Clear entire cache and reset statistics.

        Examples
        --------
        >>> lin.clear_cache()
        >>> lin.get_stats()['cache_size']
        0
        """
        self._cache.clear()
        self.reset_stats()

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def precompute_at_grid(self, x_grid: ArrayLike, u_grid: ArrayLike, method: str = "euler"):
        """
        Precompute linearizations at grid of operating points.

        Useful for gain scheduling - precompute once, use many times.

        Parameters
        ----------
        x_grid : ArrayLike
            Grid of state points, shape (n_points, nx)
        u_grid : ArrayLike
            Grid of control points, shape (n_points, nu)
        method : str
            Discretization method

        Examples
        --------
        >>> # Create operating point grid
        >>> x_grid = np.array([[0.1, 0], [0.5, 0], [1.0, 0]])
        >>> u_grid = np.zeros((3, 1))
        >>>
        >>> # Precompute
        >>> lin.precompute_at_grid(x_grid, u_grid)
        >>>
        >>> # Later lookups are cached
        >>> Ad, Bd = lin.compute(x_grid[0], u_grid[0])  # Cache hit!
        """
        n_points = x_grid.shape[0]

        for i in range(n_points):
            x = x_grid[i]
            u = u_grid[i]
            self.compute(x, u, method=method, use_cache=True)

        print(f"Precomputed {n_points} linearizations")

    def get_info(self) -> Dict[str, Any]:
        """
        Get linearization cache information.

        Returns
        -------
        dict
            Information about cache state

        Examples
        --------
        >>> info = lin.get_info()
        >>> print(info)
        """
        return {
            "system": self.system.__class__.__name__,
            "system_type": "discrete" if self.discretizer is None else "discretized",
            "discretizer": self.discretizer.method if self.discretizer else None,
            "backend": self.backend,
            "cache_size": len(self._cache),
            "statistics": self.get_stats(),
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        discretizer_str = f", discretizer={self.discretizer.method}" if self.discretizer else ""
        return (
            f"DiscreteLinearization("
            f"system={self.system.__class__.__name__}"
            f"{discretizer_str}, "
            f"cache_size={len(self._cache)})"
        )

    def __str__(self) -> str:
        """Human-readable string."""
        sys_type = "discrete" if self.discretizer is None else "discretized"
        return (
            f"DiscreteLinearization({self.system.__class__.__name__}, "
            f"{sys_type}, {len(self._cache)} cached)"
        )
