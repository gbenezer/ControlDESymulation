"""
Custom Brownian path wrapper for Diffrax to support user-provided noise.

This allows deterministic testing and custom noise patterns.
"""

import jax.numpy as jnp
from jax import Array
import diffrax as dfx
from typing import Optional, Tuple


class CustomBrownianPath(dfx.AbstractPath):
    """
    Custom Brownian motion that uses provided dW increments.
    
    This implements Diffrax's AbstractPath interface to allow
    user-specified noise instead of generating random noise.
    
    Parameters
    ----------
    t0 : float
        Start time
    t1 : float
        End time
    dW : Array
        Brownian increment for interval (t0, t1)
        Shape: (nw,) for the noise dimensions
    
    Examples
    --------
    >>> # Zero noise for deterministic testing
    >>> dW = jnp.zeros(1)
    >>> brownian = CustomBrownianPath(0.0, 0.01, dW)
    >>> 
    >>> # Custom noise pattern
    >>> dW = jnp.array([0.5])
    >>> brownian = CustomBrownianPath(0.0, 0.01, dW)
    """
    
    def __init__(self, t0: float, t1: float, dW: Array):
        self.t0 = t0
        self.t1 = t1
        self.dW = dW
        self.dt = t1 - t0
        self._shape = dW.shape
    
    @property
    def t0(self) -> float:
        """Start time of the interval."""
        return self._t0
    
    @t0.setter
    def t0(self, value: float):
        self._t0 = value
    
    @property
    def t1(self) -> float:
        """End time of the interval."""
        return self._t1
    
    @t1.setter
    def t1(self, value: float):
        self._t1 = value
    
    def evaluate(
        self, 
        t0: float, 
        t1: Optional[float] = None, 
        left: bool = True
    ) -> Array:
        """
        Evaluate Brownian increment between t0 and t1.
        
        For custom noise, we provide the exact increment for our interval.
        Diffrax will call this to get dW values.
        
        Parameters
        ----------
        t0 : float
            Start time of query
        t1 : Optional[float]
            End time of query (if None, return value at t0)
        left : bool
            Whether to use left or right limit
            
        Returns
        -------
        Array
            Brownian increment or value
        """
        if t1 is None:
            # Query for B(t0) - return cumulative value
            # For simplicity, linear interpolation
            if jnp.abs(t0 - self._t0) < 1e-10:
                return jnp.zeros_like(self.dW)
            elif jnp.abs(t0 - self._t1) < 1e-10:
                return self.dW
            else:
                # Linear interpolation
                alpha = (t0 - self._t0) / self.dt
                return self.dW * alpha
        else:
            # Query for B(t1) - B(t0) = increment
            # Check if this is our full interval
            if (jnp.abs(t0 - self._t0) < 1e-10 and 
                jnp.abs(t1 - self._t1) < 1e-10):
                return self.dW
            else:
                # Sub-interval: scale proportionally by sqrt(time)
                dt_query = t1 - t0
                if self.dt > 0:
                    scale = jnp.sqrt(dt_query / self.dt)
                    return self.dW * scale
                else:
                    return jnp.zeros_like(self.dW)


def create_custom_or_random_brownian(
    key, 
    t0: float, 
    t1: float, 
    shape: Tuple[int, ...],
    dW: Optional[Array] = None
):
    """
    Create either custom or random Brownian motion for Diffrax.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key (used if dW is None)
    t0 : float
        Start time
    t1 : float
        End time  
    shape : tuple
        Noise shape (nw,)
    dW : Optional[Array]
        Custom Brownian increment. If None, generates random.
        
    Returns
    -------
    Brownian motion object for Diffrax
    
    Examples
    --------
    >>> # Random noise
    >>> key = jax.random.PRNGKey(42)
    >>> brownian = create_custom_or_random_brownian(key, 0, 0.01, (1,))
    >>> 
    >>> # Custom noise (deterministic)
    >>> dW = jnp.array([0.5])
    >>> brownian = create_custom_or_random_brownian(key, 0, 0.01, (1,), dW=dW)
    """
    if dW is not None:
        # Use custom noise
        return CustomBrownianPath(t0, t1, dW)
    else:
        # Use Diffrax's random noise generator
        return dfx.VirtualBrownianTree(
            t0, t1, tol=1e-3, shape=shape, key=key
        )