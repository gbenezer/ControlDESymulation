"""
Dynamics Evaluator for SymbolicDynamicalSystem

Handles forward dynamics evaluation across multiple backends.

Responsibilities:
- Forward dynamics evaluation: dx/dt = f(x, u)
- Backend-specific implementations (NumPy, PyTorch, JAX)
- Input validation and shape handling
- Batched vs single evaluation
- Performance tracking
- Backend dispatch

This class manages the evaluation of the system dynamics using
generated functions from CodeGenerator.
"""

from typing import Optional, TYPE_CHECKING
import time
import numpy as np

if TYPE_CHECKING:
    import torch
    import jax.numpy as jnp
    from src.systems.base.symbolic_dynamical_system import SymbolicDynamicalSystem
    from src.systems.base.utils.code_generator import CodeGenerator
    from src.systems.base.utils.backend_manager import BackendManager

# Type alias
from typing import Union
ArrayLike = Union[np.ndarray, "torch.Tensor", "jnp.ndarray"]


class DynamicsEvaluator:
    """
    Evaluates forward dynamics across backends.
    
    Handles the evaluation of dx/dt = f(x, u) for NumPy, PyTorch, and JAX
    backends with proper shape handling, batching, and performance tracking.
    
    Example:
        >>> evaluator = DynamicsEvaluator(system, code_gen, backend_mgr)
        >>> dx = evaluator.evaluate(x, u, backend='numpy')
        >>> 
        >>> # Get performance stats
        >>> stats = evaluator.get_stats()
        >>> print(f"Average time: {stats['avg_time']:.6f}s")
    """
    
    def __init__(
        self,
        system: 'SymbolicDynamicalSystem',
        code_gen: 'CodeGenerator',
        backend_mgr: 'BackendManager'
    ):
        """
        Initialize dynamics evaluator.
        
        Args:
            system: The dynamical system
            code_gen: Code generator for accessing compiled functions
            backend_mgr: Backend manager for detection/conversion
        """
        self.system = system
        self.code_gen = code_gen
        self.backend_mgr = backend_mgr
        
        # Performance tracking
        self._stats = {
            'calls': 0,
            'time': 0.0,
        }
    
    # ========================================================================
    # Main Evaluation API
    # ========================================================================
    
    def evaluate(self, x: ArrayLike, u: ArrayLike, backend: Optional[str] = None) -> ArrayLike:
        """
        Evaluate forward dynamics: dx/dt = f(x, u).
        
        Args:
            x: State (array/tensor)
            u: Control (array/tensor)
            backend: Backend selection:
                - None: Auto-detect from input type (default)
                - 'numpy', 'torch', 'jax': Force specific backend
                - 'default': Use system's default backend
                
        Returns:
            State derivative (type matches backend)
            
        Example:
            >>> # Auto-detect backend
            >>> dx = evaluator.evaluate(x_numpy, u_numpy)  # Returns NumPy
            >>> 
            >>> # Force specific backend (converts input)
            >>> dx = evaluator.evaluate(x_numpy, u_numpy, backend='torch')  # Returns PyTorch
        """
        # Determine target backend
        if backend == 'default':
            target_backend = self.backend_mgr.default_backend
        elif backend is None:
            target_backend = self.backend_mgr.detect(x)
        else:
            target_backend = backend
        
        # Convert inputs if needed
        input_backend = self.backend_mgr.detect(x)
        if input_backend != target_backend:
            x = self.backend_mgr.convert(x, target_backend)
            u = self.backend_mgr.convert(u, target_backend)
        
        # Dispatch to backend-specific implementation
        if target_backend == 'numpy':
            return self._evaluate_numpy(x, u)
        elif target_backend == 'torch':
            return self._evaluate_torch(x, u)
        elif target_backend == 'jax':
            return self._evaluate_jax(x, u)
        else:
            raise ValueError(f"Unknown backend: {target_backend}")
    
    # ========================================================================
    # Backend-Specific Implementations
    # ========================================================================
    
    def _evaluate_numpy(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        NumPy backend implementation.
        
        Handles both single and batched evaluation.
        """
        start_time = time.time()
        
        # Input validation
        if x.ndim == 0 or u.ndim == 0:
            raise ValueError("Input arrays must be at least 1D")
        
        if x.ndim >= 1 and x.shape[-1] != self.system.nx:
            raise ValueError(
                f"Expected state dimension {self.system.nx}, got {x.shape[-1]}"
            )
        if u.ndim >= 1 and u.shape[-1] != self.system.nu:
            raise ValueError(
                f"Expected control dimension {self.system.nu}, got {u.shape[-1]}"
            )
        
        # Generate function (uses cache if available)
        f_numpy = self.code_gen.generate_dynamics('numpy')
        
        # Handle batched vs single evaluation
        if x.ndim == 1:
            # Single evaluation
            x_list = [x[i] for i in range(self.system.nx)]
            u_list = [u[i] for i in range(self.system.nu)]
            result = f_numpy(*(x_list + u_list))
            result = np.array(result).flatten()
        else:
            # Batched evaluation
            results = []
            for i in range(x.shape[0]):
                x_list = [x[i, j] for j in range(self.system.nx)]
                u_list = [u[i, j] for j in range(self.system.nu)]
                result = f_numpy(*(x_list + u_list))
                results.append(np.array(result).flatten())
            result = np.stack(results)
        
        # Update performance stats
        self._stats['calls'] += 1
        self._stats['time'] += time.time() - start_time
        
        return result
    
    def _evaluate_torch(self, x: "torch.Tensor", u: "torch.Tensor") -> "torch.Tensor":
        """
        PyTorch backend implementation.
        
        Handles both single and batched evaluation with GPU support.
        """
        import torch
        
        start_time = time.time()
        
        # Input validation
        if len(x.shape) == 0 or len(u.shape) == 0:
            raise ValueError("Input tensors must be at least 1D")
        
        if len(x.shape) >= 1 and x.shape[-1] != self.system.nx:
            raise ValueError(
                f"Expected state dimension {self.system.nx}, got {x.shape[-1]}"
            )
        if len(u.shape) >= 1 and u.shape[-1] != self.system.nu:
            raise ValueError(
                f"Expected control dimension {self.system.nu}, got {u.shape[-1]}"
            )
        
        # Generate function (uses cache if available)
        f_torch = self.code_gen.generate_dynamics('torch')
        
        # Track original input shape
        original_ndim = len(x.shape)
        
        # Handle batched vs single evaluation
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Prepare arguments
        x_list = [x[:, i] for i in range(self.system.nx)]
        u_list = [u[:, i] for i in range(self.system.nu)]
        all_args = x_list + u_list
        
        # Call generated function
        result = f_torch(*all_args)
        
        # Handle output shape
        if squeeze_output:
            # Single input case - squeeze batch dimension
            result = result.squeeze(0)
            
            # Ensure at least 1D
            if result.ndim == 0:
                result = result.unsqueeze(0)
        else:
            # Batched input case - ensure proper 2D shape (batch, nq)
            if result.ndim == 1:
                # If result is (batch,), reshape to (batch, 1) for single output systems
                if self.system.order > 1:
                    result = result.unsqueeze(1)
                else:
                    # For first-order systems with nx=1, also need (batch, 1)
                    if self.system.nx == 1:
                        result = result.unsqueeze(1)
        
        # Update performance stats
        self._stats['calls'] += 1
        self._stats['time'] += time.time() - start_time
        
        return result
    
    def _evaluate_jax(self, x: "jnp.ndarray", u: "jnp.ndarray") -> "jnp.ndarray":
        """
        JAX backend implementation.
        
        Handles both single and batched evaluation with vmap for efficiency.
        """
        import jax
        import jax.numpy as jnp
        
        start_time = time.time()
        
        # Input validation
        if x.ndim == 0 or u.ndim == 0:
            raise ValueError("Input arrays must be at least 1D")
        
        if x.ndim >= 1 and x.shape[-1] != self.system.nx:
            raise ValueError(
                f"Expected state dimension {self.system.nx}, got {x.shape[-1]}"
            )
        if u.ndim >= 1 and u.shape[-1] != self.system.nu:
            raise ValueError(
                f"Expected control dimension {self.system.nu}, got {u.shape[-1]}"
            )
        
        # Generate function (uses cache if available)
        f_jax = self.code_gen.generate_dynamics('jax', jit=True)
        
        # Handle batched vs single evaluation
        if x.ndim == 1:
            x = jnp.expand_dims(x, 0)
            u = jnp.expand_dims(u, 0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # For batched computation, use vmap
        if x.shape[0] > 1:
            @jax.vmap
            def batched_dynamics(x_i, u_i):
                x_list = [x_i[j] for j in range(self.system.nx)]
                u_list = [u_i[j] for j in range(self.system.nu)]
                return f_jax(*(x_list + u_list))
            
            result = batched_dynamics(x, u)
        else:
            # Single evaluation
            x_list = [x[0, i] for i in range(self.system.nx)]
            u_list = [u[0, i] for i in range(self.system.nu)]
            result = f_jax(*(x_list + u_list))
            result = jnp.expand_dims(result, 0)
        
        # Handle output shape
        if squeeze_output:
            result = result.squeeze(0)
            
            # Ensure at least 1D
            if result.ndim == 0:
                result = jnp.expand_dims(result, 0)
        else:
            # Batched case - ensure proper 2D shape
            if result.ndim == 1:
                if self.system.order > 1 or self.system.nx == 1:
                    result = jnp.expand_dims(result, 1)
        
        # Update performance stats
        self._stats['calls'] += 1
        self._stats['time'] += time.time() - start_time
        
        return result
    
    # ========================================================================
    # Performance Tracking
    # ========================================================================
    
    def get_stats(self) -> dict:
        """
        Get performance statistics.
        
        Returns:
            Dict with call count, total time, and average time
            
        Example:
            >>> stats = evaluator.get_stats()
            >>> print(f"Calls: {stats['calls']}")
            >>> print(f"Avg time: {stats['avg_time']:.6f}s")
        """
        return {
            'calls': self._stats['calls'],
            'total_time': self._stats['time'],
            'avg_time': self._stats['time'] / max(1, self._stats['calls']),
        }
    
    def reset_stats(self):
        """
        Reset performance counters.
        
        Example:
            >>> evaluator.reset_stats()
            >>> # Stats are now zero
        """
        self._stats['calls'] = 0
        self._stats['time'] = 0.0
    
    # ========================================================================
    # String Representations
    # ========================================================================
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"DynamicsEvaluator("
            f"nx={self.system.nx}, nu={self.system.nu}, "
            f"calls={self._stats['calls']})"
        )
    
    def __str__(self) -> str:
        """Human-readable string"""
        return f"DynamicsEvaluator(calls={self._stats['calls']})"