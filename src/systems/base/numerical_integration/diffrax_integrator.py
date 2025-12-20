"""
DiffraxIntegrator: JAX-based ODE integration using Diffrax library.

This module provides adaptive and fixed-step ODE integration with automatic
differentiation support through JAX's JIT compilation.
"""

from typing import Optional, Callable, Tuple
import jax
import jax.numpy as jnp
from jax import Array
import diffrax as dfx

from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    IntegrationResult,
    StepMode,
    ArrayLike
)


class DiffraxIntegrator(IntegratorBase):
    """
    JAX-based ODE integrator using the Diffrax library.

    Supports adaptive and fixed-step integration with various solvers
    (Tsit5, Dopri5, Euler, etc.) and automatic differentiation.

    Parameters
    ----------
    system : SymbolicDynamicalSystem
        Continuous-time system to integrate
    dt : Optional[float]
        Time step size
    step_mode : StepMode
        FIXED or ADAPTIVE stepping mode
    backend : str
        Must be 'jax' for this integrator
    solver : str, optional
        Solver name. Options: 'tsit5', 'dopri5', 'dopri8', 'euler',
        'midpoint', 'heun', 'ralston', 'bosh3', 'reversible_heun'.
        Default: 'tsit5'
    adjoint : str, optional
        Adjoint method for backpropagation. Options: 'recursive_checkpoint',
        'direct', 'implicit'. Default: 'recursive_checkpoint'
    **options
        Additional options including rtol, atol, max_steps
    """

    def __init__(
        self,
        system,
        dt: Optional[float] = None,
        step_mode: StepMode = StepMode.FIXED,
        backend: str = 'jax',
        solver: str = "tsit5",
        adjoint: str = "recursive_checkpoint",
        **options
    ):
        # Validate backend
        if backend != 'jax':
            raise ValueError(
                f"DiffraxIntegrator requires backend='jax', got '{backend}'"
            )
        
        # Initialize base class
        super().__init__(system, dt, step_mode, backend, **options)
        
        self.solver_name = solver.lower()
        self.adjoint_name = adjoint
        self._integrator_name = f"Diffrax-{solver}"

        # Map solver names to Diffrax solver classes
        self._solver_map = {
            "tsit5": dfx.Tsit5,
            "dopri5": dfx.Dopri5,
            "dopri8": dfx.Dopri8,
            "euler": dfx.Euler,
            "midpoint": dfx.Midpoint,
            "heun": dfx.Heun,
            "ralston": dfx.Ralston,
            "bosh3": dfx.Bosh3,
            "reversible_heun": dfx.ReversibleHeun,
        }

        if self.solver_name not in self._solver_map:
            raise ValueError(
                f"Unknown solver '{solver}'. Available: {list(self._solver_map.keys())}"
            )

        # Map adjoint names to Diffrax adjoint classes
        self._adjoint_map = {
            "recursive_checkpoint": dfx.RecursiveCheckpointAdjoint,
            "direct": dfx.DirectAdjoint,
            "implicit": dfx.ImplicitAdjoint,
        }

        if self.adjoint_name not in self._adjoint_map:
            raise ValueError(
                f"Unknown adjoint '{adjoint}'. Available: {list(self._adjoint_map.keys())}"
            )

    @property
    def name(self) -> str:
        """Return the name of the integrator."""
        mode_str = "Fixed Step" if self.step_mode == StepMode.FIXED else "Adaptive"
        return f"{self._integrator_name} ({mode_str})"

    def step(
        self,
        x: ArrayLike,
        u: ArrayLike,
        dt: Optional[float] = None
    ) -> ArrayLike:
        """
        Take one integration step: x(t) → x(t + dt).
        
        Parameters
        ----------
        x : ArrayLike
            Current state (nx,) or (batch, nx)
        u : ArrayLike
            Control input (nu,) or (batch, nu)
        dt : Optional[float]
            Step size (uses self.dt if None)
            
        Returns
        -------
        ArrayLike
            Next state x(t + dt)
        """
        step_size = dt if dt is not None else self.dt
        
        if step_size is None:
            raise ValueError("Step size dt must be specified")
        
        # Convert to JAX arrays if needed
        x = jnp.asarray(x)
        u = jnp.asarray(u)
        
        # Define ODE function - MUST accept (t, y, args) even if args unused
        def ode_func(t, state, args):
            return self.system(state, u, backend=self.backend)
        
        # Create ODE term
        term = dfx.ODETerm(ode_func)
        solver = self._solver_map[self.solver_name]()
        
        # Single step integration
        solution = dfx.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=step_size,
            dt0=step_size,
            y0=x,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.ConstantStepSize(),
            max_steps=10,
        )
        
        # Update stats AFTER integration
        self._stats['total_steps'] += 1
        self._stats['total_fev'] += int(solution.stats.get('num_steps', 1))
        
        return solution.ys[0]

    def integrate(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], ArrayLike],
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
        dense_output: bool = False
    ) -> IntegrationResult:
        """
        Integrate over time interval with control policy.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state (nx,)
        u_func : Callable[[float, ArrayLike], ArrayLike]
            Control policy: (t, x) → u
        t_span : Tuple[float, float]
            Integration interval (t_start, t_end)
        t_eval : Optional[ArrayLike]
            Specific times at which to store solution
        dense_output : bool
            If True, return dense interpolated solution
            
        Returns
        -------
        IntegrationResult
            Object containing t, x, success, and metadata
        """
        t0, tf = t_span
        x0 = jnp.asarray(x0)
        
        # Handle edge cases
        if t0 == tf:
            return IntegrationResult(
                t=jnp.array([t0]),
                x=x0[None, :] if x0.ndim == 1 else x0,
                success=True,
                message="Zero time span",
                nfev=0,
                nsteps=0,
            )
        
        # Handle backward integration
        if tf < t0:
            backward = True
            t0_actual, tf_actual = tf, t0
        else:
            backward = False
            t0_actual, tf_actual = t0, tf
        
        # Define ODE function - MUST accept (t, y, args) even if args unused
        def ode_func(t, state, args):
            t_actual = t if not backward else (t0 + tf - t)
            u = u_func(t_actual, state)
            xdot = self.system(state, u, backend=self.backend)
            return -xdot if backward else xdot
        
        # Create ODE term
        term = dfx.ODETerm(ode_func)
        solver = self._solver_map[self.solver_name]()
        
        # Set up step size controller and save points
        if self.step_mode == StepMode.FIXED:
            if t_eval is not None:
                t_points = jnp.asarray(t_eval)
                if backward:
                    t_points_actual = t0_actual + tf_actual - t_points[::-1]
                else:
                    t_points_actual = t_points
            else:
                n_steps = max(2, int((tf_actual - t0_actual) / self.dt) + 1)
                t_points_actual = jnp.linspace(t0_actual, tf_actual, n_steps)
                t_points = jnp.linspace(t0, tf, n_steps)
            
            stepsize_controller = dfx.StepTo(ts=t_points_actual)
            saveat = dfx.SaveAt(ts=t_points_actual)
            dt0_value = None  # StepTo determines step locations
            
        else:
            # Adaptive step mode
            stepsize_controller = dfx.PIDController(
                rtol=self.rtol,
                atol=self.atol,
                dtmin=self.options.get("dtmin", None),
                dtmax=self.options.get("dtmax", None),
            )
            
            if t_eval is not None:
                t_points = jnp.asarray(t_eval)
                if backward:
                    t_points_actual = t0_actual + tf_actual - t_points[::-1]
                else:
                    t_points_actual = t_points
                saveat = dfx.SaveAt(ts=t_points_actual)
            else:
                n_dense = max(2, self.options.get("n_dense", 100))
                t_points_actual = jnp.linspace(t0_actual, tf_actual, n_dense)
                t_points = jnp.linspace(t0, tf, n_dense)
                saveat = dfx.SaveAt(ts=t_points_actual)
            
            dt0_value = self.dt if self.dt is not None else (tf_actual - t0_actual) / 100
        
        # Set up adjoint method
        adjoint_method = self._adjoint_map[self.adjoint_name]()
        
        # Solve ODE
        try:
            solution = dfx.diffeqsolve(
                term,
                solver,
                t0=t0_actual,
                t1=tf_actual,
                dt0=dt0_value,
                y0=x0,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                max_steps=self.max_steps,
                adjoint=adjoint_method,
                throw=False,
            )
            
            # Check success
            success = jnp.all(jnp.isfinite(solution.ys))
            
            # Extract results
            t_result = solution.ts if not backward else (t0 + tf - solution.ts[::-1])
            x_result = solution.ys if not backward else solution.ys[::-1]
            
            # Update statistics
            nsteps = int(solution.stats.get("num_steps", 0))
            nfev = int(solution.stats.get("num_steps", 0))
            self._stats['total_steps'] += nsteps
            self._stats['total_fev'] += nfev
            
            return IntegrationResult(
                t=t_result,
                x=x_result,
                success=bool(success),
                message="Integration successful" if success else "Integration failed (NaN/Inf detected)",
                nfev=nfev,
                nsteps=nsteps,
                solver=self.solver_name,
                njev=int(solution.stats.get("num_jacobian_evals", 0)),
            )
            
        except Exception as e:
            return IntegrationResult(
                t=jnp.array([t0]),
                x=x0[None, :] if x0.ndim == 1 else x0,
                success=False,
                message=f"Integration failed: {str(e)}",
                nfev=0,
                nsteps=0,
            )

    # ========================================================================
    # JAX-Specific Methods
    # ========================================================================

    def integrate_with_gradient(
        self,
        x0: ArrayLike,
        u_func: Callable[[float, ArrayLike], ArrayLike],
        t_span: Tuple[float, float],
        loss_fn: Callable[[IntegrationResult], float],
        t_eval: Optional[ArrayLike] = None,
    ):
        """Integrate and compute gradients w.r.t. initial conditions."""
        def compute_loss(x0_val):
            result = self.integrate(x0_val, u_func, t_span, t_eval)
            return loss_fn(result)
        
        loss, grad = jax.value_and_grad(compute_loss)(x0)
        return loss, grad

    def jit_compile_step(self):
        """Return a JIT-compiled version of the step function."""
        # Store reference to system and solver outside of JIT
        system = self.system
        backend = self.backend
        solver_class = self._solver_map[self.solver_name]
        
        @jax.jit
        def jitted_step(x, u, dt):
            # Pure JIT-compatible step (no stats tracking)
            x = jnp.asarray(x)
            u = jnp.asarray(u)
            
            def ode_func(t, state, args):
                return system(state, u, backend=backend)
            
            term = dfx.ODETerm(ode_func)
            solver = solver_class()
            
            solution = dfx.diffeqsolve(
                term, solver,
                t0=0.0, t1=dt,
                dt0=dt, y0=x,
                saveat=dfx.SaveAt(t1=True),
                stepsize_controller=dfx.ConstantStepSize(),
                max_steps=10
            )
            
            return solution.ys[0]
        
        return jitted_step

    def vectorized_step(self, x_batch: ArrayLike, u_batch: ArrayLike, dt: Optional[float] = None):
        """Vectorized step over batch of states and controls."""
        return jax.vmap(lambda x, u: self.step(x, u, dt))(x_batch, u_batch)

    def vectorized_integrate(
        self,
        x0_batch: ArrayLike,
        u_func: Callable[[float, ArrayLike], ArrayLike],
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
    ):
        """Vectorized integration over batch of initial conditions."""
        results = []
        for i in range(x0_batch.shape[0]):
            results.append(self.integrate(x0_batch[i], u_func, t_span, t_eval))
        return results