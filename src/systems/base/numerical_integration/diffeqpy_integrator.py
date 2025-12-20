"""
DiffEqPyODEIntegrator: Julia DifferentialEquations.jl ODE solver via diffeqpy.

Provides access to Julia's DifferentialEquations.jl ecosystem - the most
comprehensive and performant ODE solver suite available in any language.

Requires: Julia and diffeqpy installed
    julia> using Pkg
    julia> Pkg.add("DifferentialEquations")
    
    $ pip install diffeqpy
"""

from typing import Optional, Callable, Tuple, Dict, Any
import numpy as np

from src.systems.base.numerical_integration.integrator_base import (
    IntegratorBase,
    IntegrationResult,
    StepMode,
    ArrayLike
)


class DiffEqPyODEIntegrator(IntegratorBase):
    """
    ODE integrator using Julia's DifferentialEquations.jl via diffeqpy.
    
    Provides access to 100+ ODE solvers from the Julia ecosystem, including:
    - Adaptive explicit: Tsit5, Vern7, Vern9, DP5, DP8
    - Stiff implicit: Rosenbrock23, Rodas4, RadauIIA5, KenCarp4
    - Geometric: SymplecticEuler, VelocityVerlet, McAte4
    - Stabilized: ROCK2, ROCK4, ESERK
    - And many more...
    
    Parameters
    ----------
    system : SymbolicDynamicalSystem
        Continuous-time system
    dt : Optional[float]
        Time step (initial guess for adaptive)
    backend : str
        Must be 'numpy' (Julia arrays convert to NumPy)
    algorithm : str, optional
        Julia solver algorithm. Default: 'Tsit5'
        Common choices:
        - Non-stiff: 'Tsit5', 'Vern7', 'Vern9', 'DP5'
        - Stiff: 'Rosenbrock23', 'Rodas4', 'KenCarp4'
        - Auto: 'AutoTsit5(Rosenbrock23())' (auto-switches)
    **options
        Additional solver options (abstol, reltol, saveat, etc.)
    
    Examples
    --------
    >>> system = SymbolicDynamicalSystem(...)
    >>> 
    >>> # High-accuracy non-stiff
    >>> integrator = DiffEqPyODEIntegrator(
    ...     system,
    ...     backend='numpy',
    ...     algorithm='Vern9',
    ...     reltol=1e-12,
    ...     abstol=1e-14
    ... )
    >>> 
    >>> # Stiff system
    >>> integrator = DiffEqPyODEIntegrator(
    ...     system,
    ...     algorithm='Rosenbrock23'
    ... )
    >>> 
    >>> # Auto-switching
    >>> integrator = DiffEqPyODEIntegrator(
    ...     system,
    ...     algorithm='AutoTsit5(Rosenbrock23())'
    ... )
    """
    
    def __init__(
        self,
        system,
        dt: Optional[float] = None,
        step_mode: StepMode = StepMode.ADAPTIVE,
        backend: str = 'numpy',
        algorithm: str = 'Tsit5',
        **options
    ):
        if backend != 'numpy':
            raise ValueError(
                f"DiffEqPyODEIntegrator requires backend='numpy', got '{backend}'. "
                f"Julia arrays are converted to NumPy."
            )
        
        super().__init__(system, dt, step_mode, backend, **options)
        
        self.algorithm = algorithm
        self._integrator_name = f"DiffEqPy-{algorithm}"
        
        # Try to import diffeqpy
        try:
            from diffeqpy import de
            self.de = de
        except ImportError:
            raise ImportError(
                "diffeqpy is required for DiffEqPyODEIntegrator.\n"
                "Install Julia, then:\n"
                "  julia> using Pkg\n"
                "  julia> Pkg.add('DifferentialEquations')\n"
                "Then install Python package:\n"
                "  pip install diffeqpy"
            )
    
    @property
    def name(self) -> str:
        """Return integrator name."""
        mode_str = "Fixed" if self.step_mode == StepMode.FIXED else "Adaptive"
        return f"{self._integrator_name} ({mode_str})"
    
    def step(self, x: ArrayLike, u: ArrayLike, dt: Optional[float] = None) -> ArrayLike:
        """Single ODE step using Julia solver."""
        step_size = dt if dt is not None else self.dt
        if step_size is None:
            raise ValueError("Step size dt must be specified")
        
        # Use integrate for single step
        result = self.integrate(
            x,
            lambda t, x_cur: u,
            (0.0, step_size),
            t_eval=np.array([step_size])
        )
        
        self._stats['total_steps'] += 1
        
        return result.x[-1]
    
    def integrate(
        self,
        x0: ArrayLike,
        u_func: Callable,
        t_span: Tuple[float, float],
        t_eval: Optional[ArrayLike] = None,
        dense_output: bool = False
    ) -> IntegrationResult:
        """
        Integrate ODE using Julia's DifferentialEquations.jl.
        
        Parameters
        ----------
        x0 : ArrayLike
            Initial state
        u_func : Callable
            Control policy (t, x) → u
        t_span : Tuple[float, float]
            Time interval
        t_eval : ArrayLike, optional
            Evaluation times
        dense_output : bool
            Return dense output for interpolation
        
        Returns
        -------
        IntegrationResult
            Integration result
        """
        t0, tf = t_span
        x0 = np.asarray(x0)
        
        # Handle edge case
        if t0 == tf:
            return IntegrationResult(
                t=np.array([t0]),
                x=x0[None, :] if x0.ndim == 1 else x0,
                success=True,
                message="Zero time span",
                nfev=0,
                nsteps=0,
            )
        
        # Define ODE function for Julia
        def ode_func(u_val, p, t):
            """
            Julia signature: du = f(u, p, t)
            where u=state, p=parameters, t=time
            """
            # Convert from Julia arrays to NumPy
            x_np = np.array(u_val)
            
            # Evaluate control
            u_control = u_func(t, x_np)
            u_np = np.asarray(u_control)
            
            # Evaluate dynamics
            dx = self.system(x_np, u_np, backend='numpy')
            self._stats['total_fev'] += 1
            
            return dx
        
        # Prepare time span
        tspan = (t0, tf)
        
        # Prepare save points
        if t_eval is not None:
            saveat = list(t_eval)
        elif self.step_mode == StepMode.FIXED:
            n_steps = int((tf - t0) / self.dt) + 1
            saveat = list(np.linspace(t0, tf, n_steps))
        else:
            saveat = []  # Julia chooses adaptively
        
        # Set up problem
        prob = self.de.ODEProblem(ode_func, x0, tspan)
        
        # Solve
        try:
            sol = self.de.solve(
                prob,
                self._get_algorithm(),
                reltol=self.rtol,
                abstol=self.atol,
                saveat=saveat if saveat else None,
                save_everystep=(len(saveat) == 0),
                dense=dense_output,
                dt=self.dt if self.step_mode == StepMode.FIXED else None,
            )
            
            # Extract solution
            # Julia returns solution object, convert to NumPy
            t_out = np.array(sol.t)
            x_out = np.array(sol.u).T  # Julia gives list of vectors
            
            success = True
            message = "Integration successful"
            
            # Update stats
            nsteps = len(t_out) - 1
            self._stats['total_steps'] += nsteps
            
            return IntegrationResult(
                t=t_out,
                x=x_out,
                success=success,
                message=message,
                nfev=self._stats['total_fev'],
                nsteps=nsteps,
                algorithm=self.algorithm,
                sol=sol if dense_output else None,
            )
            
        except Exception as e:
            return IntegrationResult(
                t=np.array([t0]),
                x=x0[None, :],
                success=False,
                message=f"Integration failed: {str(e)}",
                nfev=0,
                nsteps=0,
            )
    
    def _get_algorithm(self):
        """Get Julia algorithm object."""
        # Parse algorithm string
        # Handle special cases like 'AutoTsit5(Rosenbrock23())'
        try:
            return eval(f"self.de.{self.algorithm}")
        except:
            # Fallback: try as simple algorithm name
            return getattr(self.de, self.algorithm)()


# ============================================================================
# Convenience Functions
# ============================================================================

def list_diffeqpy_algorithms() -> Dict[str, list]:
    """
    List available Julia DifferentialEquations.jl algorithms.
    
    Returns
    -------
    Dict[str, list]
        Categories of algorithms
    
    Examples
    --------
    >>> algos = list_diffeqpy_algorithms()
    >>> print(algos['nonstiff'])
    ['Tsit5', 'Vern7', 'Vern9', 'DP5', 'DP8', ...]
    """
    return {
        'nonstiff': [
            'Tsit5',  # Tsitouras 5(4) - default, excellent
            'Vern6', 'Vern7', 'Vern8', 'Vern9',  # Verner methods (high order)
            'DP5', 'DP8',  # Dormand-Prince
            'TanYam7',  # Tanaka-Yamashita 7
            'TsitPap8',  # Tsitouras-Papakostas 8
        ],
        'stiff_explicit': [
            'Rosenbrock23', 'Rosenbrock32',  # Rosenbrock (recommended)
            'TRBDF2', 'KenCarp3', 'KenCarp4', 'KenCarp5',  # ESDIRK
        ],
        'stiff_implicit': [
            'Rodas4', 'Rodas4P', 'Rodas5',  # Rosenbrock (high accuracy)
            'RadauIIA5',  # Radau
            'QNDF', 'FBDF',  # BDF variants
        ],
        'auto_switching': [
            'AutoTsit5(Rosenbrock23())',  # Auto non-stiff/stiff
            'AutoVern7(Rodas5())',
        ],
        'geometric': [
            'SymplecticEuler', 'VelocityVerlet',  # Symplectic
            'McAte2', 'McAte4', 'McAte5',  # McClellan-Aitken
        ],
        'stabilized': [
            'ROCK2', 'ROCK4',  # For moderately stiff
            'ESERK4', 'ESERK5',  # Stabilized ERK
        ],
    }