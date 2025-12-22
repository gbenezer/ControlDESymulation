"""
Unit Tests for DiffEqPySDEIntegrator

Tests Julia-based SDE integration via DiffEqPy, including:
- Initialization and validation
- Algorithm selection and availability
- Integration with autonomous and controlled systems
- Pure diffusion systems (zero drift)
- Comparison with known analytical solutions
- Error handling and edge cases
- Algorithm recommendations
- Performance and convergence properties
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Check if diffeqpy is available
try:
    from diffeqpy import de
    DIFFEQPY_AVAILABLE = True
except ImportError:
    DIFFEQPY_AVAILABLE = False

from src.systems.base.numerical_integration.stochastic.diffeqpy_sde_integrator import (
    DiffEqPySDEIntegrator,
    create_diffeqpy_sde_integrator,
    list_julia_sde_algorithms,
)
from src.systems.base.numerical_integration.stochastic.sde_integrator_base import (
    SDEType,
    ConvergenceType,
    StepMode
)
from src.systems.base.stochastic_dynamical_system import StochasticDynamicalSystem


# ============================================================================
# Skip Tests if DiffEqPy Not Available
# ============================================================================

pytestmark = pytest.mark.skipif(
    not DIFFEQPY_AVAILABLE,
    reason="diffeqpy not installed. Install: pip install diffeqpy"
)


# ============================================================================
# Mock SDE Systems for Testing
# ============================================================================

class OrnsteinUhlenbeck(StochasticDynamicalSystem):
    """
    Ornstein-Uhlenbeck process with known analytical properties.
    
    dx = -alpha * x * dt + sigma * dW
    
    Properties:
    - Stationary distribution: N(0, sigma^2 / (2*alpha))
    - Mean: E[X(t)] = x0 * exp(-alpha*t)
    - Variance: Var[X(t)] = (sigma^2 / (2*alpha)) * (1 - exp(-2*alpha*t))
    """
    
    def define_system(self, alpha=1.0, sigma=0.5):
        import sympy as sp
        
        x = sp.symbols('x', real=True)
        alpha_sym = sp.symbols('alpha', positive=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        # Drift
        self.state_vars = [x]
        self.control_vars = []  # Autonomous
        self._f_sym = sp.Matrix([[-alpha_sym * x]])
        self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
        self.order = 1
        
        # Diffusion (additive)
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = 'ito'


class GeometricBrownianMotion(StochasticDynamicalSystem):
    """
    Geometric Brownian motion (multiplicative noise).
    
    dx = mu * x * dt + sigma * x * dW
    
    Solution: X(t) = X(0) * exp((mu - 0.5*sigma^2)*t + sigma*W(t))
    """
    
    def define_system(self, mu=0.1, sigma=0.2):
        import sympy as sp
        
        x = sp.symbols('x', positive=True)
        mu_sym = sp.symbols('mu', real=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        # Drift
        self.state_vars = [x]
        self.control_vars = []  # Autonomous
        self._f_sym = sp.Matrix([[mu_sym * x]])
        self.parameters = {mu_sym: mu, sigma_sym: sigma}
        self.order = 1
        
        # Diffusion (multiplicative)
        self.diffusion_expr = sp.Matrix([[sigma_sym * x]])
        self.sde_type = 'ito'


class BrownianMotion(StochasticDynamicalSystem):
    """
    Pure Brownian motion (zero drift, pure diffusion).
    
    dx = 0 * dt + sigma * dW
    or simply: dx = sigma * dW
    
    Properties:
    - Mean: E[X(t)] = X(0)
    - Variance: Var[X(t)] = sigma^2 * t
    - Gaussian: X(t) ~ N(X(0), sigma^2 * t)
    """
    
    def define_system(self, sigma=1.0):
        import sympy as sp
        
        x = sp.symbols('x', real=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        # Zero drift (pure diffusion)
        self.state_vars = [x]
        self.control_vars = []  # Autonomous
        self._f_sym = sp.Matrix([[0]])  # Zero drift!
        self.parameters = {sigma_sym: sigma}
        self.order = 1
        
        # Diffusion only
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = 'ito'


class ControlledOU(StochasticDynamicalSystem):
    """
    Controlled Ornstein-Uhlenbeck process.
    
    dx = (-alpha * x + u) * dt + sigma * dW
    """
    
    def define_system(self, alpha=1.0, sigma=0.5):
        import sympy as sp
        
        x = sp.symbols('x', real=True)
        u = sp.symbols('u', real=True)
        alpha_sym = sp.symbols('alpha', positive=True)
        sigma_sym = sp.symbols('sigma', positive=True)
        
        # Drift
        self.state_vars = [x]
        self.control_vars = [u]
        self._f_sym = sp.Matrix([[-alpha_sym * x + u]])
        self.parameters = {alpha_sym: alpha, sigma_sym: sigma}
        self.order = 1
        
        # Diffusion
        self.diffusion_expr = sp.Matrix([[sigma_sym]])
        self.sde_type = 'ito'


class TwoDimensionalOU(StochasticDynamicalSystem):
    """
    2D Ornstein-Uhlenbeck with diagonal noise (autonomous).
    
    dx1 = -alpha * x1 * dt + sigma1 * dW1
    dx2 = -alpha * x2 * dt + sigma2 * dW2
    """
    
    def define_system(self, alpha=1.0, sigma1=0.5, sigma2=0.3):
        import sympy as sp
        
        x1, x2 = sp.symbols('x1 x2', real=True)
        alpha_sym = sp.symbols('alpha', positive=True)
        sigma1_sym = sp.symbols('sigma1', positive=True)
        sigma2_sym = sp.symbols('sigma2', positive=True)
        
        # Drift
        self.state_vars = [x1, x2]
        self.control_vars = []  # Autonomous
        self._f_sym = sp.Matrix([
            [-alpha_sym * x1],
            [-alpha_sym * x2]
        ])
        self.parameters = {
            alpha_sym: alpha,
            sigma1_sym: sigma1,
            sigma2_sym: sigma2
        }
        self.order = 1
        
        # Diffusion (diagonal)
        self.diffusion_expr = sp.Matrix([
            [sigma1_sym, 0],
            [0, sigma2_sym]
        ])
        self.sde_type = 'ito'


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def ou_system():
    """Create Ornstein-Uhlenbeck system (autonomous, additive noise)."""
    return OrnsteinUhlenbeck(alpha=1.0, sigma=0.5)


@pytest.fixture
def gbm_system():
    """Create Geometric Brownian Motion (autonomous, multiplicative noise)."""
    return GeometricBrownianMotion(mu=0.1, sigma=0.2)


@pytest.fixture
def brownian_system():
    """Create pure Brownian motion (zero drift, autonomous)."""
    return BrownianMotion(sigma=1.0)


@pytest.fixture
def controlled_system():
    """Create controlled OU system."""
    return ControlledOU(alpha=1.0, sigma=0.5)


@pytest.fixture
def ou_2d_system():
    """Create 2D OU system (autonomous, diagonal noise)."""
    return TwoDimensionalOU(alpha=1.0, sigma1=0.5, sigma2=0.3)


@pytest.fixture
def integrator_em(ou_system):
    """Create Euler-Maruyama integrator."""
    return DiffEqPySDEIntegrator(
        ou_system,
        dt=0.01,
        algorithm='EM',
        seed=42
    )


@pytest.fixture
def integrator_sriw1(ou_system):
    """Create high-accuracy SRIW1 integrator."""
    return DiffEqPySDEIntegrator(
        ou_system,
        dt=0.001,
        algorithm='SRIW1',
        seed=42
    )


# ============================================================================
# Test Class: Initialization and Validation
# ============================================================================

class TestDiffEqPySDEInitialization:
    """Test initialization and validation."""
    
    def test_basic_initialization(self, ou_system):
        """Test basic integrator initialization."""
        integrator = DiffEqPySDEIntegrator(
            ou_system,
            dt=0.01,
            algorithm='EM'
        )
        
        assert integrator.sde_system is ou_system
        assert integrator.dt == 0.01
        assert integrator.algorithm == 'EM'
        assert integrator.backend == 'numpy'
    
    def test_backend_must_be_numpy(self, ou_system):
        """Test that non-numpy backend raises error."""
        with pytest.raises(ValueError, match="requires backend='numpy'"):
            DiffEqPySDEIntegrator(
                ou_system,
                dt=0.01,
                algorithm='EM',
                backend='torch'  # Not allowed
            )
    
    def test_invalid_algorithm_raises(self, ou_system):
        """Test that invalid algorithm raises error."""
        with pytest.raises(ValueError, match="Unknown Julia SDE algorithm"):
            DiffEqPySDEIntegrator(
                ou_system,
                dt=0.01,
                algorithm='NonExistentAlgorithm'
            )
    
    def test_valid_algorithms_accepted(self, ou_system):
        """Test that all listed algorithms are accepted."""
        algorithms = ['EM', 'LambaEM', 'SRIW1', 'SRA1', 'RKMil']
        
        for alg in algorithms:
            integrator = DiffEqPySDEIntegrator(
                ou_system,
                dt=0.01,
                algorithm=alg
            )
            assert integrator.algorithm == alg
    
    def test_step_mode_defaults_to_adaptive(self, ou_system):
        """Test that default step mode is ADAPTIVE."""
        integrator = DiffEqPySDEIntegrator(
            ou_system,
            dt=0.01,
            algorithm='EM'
        )
        
        assert integrator.step_mode == StepMode.ADAPTIVE
    
    def test_custom_tolerances(self, ou_system):
        """Test custom tolerance settings."""
        integrator = DiffEqPySDEIntegrator(
            ou_system,
            dt=0.01,
            algorithm='EM',
            rtol=1e-6,
            atol=1e-8
        )
        
        assert integrator.rtol == 1e-6
        assert integrator.atol == 1e-8
    
    def test_seed_initialization(self, ou_system):
        """Test random seed initialization."""
        integrator = DiffEqPySDEIntegrator(
            ou_system,
            dt=0.01,
            algorithm='EM',
            seed=42
        )
        
        assert integrator.seed == 42
    
    def test_diffeqpy_import_available(self, integrator_em):
        """Test that diffeqpy is properly imported."""
        assert hasattr(integrator_em, 'de')
        assert integrator_em.de is not None


# ============================================================================
# Test Class: Autonomous Systems
# ============================================================================

class TestAutonomousSystems:
    """Test integration of autonomous SDE systems."""
    
    def test_autonomous_ou_integration(self, ou_system):
        """Test integration of autonomous OU process."""
        integrator = DiffEqPySDEIntegrator(
            ou_system,
            dt=0.01,
            algorithm='EM',
            seed=42
        )
        
        x0 = np.array([1.0])
        u_func = lambda t, x: None  # Autonomous
        t_span = (0.0, 1.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert result.x.shape[0] > 1
        assert result.x.shape[1] == 1
        assert result.n_paths == 1
    
    def test_autonomous_2d_integration(self, ou_2d_system):
        """Test integration of 2D autonomous system."""
        integrator = DiffEqPySDEIntegrator(
            ou_2d_system,
            dt=0.01,
            algorithm='EM',
            seed=42
        )
        
        x0 = np.array([1.0, 2.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert result.x.shape[1] == 2  # 2D system
    
    def test_autonomous_ou_mean_decay(self, ou_system):
        """Test that OU mean decays exponentially (Monte Carlo)."""
        integrator = DiffEqPySDEIntegrator(
            ou_system,
            dt=0.01,
            algorithm='EM',
            seed=42
        )
        
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 2.0)
        
        # Run multiple trajectories manually
        n_paths = 100
        final_states = []
        
        for i in range(n_paths):
            integrator.seed = 42 + i
            integrator._rng = integrator._initialize_rng('numpy', 42 + i)
            result = integrator.integrate(x0, u_func, t_span)
            final_states.append(result.x[-1, 0])
        
        # Mean should decay: E[X(t)] = x0 * exp(-alpha * t)
        alpha = 1.0
        t_final = 2.0
        expected_mean = 1.0 * np.exp(-alpha * t_final)
        observed_mean = np.mean(final_states)
        
        # Allow 20% error due to finite sampling
        assert abs(observed_mean - expected_mean) < 0.2 * abs(expected_mean)


# ============================================================================
# Test Class: Pure Diffusion Systems
# ============================================================================

class TestPureDiffusionSystems:
    """Test pure diffusion systems (zero drift)."""
    
    def test_pure_diffusion_properties(self, brownian_system):
        """Test that Brownian motion system has correct properties."""
        assert brownian_system.is_pure_diffusion()
        assert brownian_system.nu == 0  # Autonomous
        assert brownian_system.nx == 1
        assert brownian_system.nw == 1
    
    def test_pure_diffusion_integration(self, brownian_system):
        """Test integration of pure Brownian motion."""
        integrator = DiffEqPySDEIntegrator(
            brownian_system,
            dt=0.01,
            algorithm='EM',
            seed=42
        )
        
        x0 = np.array([0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert result.x.shape[0] > 1
    
    def test_pure_diffusion_zero_mean(self, brownian_system):
        """Test that Brownian motion starting at zero has zero mean."""
        integrator = DiffEqPySDEIntegrator(
            brownian_system,
            dt=0.01,
            algorithm='EM',
            seed=42
        )
        
        x0 = np.array([0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        # Multiple trajectories
        n_paths = 200
        final_states = []
        
        for i in range(n_paths):
            integrator.seed = 100 + i
            integrator._rng = integrator._initialize_rng('numpy', 100 + i)
            result = integrator.integrate(x0, u_func, t_span)
            final_states.append(result.x[-1, 0])
        
        # Mean should be near zero (no drift)
        mean = np.mean(final_states)
        se = np.std(final_states) / np.sqrt(n_paths)
        
        # Should be within 3 standard errors of zero
        assert abs(mean) < 3 * se
    
    def test_pure_diffusion_variance_growth(self, brownian_system):
        """Test that variance grows linearly: Var(X(t)) = sigma^2 * t."""
        integrator = DiffEqPySDEIntegrator(
            brownian_system,
            dt=0.01,
            algorithm='EM',
            seed=42
        )
        
        x0 = np.array([0.0])
        u_func = lambda t, x: None
        
        # Test at two different times
        t1 = 0.5
        t2 = 1.5
        n_paths = 300
        
        # Collect samples at t1
        states_t1 = []
        for i in range(n_paths):
            integrator.seed = 200 + i
            integrator._rng = integrator._initialize_rng('numpy', 200 + i)
            result = integrator.integrate(x0, u_func, (0.0, t1))
            states_t1.append(result.x[-1, 0])
        
        # Collect samples at t2
        states_t2 = []
        for i in range(n_paths):
            integrator.seed = 200 + i
            integrator._rng = integrator._initialize_rng('numpy', 200 + i)
            result = integrator.integrate(x0, u_func, (0.0, t2))
            states_t2.append(result.x[-1, 0])
        
        var_t1 = np.var(states_t1)
        var_t2 = np.var(states_t2)
        
        # Variance ratio should equal time ratio: var(t2)/var(t1) ≈ t2/t1
        time_ratio = t2 / t1
        var_ratio = var_t2 / var_t1
        
        # Allow 30% tolerance for statistical fluctuation
        assert abs(var_ratio - time_ratio) / time_ratio < 0.3
    
    def test_pure_diffusion_gaussian_distribution(self, brownian_system):
        """Test that Brownian motion has Gaussian distribution."""
        integrator = DiffEqPySDEIntegrator(
            brownian_system,
            dt=0.01,
            algorithm='EM',
            seed=42
        )
        
        x0 = np.array([0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        # Collect many samples
        n_paths = 500
        final_states = []
        
        for i in range(n_paths):
            integrator.seed = 300 + i
            integrator._rng = integrator._initialize_rng('numpy', 300 + i)
            result = integrator.integrate(x0, u_func, t_span)
            final_states.append(result.x[-1, 0])
        
        final_states = np.array(final_states)
        
        # For Brownian motion: X(1) ~ N(0, sigma^2) = N(0, 1)
        mean = np.mean(final_states)
        variance = np.var(final_states)
        
        # Check mean near 0 and variance near 1
        assert abs(mean) < 0.15
        assert 0.8 < variance < 1.2


# ============================================================================
# Test Class: Controlled Systems
# ============================================================================

class TestControlledSystems:
    """Test integration with control inputs."""
    
    def test_controlled_integration(self, controlled_system):
        """Test integration with control input."""
        integrator = DiffEqPySDEIntegrator(
            controlled_system,
            dt=0.01,
            algorithm='EM',
            seed=42
        )
        
        x0 = np.array([1.0])
        u_func = lambda t, x: np.array([0.5])  # Constant control
        t_span = (0.0, 1.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        assert result.x.shape[0] > 1
    
    def test_state_feedback_control(self, controlled_system):
        """Test state feedback control."""
        integrator = DiffEqPySDEIntegrator(
            controlled_system,
            dt=0.01,
            algorithm='EM',
            seed=42
        )
        
        x0 = np.array([1.0])
        K = np.array([2.0])  # Feedback gain
        u_func = lambda t, x: -K * x  # State feedback
        t_span = (0.0, 2.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success
        # With strong feedback, state should be driven toward zero
        assert abs(result.x[-1, 0]) < abs(x0[0])
    
    def test_time_varying_control(self, controlled_system):
        """Test time-varying control."""
        integrator = DiffEqPySDEIntegrator(
            controlled_system,
            dt=0.01,
            algorithm='EM',
            seed=42
        )
        
        x0 = np.array([0.0])
        u_func = lambda t, x: np.array([np.sin(2*np.pi*t)])  # Sinusoidal
        t_span = (0.0, 2.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success


# ============================================================================
# Test Class: Integration Methods
# ============================================================================

class TestIntegrationMethods:
    """Test integration functionality."""
    
    def test_integrate_returns_result(self, integrator_em):
        """Test that integrate returns SDEIntegrationResult."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        result = integrator_em.integrate(x0, u_func, t_span)
        
        assert hasattr(result, 't')
        assert hasattr(result, 'x')
        assert hasattr(result, 'success')
        assert hasattr(result, 'nsteps')
    
    def test_integrate_with_t_eval(self, integrator_em):
        """Test integration with specific evaluation times."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        t_eval = np.linspace(0, 1, 51)
        
        result = integrator_em.integrate(x0, u_func, t_span, t_eval=t_eval)
        
        assert result.success
        assert len(result.t) == len(t_eval)
    
    def test_step_method(self, integrator_em):
        """Test single step method."""
        x0 = np.array([1.0])
        u = None
        dt = 0.01
        
        x1 = integrator_em.step(x0, u, dt)
        
        assert x1.shape == x0.shape
        assert not np.array_equal(x0, x1)  # Should change
    
    def test_statistics_tracked(self, integrator_em):
        """Test that statistics are tracked during integration."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 0.5)
        
        integrator_em.reset_stats()
        result = integrator_em.integrate(x0, u_func, t_span)
        
        stats = integrator_em.get_sde_stats()
        assert stats['total_fev'] > 0
        assert stats['diffusion_evals'] > 0


# ============================================================================
# Test Class: Algorithm Selection
# ============================================================================

class TestAlgorithmSelection:
    """Test algorithm recommendation and information."""
    
    def test_list_algorithms(self):
        """Test that list_algorithms returns categories."""
        algorithms = DiffEqPySDEIntegrator.list_algorithms()
        
        assert 'euler_maruyama' in algorithms
        assert 'stochastic_rk' in algorithms
        assert 'implicit' in algorithms
        assert isinstance(algorithms['euler_maruyama'], list)
    
    def test_get_algorithm_info(self):
        """Test getting algorithm information."""
        info = DiffEqPySDEIntegrator.get_algorithm_info('EM')
        
        assert 'name' in info
        assert 'description' in info
        assert info['strong_order'] == 0.5
        assert info['weak_order'] == 1.0
    
    def test_recommend_algorithm_additive(self):
        """Test algorithm recommendation for additive noise."""
        alg = DiffEqPySDEIntegrator.recommend_algorithm(
            noise_type='additive',
            stiffness='none',
            accuracy='high'
        )
        
        assert alg == 'SRA3'
    
    def test_recommend_algorithm_diagonal(self):
        """Test algorithm recommendation for diagonal noise."""
        alg = DiffEqPySDEIntegrator.recommend_algorithm(
            noise_type='diagonal',
            stiffness='none',
            accuracy='high'
        )
        
        assert alg == 'SRIW1'
    
    def test_recommend_algorithm_stiff(self):
        """Test algorithm recommendation for stiff systems."""
        alg = DiffEqPySDEIntegrator.recommend_algorithm(
            noise_type='any',
            stiffness='severe',
            accuracy='medium'
        )
        
        assert alg == 'ImplicitEM'
    
    def test_recommend_algorithm_low_accuracy(self):
        """Test algorithm recommendation for low accuracy."""
        alg = DiffEqPySDEIntegrator.recommend_algorithm(
            noise_type='any',
            stiffness='none',
            accuracy='low'
        )
        
        assert alg == 'EM'


# ============================================================================
# Test Class: Convergence and Accuracy
# ============================================================================

class TestConvergenceAccuracy:
    """Test convergence properties and accuracy."""
    
    def test_ou_exponential_decay_trend(self, ou_system):
        """Test that OU process shows exponential decay trend."""
        integrator = DiffEqPySDEIntegrator(
            ou_system,
            dt=0.001,
            algorithm='SRIW1',
            seed=42
        )
        
        x0 = np.array([5.0])  # Start far from equilibrium
        u_func = lambda t, x: None
        t_span = (0.0, 3.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        # State should generally decrease toward zero
        assert abs(result.x[-1, 0]) < abs(x0[0])
    
    def test_gbm_positive_paths(self, gbm_system):
        """Test that GBM stays positive (multiplicative noise)."""
        integrator = DiffEqPySDEIntegrator(
            gbm_system,
            dt=0.01,
            algorithm='EM',
            seed=42
        )
        
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        result = integrator.integrate(x0, u_func, t_span)
        
        # All states should remain positive (GBM property)
        assert np.all(result.x > 0)
    
    def test_higher_order_more_accurate(self, ou_system):
        """Test that higher-order methods are more accurate."""
        # This is a qualitative test - higher order should give
        # better agreement with analytical mean
        
        # Low order
        integrator_em = DiffEqPySDEIntegrator(
            ou_system,
            dt=0.01,
            algorithm='EM',
            seed=42
        )
        
        # High order
        integrator_sriw1 = DiffEqPySDEIntegrator(
            ou_system,
            dt=0.01,
            algorithm='SRIW1',
            seed=42
        )
        
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 1.0)
        
        # Multiple paths for averaging
        n_paths = 50
        
        means_em = []
        means_sriw1 = []
        
        for i in range(n_paths):
            integrator_em.seed = 500 + i
            integrator_em._rng = integrator_em._initialize_rng('numpy', 500 + i)
            result_em = integrator_em.integrate(x0, u_func, t_span)
            means_em.append(result_em.x[-1, 0])
            
            integrator_sriw1.seed = 500 + i
            integrator_sriw1._rng = integrator_sriw1._initialize_rng('numpy', 500 + i)
            result_sriw1 = integrator_sriw1.integrate(x0, u_func, t_span)
            means_sriw1.append(result_sriw1.x[-1, 0])
        
        # Expected mean: E[X(1)] = 1.0 * exp(-1.0 * 1.0) ≈ 0.368
        expected_mean = 1.0 * np.exp(-1.0)
        
        error_em = abs(np.mean(means_em) - expected_mean)
        error_sriw1 = abs(np.mean(means_sriw1) - expected_mean)
        
        # SRIW1 should be more accurate (or at least not worse)
        # Allow some statistical noise
        assert error_sriw1 <= error_em * 1.5


# ============================================================================
# Test Class: Edge Cases and Error Handling
# ============================================================================

class TestEdgeCasesErrorHandling:
    """Test edge cases and error handling."""
    
    def test_zero_time_span(self, integrator_em):
        """Test integration with zero time span."""
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 0.0)
        
        # Should handle gracefully
        result = integrator_em.integrate(x0, u_func, t_span)
        
        # May succeed with single point or fail gracefully
        assert len(result.t) >= 1
    
    def test_very_small_dt(self, ou_system):
        """Test with very small time step."""
        integrator = DiffEqPySDEIntegrator(
            ou_system,
            dt=1e-6,
            algorithm='EM'
        )
        
        x0 = np.array([1.0])
        u_func = lambda t, x: None
        t_span = (0.0, 0.001)  # Very short span
        
        result = integrator.integrate(x0, u_func, t_span)
        
        assert result.success


# ============================================================================
# Test Class: Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Test utility and helper functions."""
    
    def test_create_diffeqpy_sde_integrator(self, ou_system):
        """Test factory function."""
        integrator = create_diffeqpy_sde_integrator(
            ou_system,
            algorithm='EM',
            dt=0.01,
            rtol=1e-6
        )
        
        assert isinstance(integrator, DiffEqPySDEIntegrator)
        assert integrator.algorithm == 'EM'
        assert integrator.rtol == 1e-6
    
    def test_list_julia_sde_algorithms_output(self, capsys):
        """Test that list function prints output."""
        list_julia_sde_algorithms()
        
        captured = capsys.readouterr()
        assert 'Julia SDE Algorithms' in captured.out
        assert 'Euler-Maruyama' in captured.out


# ============================================================================
# Test Class: Comparison with Analytical Solutions
# ============================================================================

class TestAnalyticalComparison:
    """Compare numerical results with known analytical solutions."""
    
    def test_ou_stationary_variance(self, ou_system):
        """Test OU reaches stationary variance."""
        integrator = DiffEqPySDEIntegrator(
            ou_system,
            dt=0.01,
            algorithm='EM',
            seed=42
        )
        
        x0 = np.array([0.0])
        u_func = lambda t, x: None
        t_span = (0.0, 10.0)  # Long time for stationarity
        
        # Multiple trajectories
        n_paths = 200
        final_states = []
        
        for i in range(n_paths):
            integrator.seed = 600 + i
            integrator._rng = integrator._initialize_rng('numpy', 600 + i)
            result = integrator.integrate(x0, u_func, t_span)
            final_states.append(result.x[-1, 0])
        
        # Stationary variance: sigma^2 / (2*alpha) = 0.5^2 / (2*1.0) = 0.125
        expected_variance = 0.125
        observed_variance = np.var(final_states)
        
        # Allow 30% error
        assert abs(observed_variance - expected_variance) < 0.3 * expected_variance


# ============================================================================
# Test Class: String Representations
# ============================================================================

class TestStringRepresentations:
    """Test string representations."""
    
    def test_integrator_name(self, integrator_em):
        """Test integrator name property."""
        name = integrator_em.name
        
        assert 'Julia' in name
        assert 'EM' in name
        assert 'Adaptive' in name or 'Fixed' in name
    
    def test_repr(self, integrator_em):
        """Test __repr__ method."""
        repr_str = repr(integrator_em)
        
        assert 'DiffEqPySDEIntegrator' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])