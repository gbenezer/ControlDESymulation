import numpy as np
import torch
from typing import Optional


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) for nonlinear state estimation.

    The EKF estimates the state of a nonlinear system by:
    1. Propagating state through NONLINEAR dynamics
    2. Re-linearizing at the current estimate each time step
    3. Using the linearization to propagate uncertainty
    4. Updating based on measurements

    Theory:
    ------
    **Predict Step**:
        x̂[k|k-1] = f(x̂[k-1|k-1], u[k-1])               [Nonlinear dynamics]
        A[k] = ∂f/∂x |_{x̂[k-1|k-1], u[k-1]}            [Linearization at current estimate]
        P[k|k-1] = A[k] P[k-1|k-1] A[k]^T + Q          [Covariance propagation]

    **Update Step**:
        ŷ[k|k-1] = h(x̂[k|k-1])                         [Nonlinear observation]
        C[k] = ∂h/∂x |_{x̂[k|k-1]}                      [Observation Jacobian]
        S[k] = C[k] P[k|k-1] C[k]^T + R                [Innovation covariance]
        K[k] = P[k|k-1] C[k]^T S[k]^{-1}               [Kalman gain - varies with time!]
        x̂[k|k] = x̂[k|k-1] + K[k](y[k] - ŷ[k|k-1])      [State update]
        P[k|k] = (I - K[k]C[k]) P[k|k-1]               [Covariance update]

    Attributes:
        system: SymbolicDynamicalSystem or GenericDiscreteTimeSystem
        Q: Process noise covariance (nx, nx)
        R: Measurement noise covariance (ny, ny)
        x_hat: Current state estimate (nx,)
        P: Current covariance estimate (nx, nx)
        is_discrete: Whether system is discrete or continuous

    Example:
        >>> # Create EKF for pendulum
        >>> pendulum = SymbolicPendulum(m=0.15, l=0.5, beta=0.1, g=9.81)
        >>> Q_process = np.diag([0.001, 0.01])
        >>> R_measurement = np.array([[0.1]])
        >>>
        >>> ekf = ExtendedKalmanFilter(pendulum, Q_process, R_measurement)
        >>>
        >>> # Initialize at origin
        >>> ekf.reset(x0=torch.tensor([0.1, 0.0]))
        >>>
        >>> # Estimation loop
        >>> for t in range(num_steps):
        >>>     # Predict
        >>>     ekf.predict(u[t], dt=0.01)
        >>>
        >>>     # Get noisy measurement
        >>>     y_measured = measure_angle(x_true[t]) + np.random.randn() * 0.1
        >>>
        >>>     # Update
        >>>     ekf.update(torch.tensor([y_measured]))
        >>>
        >>>     # Get estimate
        >>>     x_estimate = ekf.x_hat
        >>>     uncertainty = ekf.P
        >>>
        >>> # EKF can track large swings
        >>> # Unlike constant-gain observer which is only valid near equilibrium

    Notes:
        - Process noise Q represents model uncertainty and disturbances
        - Measurement noise R represents sensor characteristics
        - Larger Q → trust measurements more (higher gain)
        - Larger R → trust model more (lower gain)
        - Covariance P tracks estimate uncertainty
        - Can be used with nonlinear controllers (MPC, feedback linearization)

    See Also:
        kalman_gain: Constant-gain observer for linear systems
        LinearObserver: Linear observer with constant gain
        discrete_kalman_gain: Discrete-time constant-gain design
    """

    def __init__(self, system, Q_process: np.ndarray, R_measurement: np.ndarray):
        """
        Initialize Extended Kalman Filter.

        Args:
            system: SymbolicDynamicalSystem or GenericDiscreteTimeSystem
                   Must have forward(), h(), linearized_dynamics(), and
                   linearized_observation() methods
            Q_process: Process noise covariance (nx, nx). Represents model
                      uncertainty and unmodeled disturbances.
            R_measurement: Measurement noise covariance (ny, ny). Represents
                          sensor noise characteristics.

        Example:
            >>> system = SymbolicQuadrotor2D()
            >>> Q = np.eye(6) * 0.01  # Low process noise
            >>> R = np.eye(3) * 0.1   # Moderate measurement noise
            >>> ekf = ExtendedKalmanFilter(system, Q, R)
        """
        self.system = system
        self.Q = Q_process
        self.R = R_measurement

        # State estimate and covariance
        self.x_hat = system.x_equilibrium.clone()
        self.P = torch.eye(system.nx) * 0.1

        self.is_discrete = hasattr(system, "continuous_time_system")

    def predict(self, u: torch.Tensor, dt: Optional[float] = None):
        """
        EKF prediction step: propagate state estimate and covariance.

        Args:
            u: Control input (nu,)
            dt: Time step (required for continuous-time systems, ignored for discrete)

        Example:
            >>> ekf.predict(u=torch.tensor([1.0]), dt=0.01)
            >>> print(f"Predicted state: {ekf.x_hat}")
            >>> print(f"Predicted covariance: {ekf.P}")

        Notes:
            - Must call predict() before update() in each cycle
            - For discrete systems, dt is ignored
            - For continuous systems, uses Euler integration
            - Covariance grows during prediction (adds Q)
        """
        if self.is_discrete:
            # Discrete system: x̂[k+1|k] = f(x̂[k|k], u[k])
            with torch.no_grad():
                self.x_hat = self.system(self.x_hat, u)

            # Propagate covariance: P[k+1|k] = A P[k|k] A^T + Q
            A, _ = self.system.linearized_dynamics(
                self.x_hat.unsqueeze(0), u.unsqueeze(0)
            )
            A = A.squeeze()
        else:
            # Continuous system: integrate forward
            if dt is None:
                raise ValueError("dt required for continuous systems")

            with torch.no_grad():
                dx = self.system.forward(self.x_hat, u)
                self.x_hat = self.x_hat + dx * dt

            A, _ = self.system.linearized_dynamics(
                self.x_hat.unsqueeze(0), u.unsqueeze(0)
            )
            A = A.squeeze()
            A = torch.eye(self.system.nx) + A * dt  # Euler discretization

        Q_tensor = torch.tensor(self.Q, dtype=self.P.dtype, device=self.P.device)
        self.P = A @ self.P @ A.T + Q_tensor

    def update(self, y_measurement: torch.Tensor):
        """
        EKF update step: correct estimate using measurement.

        Args:
            y_measurement: Measurement vector (ny,). Should match the
                          output dimension of h(x).

        Example:
            >>> # After prediction
            >>> y_measured = torch.tensor([0.15, 2.1, 0.05])  # Noisy measurement
            >>> ekf.update(y_measured)
            >>> print(f"Updated state: {ekf.x_hat}")
            >>> print(f"Updated covariance: {ekf.P}")
            >>> print(f"Uncertainty reduced: {np.trace(ekf.P)}")

        Notes:
            - Covariance shrinks during update (information gained)
            - Large innovation → either bad estimate or bad measurement
            - Gain K[k] adapts based on current uncertainty P
            - Must call predict() before update()
        """
        # Ensure y_measurement is 1D
        if len(y_measurement.shape) == 0:
            y_measurement = y_measurement.unsqueeze(0)

        # Predicted measurement
        with torch.no_grad():
            if self.is_discrete:
                y_pred = self.system.continuous_time_system.h(
                    self.x_hat.unsqueeze(0)
                ).squeeze()
            else:
                y_pred = self.system.h(self.x_hat.unsqueeze(0)).squeeze()

        # Ensure y_pred is 1D
        if len(y_pred.shape) == 0:
            y_pred = y_pred.unsqueeze(0)

        # Measurement residual (innovation)
        innovation = y_measurement - y_pred
        if len(innovation.shape) == 0:
            innovation = innovation.unsqueeze(0)

        # Get measurement Jacobian
        if self.is_discrete:
            C = self.system.continuous_time_system.linearized_observation(
                self.x_hat.unsqueeze(0)
            ).squeeze()
        else:
            C = self.system.linearized_observation(self.x_hat.unsqueeze(0)).squeeze()

        # Ensure C is 2D (ny, nx)
        if len(C.shape) == 1:
            C = C.unsqueeze(0)  # (ny, nx)

        # Innovation covariance: S = C P C^T + R
        R_tensor = torch.tensor(self.R, dtype=self.P.dtype, device=self.P.device)
        S = C @ self.P @ C.mT + R_tensor  # Use .mT for matrix transpose

        # Ensure S is 2D
        if len(S.shape) == 0:
            S = S.unsqueeze(0).unsqueeze(0)
        elif len(S.shape) == 1:
            S = S.unsqueeze(0)

        # Kalman gain: K = P C^T S^{-1}
        Kt = self.P @ C.mT @ torch.inverse(S)  # (nx, ny)

        # Update state estimate: x̂ = x̂ + K * innovation
        correction = (Kt @ innovation.unsqueeze(-1)).squeeze(-1)
        self.x_hat = self.x_hat + correction

        # Update covariance: P = (I - K C) P
        nx = (
            self.system.nx
            if not self.is_discrete
            else self.system.continuous_time_system.nx
        )
        I = torch.eye(nx, device=self.P.device, dtype=self.P.dtype)
        self.P = (I - Kt @ C) @ self.P

    def reset(
        self, x0: Optional[torch.Tensor] = None, P0: Optional[torch.Tensor] = None
    ):
        """
        Reset filter to initial state and covariance.

        Useful for:
        - Starting a new estimation sequence
        - Recovering from filter divergence
        - Testing different initial conditions

        Args:
            x0: Initial state estimate (nx,). Uses equilibrium if None.
            P0: Initial covariance (nx, nx). Uses 0.1*I if None.

        Example:
            >>> # Reset to known initial condition
            >>> ekf.reset(x0=torch.tensor([0.1, 0.0]),
            ...           P0=torch.eye(2) * 0.01)  # Low initial uncertainty
            >>>
            >>> # Reset to equilibrium with high uncertainty
            >>> ekf.reset()  # Uses default x_equilibrium and 0.1*I

        Notes:
            - Called automatically in __init__()
            - P0 represents initial uncertainty about x0
            - Larger P0 → less confident in initial estimate
            - After reset, start with predict() then update()
        """
        if x0 is not None:
            self.x_hat = x0.clone()
        else:
            if self.is_discrete:
                self.x_hat = self.system.x_equilibrium.clone()
            else:
                self.x_hat = self.system.x_equilibrium.clone()

        if P0 is not None:
            self.P = P0.clone()
        else:
            nx = (
                self.system.nx
                if not self.is_discrete
                else self.system.continuous_time_system.nx
            )
            self.P = torch.eye(nx) * 0.1


class LinearObserver:
    """
    Linear state observer with constant gain.

    Implements the observer:
        d x̂/dt = f(x̂, u) + L(y - h(x̂))                         [Continuous-time]
        x̂[k+1] = f(x̂[k], u[k]) + L(y[k+1] - h(f(x̂[k], u[k])))  [Discrete-time]

    where:
    - x̂ is the state estimate
    - L is the observer gain matrix
    - y is the measurement
    - h(x̂) is the predicted measurement

    Attributes:
        system: SymbolicDynamicalSystem or GenericDiscreteTimeSystem
        L: Observer gain matrix (nx, ny)
        x_hat: Current state estimate (nx,)

    Example - Continuous-Time Observer:
        >>> # Design Kalman gain
        >>> system = SymbolicPendulum(m=1.0, l=0.5)
        >>> Q_process = np.diag([0.001, 0.01])
        >>> R_measurement = np.array([[0.1]])
        >>> L = system.kalman_gain(Q_process, R_measurement)
        >>>
        >>> # Create observer
        >>> observer = LinearObserver(system, L)
        >>> observer.reset(x0=torch.zeros(2))
        >>>
        >>> # Observer loop
        >>> dt = 0.01
        >>> for t in range(num_steps):
        >>>     observer.update(u[t], y_measured[t], dt)
        >>>     x_estimate = observer.x_hat

    Example - Discrete-Time Observer:
        >>> # Create discrete system
        >>> system_ct = SymbolicQuadrotor2D()
        >>> system_dt = GenericDiscreteTimeSystem(system_ct, dt=0.01)
        >>>
        >>> # Design discrete Kalman gain
        >>> L = system_dt.discrete_kalman_gain(Q_process, R_measurement)
        >>>
        >>> # Create observer
        >>> observer = LinearObserver(system_dt, L)
        >>>
        >>> # Observer loop (no dt needed for discrete)
        >>> for k in range(num_steps):
        >>>     observer.update(u[k], y[k], dt=None)  # dt ignored for discrete
        >>>     x_estimate = observer.x_hat

    Example - Output Feedback Control:
        >>> # Design LQG controller
        >>> K, L = system.lqg_control(Q_lqr, R_lqr, Q_process, R_measurement)
        >>>
        >>> # Create controller and observer
        >>> controller = LinearController(K, system.x_equilibrium, system.u_equilibrium)
        >>> observer = LinearObserver(system, L)
        >>>
        >>> # Closed-loop with output feedback
        >>> x_true = x0
        >>> observer.reset(x0=system.x_equilibrium)  # Start from equilibrium guess
        >>>
        >>> for t in range(num_steps):
        >>>     # Measure (with noise)
        >>>     y = system.h(x_true) + torch.randn(system.ny) * 0.1
        >>>
        >>>     # Update observer
        >>>     observer.update(u, y, dt)
        >>>
        >>>     # Compute control based on estimate
        >>>     u = controller(observer.x_hat)
        >>>
        >>>     # Update true system
        >>>     x_true = system(x_true, u) if discrete else integrate(x_true, u, dt)

    Notes:
        - Gain L is constant and designed at equilibrium
        - For nonlinear systems, only accurate near equilibrium
        - No covariance tracking
        - Can be used with both continuous and discrete systems
        - Lower computational cost than EKF

    See Also:
        kalman_gain: Design L for continuous systems
        discrete_kalman_gain: Design L for discrete systems
        ExtendedKalmanFilter: Adaptive nonlinear observer
        LinearController: State feedback controller
        lqg_control: Combined controller and observer design
    """

    def __init__(self, system, L: np.ndarray):
        """
        Initialize linear observer with constant gain.

        Args:
            system: SymbolicDynamicalSystem or GenericDiscreteTimeSystem
            L: Observer gain matrix (nx, ny). Typically from kalman_gain()
               or discrete_kalman_gain().

        Example:
            >>> L = system.kalman_gain(Q_process, R_measurement)
            >>> observer = LinearObserver(system, L)
        """
        self.system = system
        self.L = torch.tensor(L, dtype=torch.float32)
        self.x_hat = system.x_equilibrium.clone()

    def update(self, u: torch.Tensor, y: torch.Tensor, dt: float):
        """
        Update observer state estimate.

        Continuous-time:
            d x̂/dt = f(x̂, u) + L(y - h(x̂))
            x̂_new ≈ x̂_old + dt * [f(x̂, u) + L(y - h(x̂))]

        Discrete-time:
            x̂_pred = f(x̂, u)
            x̂_new = x̂_pred + L(y - h(x̂_pred))

        Args:
            u: Control input (nu,)
            y: Measurement (ny,)
            dt: Time step (used for continuous systems, ignored for discrete)

        Example:
            >>> u = torch.tensor([1.0])
            >>> y_measured = torch.tensor([0.15])  # Noisy angle measurement
            >>> observer.update(u, y_measured, dt=0.01)
            >>> print(f"Estimate: {observer.x_hat}")

        Notes:
            - For continuous systems: uses Euler integration
            - For discrete systems: dt parameter is ignored
            - Innovation = y - h(x̂_pred) is the measurement residual
            - Large innovation suggests either bad estimate or bad measurement
            - Gain L determines how much to trust innovation vs model
        """
        # Predict
        with torch.no_grad():
            if hasattr(self.system, "continuous_time_system"):
                # Discrete system
                x_pred = self.system(self.x_hat.unsqueeze(0), u.unsqueeze(0)).squeeze(0)
                y_pred = self.system.continuous_time_system.h(
                    x_pred.unsqueeze(0)
                ).squeeze(0)
            else:
                # Continuous system
                dx = self.system.forward(
                    self.x_hat.unsqueeze(0), u.unsqueeze(0)
                ).squeeze(0)
                x_pred = self.x_hat + dx * dt
                y_pred = self.system.h(x_pred.unsqueeze(0)).squeeze(0)

            # Correct
            innovation = y - y_pred
            self.x_hat = x_pred + (self.L @ innovation.unsqueeze(-1)).squeeze(-1)

    def reset(self, x0: Optional[torch.Tensor] = None):
        """
        Reset observer to initial state estimate.

        Args:
            x0: Initial state estimate (nx,). Uses equilibrium if None.

        Example:
            >>> # Reset to known initial condition
            >>> observer.reset(x0=torch.tensor([0.1, 0.0]))
            >>>
            >>> # Reset to equilibrium
            >>> observer.reset()  # Uses system.x_equilibrium

        Notes:
            - Called automatically in __init__()
            - Unlike EKF, no covariance to reset
            - Start observer from best available estimate
        """
        if x0 is not None:
            self.x_hat = x0.clone()
        else:
            self.x_hat = self.system.x_equilibrium.clone()

    def to(self, device):
        """
        Move observer to specified device (CPU/GPU).

        Args:
            device: torch.device or string ('cpu', 'cuda')

        Returns:
            Self for chaining
        """
        self.L = self.L.to(device)
        self.x_hat = self.x_hat.to(device)
        return self
