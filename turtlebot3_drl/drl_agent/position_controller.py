import math

class PositionController:
    """Position controller for a pendulum-driven spherical robot."""

    def __init__(self, k_alpha, k_alpha_acc, V_max=0.3, k_r=0.1, k_t=1/1000):
        """
        Args:
            k_alpha (float): Proportional gain for angular error.
            k_alpha_acc (float): Gain for accumulated angular error.
            V_max (float): Maximum linear velocity (m/s).
            k_r (float): Distance threshold (m) for linear scaling.
            k_t (float): Scaling factor for the integral of angular error.
        """
        self.k_alpha = k_alpha
        self.k_alpha_acc = k_alpha_acc
        self.V_max = V_max
        self.k_r = k_r
        self.k_t = k_t
        self.alpha_eacc = 0.0

    def reset(self):
        """Reset the accumulated angular error."""
        self.alpha_eacc = 0.0

    def __call__(self, d, alpha_e, dt):
        """
        Args:
          d       (float): distance to target [m]
          alpha_e (float): angular error [rad]
          dt      (float): timestep [s]
        Returns:
          V (float): linear speed command [m/s]
          w (float): angular speed command [rad/s]
        """
        # Accumulate error for integral term
        self.alpha_eacc += self.k_t * alpha_e * dt

        # Angular (PID‐style) control
        w = -self.k_alpha * alpha_e - self.k_alpha_acc * self.alpha_eacc

        # Linear “slow‐in” control
        if abs(d) > self.k_r:
            V = self.V_max
        else:
            V = (d * self.V_max) / self.k_r

        return V, w

def normalize_angle(angle):
    """Wrap angle to [-π, π]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

# Usage example:
# controller = PositionController(k_alpha=2.18, k_alpha_acc=238.74)
# V, w = controller(x, y, theta, x_goal, y_goal, dt)

