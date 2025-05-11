import numpy as np
import math
import time
import torch
from ..common.settings import POLICY_NOISE, POLICY_NOISE_CLIP, POLICY_UPDATE_FREQUENCY
from ..common.ounoise import OUNoise
from ..common.settings import ENABLE_BACKWARD, SPEED_LINEAR_MAX, SPEED_ANGULAR_MAX
from ..drl_environment.drl_environment import MAX_GOAL_DISTANCE
from .off_policy_agent import OffPolicyAgent, Network
from .td3 import TD3
from .position_controller import PositionController

class TD3WithSphere(TD3):
    def __init__(self, device, sim_speed, controller_params, blend=0.5):
        """
        controller_params: dict for PositionController (kp, ki, kd, V_max, k_r, k_t)
        blend: weight for RL vs controller (0=all PID, 1=all RL)
        """
        super().__init__(device, sim_speed)
        self.controller = PositionController(**controller_params)
        self.blend = blend
        self.max_lin = SPEED_LINEAR_MAX
        self.max_ang = SPEED_ANGULAR_MAX
        self.prev_time = None

    def reset(self):
        super().reset()
        self.controller.reset()
        self.prev_time = None

    def get_pure_action(self, state, is_training, step, visualize=False):
        # Underlying TD3 action (with exploration noise)
        return super().get_action(state, is_training, step, visualize)

    def get_action(self, state, is_training, step, visualize=False):
        # 1) Pure RL action (normalized [-1,1] for lin/ang)
        rl_norm = self.get_pure_action(state, is_training, step, visualize)

        # 2) Extract geometry: distance and heading error
        d = state[-4] * MAX_GOAL_DISTANCE       # un-normalize distance
        alpha_e = state[-3] * math.pi           # un-normalize angle

        # 3) Compute dt
        now = time.time()
        dt = now - self.prev_time if self.prev_time else 0.0
        self.prev_time = now

        # 4) PID controller outputs (m/s, rad/s)
        V_ctrl, w_ctrl = self.controller(d, alpha_e, dt)

        # 5) Convert PID outputs to normalized action range [-1,1]
        if ENABLE_BACKWARD:
            a_ctrl_lin = V_ctrl / self.max_lin
        else:
            a_ctrl_lin = 2 * (V_ctrl / self.max_lin) - 1
        a_ctrl_ang = w_ctrl / self.max_ang

        # 6) Blend RL and PID
        raw_lin = self.blend * rl_norm[0] + (1.0 - self.blend) * a_ctrl_lin
        raw_ang = self.blend * rl_norm[1] + (1.0 - self.blend) * a_ctrl_ang

        # 7) Clip and return as Python floats
        lin = float(np.clip(raw_lin, -1.0, 1.0))
        ang = float(np.clip(raw_ang, -1.0, 1.0))
        return [lin, ang]

    # train() and get_action_random() inherited from TD3
