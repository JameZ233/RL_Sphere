import numpy as np
import copy
import math
import time
import torch # type: ignore
import torch.nn.functional as F # type: ignore
import torch.nn as nn # type: ignore
from turtlebot3_drl.drl_environment.reward import REWARD_FUNCTION
from .off_policy_agent import OffPolicyAgent, Network
from .ddpg import DDPG
from ..common.ounoise import OUNoise
from .position_controller import PositionController
from ..common.settings import ENABLE_BACKWARD, ENABLE_STACKING, SPEED_LINEAR_MAX, SPEED_ANGULAR_MAX
from ..drl_environment.drl_environment import NUM_SCAN_SAMPLES, MAX_GOAL_DISTANCE

LINEAR = 0
ANGULAR = 1

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py

class DDPGWithSphere(DDPG):
    def __init__(self, device, sim_speed, controller_params, blend=0.5):
        # Initialize base DDPG (sets up actor, critic, noise, etc.)
        super().__init__(device, sim_speed)

        # Integrate the classical PositionController
        self.controller = PositionController(**controller_params)
        self.blend = blend
        self.max_lin = SPEED_LINEAR_MAX
        self.max_ang = SPEED_ANGULAR_MAX
        self.prev_time = None

    def reset(self):
        super().reset()
        self.controller.reset()
        self.prev_time = None

    def get_action(self, state, is_training, step, visualize=False):
        # 1) High‑level RL policy output (normalized [-1,1])
        rl_norm = super().get_action(state, is_training, step, visualize)

        # 2) Extract distance & bearing error from state
        d       = state[-4] * MAX_GOAL_DISTANCE   # goal distance normalized
        alpha_e = state[-3] * math.pi             # goal angle normalized

        # 3) Compute dt
        now = time.time()
        dt  = now - self.prev_time if self.prev_time else 0.0
        self.prev_time = now

        # 4) Classical controller outputs V, w
        V_ctrl, w_ctrl = self.controller(d, alpha_e, dt)

        # 5) Normalize to action range [-1,1]
        if ENABLE_BACKWARD:
            a_ctrl_lin = V_ctrl / self.max_lin
        else:
            a_ctrl_lin = 2 * (V_ctrl / self.max_lin) - 1
        a_ctrl_ang = w_ctrl / self.max_ang

        # 6) Blend RL + Classical
        raw = [
            self.blend * float(rl) + (1.0 - self.blend) * float(ctrl)
            for rl, ctrl in zip(rl_norm, [a_ctrl_lin, a_ctrl_ang])
        ]

        # 7) Clip and return
        clipped = [min(max(x, -1.0), 1.0) for x in raw]
        return clipped