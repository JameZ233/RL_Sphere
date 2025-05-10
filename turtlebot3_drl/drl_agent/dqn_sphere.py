import numpy as np
import math
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from ..common.settings import DQN_ACTION_SIZE, TARGET_UPDATE_FREQUENCY
from ..common.ounoise import OUNoise
from ..common.settings import ENABLE_BACKWARD, SPEED_LINEAR_MAX, SPEED_ANGULAR_MAX
from ..drl_environment.drl_environment import MAX_GOAL_DISTANCE
from .off_policy_agent import OffPolicyAgent, Network
from .dqn import DQN, POSSIBLE_ACTIONS
from .position_controller import PositionController

LINEAR = 0
ANGULAR = 1

class DQNWithSphere(DQN):
    def __init__(self, device, sim_speed, controller_params, blend=0.5):
        # Initialize base DQN (sets up actor, target, optimizer, epsilon schedule)
        super().__init__(device, sim_speed)

        # Classical controller for low-level tracking
        self.controller = PositionController(**controller_params)
        self.blend = blend
        self.max_lin = SPEED_LINEAR_MAX
        self.max_ang = SPEED_ANGULAR_MAX
        self.prev_time = None

    def reset(self):
        # If your agent has any per-episode state, reset here
        self.controller.reset()
        self.prev_time = None

    def get_pure_action(self, state, is_training, step, visualize=False):
        # Use DQN’s discrete selection (with ε-greedy)
        return super().get_action(state, is_training, step, visualize)

    def get_action(self, state, is_training, step, visualize=False):
        # 1) Choose discrete action index (pure RL)
        idx = self.get_pure_action(state, is_training, step, visualize)

        # 2) Map index to continuous set-point [V_set, w_set]
        rl_V, rl_w = POSSIBLE_ACTIONS[idx]

        # 3) Extract geometry from state
        d       = state[-4] * MAX_GOAL_DISTANCE
        alpha_e = state[-3] * math.pi

        # 4) Compute dt
        now = time.time()
        dt  = now - self.prev_time if self.prev_time else 0.0
        self.prev_time = now

        # 5) Classical controller outputs V_ctrl, w_ctrl
        V_ctrl, w_ctrl = self.controller(d, alpha_e, dt)

        # 6) Blend RL set-point + classical correction
        V_mix = self.blend * rl_V     + (1.0 - self.blend) * V_ctrl
        w_mix = self.blend * rl_w     + (1.0 - self.blend) * w_ctrl

        # 7) Normalize into [-1,1] action space
        if ENABLE_BACKWARD:
            a_lin = V_mix / self.max_lin
        else:
            a_lin = 2 * (V_mix / self.max_lin) - 1
        a_ang = w_mix / self.max_ang

        # 8) Clip and return a Python list of floats
        return [float(np.clip(a_lin, -1.0, 1.0)),
                float(np.clip(a_ang, -1.0, 1.0))]

    # train(), get_action_random(), etc. inherited unchanged from DQN
