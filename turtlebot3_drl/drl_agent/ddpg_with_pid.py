import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from turtlebot3_drl.drl_environment.reward import REWARD_FUNCTION
from ..common.settings import ENABLE_BACKWARD, ENABLE_STACKING

from ..common.ounoise import OUNoise
from ..drl_environment.drl_environment import NUM_SCAN_SAMPLES
from .ddpg import DDPG

from .off_policy_agent import OffPolicyAgent, Network

LINEAR = 0
ANGULAR = 1

# -----------------------
# 1) A simple PID class
# -----------------------

class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self._integral = 0.0
        self._prev_error = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = None

    def compute(self, setpoint, measurement):
        error = setpoint - measurement
        self._integral += error * self.dt
        derivative = 0.0 if (self._prev_error is None) else (error - self._prev_error) / self.dt
        self._prev_error = error

        # PID formula
        return self.kp * error + self.ki * self._integral + self.kd * derivative



# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py

class DDPGWithPID(DDPG):
    def __init__(self, device, sim_speed, pid_params, blend=0.5):
        """
        pid_params: dict with keys 'kp','ki','kd','dt'
        blend: how much weight on RL vs PID (0.0 = all PID, 1.0 = all RL)
        """
        super().__init__(device, sim_speed)
        # now that step_time exists, fill it in
        pid_params = { **pid_params, 'dt': self.step_time }
        self.pid = PIDController(**pid_params)
        self.blend = blend

    def reset(self):
        super().reset()
        self.pid.reset()

    def get_action(self, state, is_training, step, visualize=False):
        # 1) Raw RL action (e.g. desired roll‐velocity vector of size 2)
        rl_action = super().get_action(state, is_training, step, visualize)
        
        # 2) Read out the actual measurement from your environment/state
        actual = self._extract_robot_velocity(state)  # implement this
        
        # 3) Compute PID correction
        pid_action = [
            self.pid.compute(sp, meas) 
            for sp, meas in zip(rl_action, actual)
        ]
        
        # 4) Blend RL + PID
        final_action = [
            self.blend * r + (1.0 - self.blend) * p
            for r, p in zip(rl_action, pid_action)
        ]
        
        # 5) Clip to [-1,1] and return **native Python floats**
        clipped = [float(max(-1.0, min(1.0, x))) for x in final_action]
        return clipped

    def _extract_robot_velocity(self, state):
        # user‐specific: pull out the sphere‐robot’s current velocity
        # from your state vector or simulator API
        return [state[-2], state[-1]]  
