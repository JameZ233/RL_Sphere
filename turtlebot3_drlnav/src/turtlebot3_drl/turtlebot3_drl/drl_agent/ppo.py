#!/usr/bin/env python3
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common.settings import ENABLE_STACKING
from .off_policy_agent import OffPolicyAgent, Network
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


def init_weights(module: nn.Module):
    """
    Orthogonal initialization for Linear layers. Final mu_head gets gain=0.01.
    """
    if isinstance(module, nn.Linear):
        gain = 0.01 if getattr(module, 'is_mu_head', False) else 1.0
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


class ActorPPO(Network):
    def __init__(self, name, state_size, action_size, hidden_size, train_log_std=True):
        super().__init__(name)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu_head = nn.Linear(hidden_size, action_size)
        self.mu_head.is_mu_head = True
        self.log_std = nn.Parameter(torch.zeros(action_size), requires_grad=train_log_std)
        self.apply(init_weights)

    def forward(self, states: torch.Tensor):
        x = torch.tanh(self.fc1(states))
        x = torch.tanh(self.fc2(x))
        mu = self.mu_head(x)
        log_std = self.log_std.clamp(-5.0, 2.0)
        std = log_std.exp().clamp(min=1e-3, max=1.0)
        return mu, std

    def get_dist(self, states: torch.Tensor) -> TransformedDistribution:
        mu, std = self.forward(states)
        base = Normal(mu, std)
        return TransformedDistribution(base, [TanhTransform(cache_size=1)])


class CriticV(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super().__init__(name)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.v_head = nn.Linear(hidden_size, 1)
        self.apply(super().init_weights)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(states))
        x = torch.tanh(self.fc2(x))
        return self.v_head(x).squeeze(-1)


class PPO(OffPolicyAgent):
    def __init__(
        self,
        device,
        sim_speed,
        hidden_size=256,
        gamma=0.99,
        lam=0.95,
        entropy_coef=1e-3,
        value_coef=0.5,
        clip_param=0.2,
        ppo_epochs=10,
        batch_size=64,
        train_log_std=False
    ):
        super().__init__(device, sim_speed)
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.actor = self.create_network(
            lambda *args: ActorPPO(*args, train_log_std=train_log_std), 'actor'
        )
        self.critic = self.create_network(CriticV, 'critic')

        self.actor_opt = self.create_optimizer(self.actor)
        self.critic_opt = self.create_optimizer(self.critic)

        self.reset_buffer()
        self.last_ploss = torch.tensor(0.0)
        self.last_vloss = torch.tensor(0.0)

    def reset_buffer(self):
        self.ep_states = []
        self.ep_actions = []
        self.ep_logp = []
        self.ep_rewards = []
        self.ep_values = []
        self.ep_dones = []

    def _sample(self, s: torch.Tensor, train: bool):
        dist = self.actor.get_dist(s)
        raw_a = dist.rsample() if train else dist.mean
        logp = dist.log_prob(raw_a).sum(-1)
        return raw_a, logp

    def get_action(self, state, is_training, step, visualize=False):
        s = torch.from_numpy(np.asarray(state, np.float32)).to(self.device)
        with torch.no_grad():
            raw_a, _ = self._sample(s, is_training)
        return torch.tanh(raw_a).cpu().numpy().tolist()

    def get_action_random(self):
        return np.random.uniform(-1.0, 1.0, size=self.action_size).tolist()

    def compute_gae(self, next_value: torch.Tensor):
        gae = 0
        advantages = []
        values = self.ep_values + [next_value]
        for i in reversed(range(len(self.ep_rewards))):
            delta = (
                self.ep_rewards[i]
                + self.gamma * values[i+1] * (1 - self.ep_dones[i])
                - values[i]
            )
            gae = delta + self.gamma * self.lam * (1 - self.ep_dones[i]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, self.ep_values)]
        return advantages, returns

    def _update(self):
        states = torch.stack(self.ep_states).to(self.device)
        actions = torch.stack(self.ep_actions).to(self.device)
        old_logp = torch.stack(self.ep_logp).to(self.device)
        with torch.no_grad():
            next_value = self.critic(states[-1].unsqueeze(0))
        advs, rets = self.compute_gae(next_value)
        ad = torch.tensor(advs, dtype=torch.float32, device=self.device)
        rt = torch.tensor(rets, dtype=torch.float32, device=self.device)
        ad = (ad - ad.mean()) / (ad.std() + 1e-8)

        dataset = torch.utils.data.TensorDataset(states, actions, old_logp, rt, ad)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.ppo_epochs):
            for b_s, b_a, b_old_logp, b_ret, b_adv in loader:
                dist = self.actor.get_dist(b_s)
                b_new_logp = dist.log_prob(b_a).sum(-1)
                b_val = self.critic(b_s)

                ratio = torch.exp(b_new_logp - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(b_val, b_ret)
                entropy = dist.base_dist.entropy().sum(-1).mean()

                total_loss = (policy_loss - self.entropy_coef * entropy) + self.value_coef * value_loss

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                total_loss.backward()
                self.actor_opt.step()
                self.critic_opt.step()

                self.last_ploss = policy_loss.detach()
                self.last_vloss = value_loss.detach()

        self.reset_buffer()

    def _train(self, replaybuffer):
        """
        Override off-policy _train: consume entire buffer as on-policy data,
        then perform update and clear buffer.
        Returns (critic_loss, actor_loss)
        """
        # Accumulate on-policy training data from replay buffer
        for s_np, a_np, r_list, ns_np, d_list in replaybuffer.buffer:
            # Prepare state and action tensors
            s = torch.from_numpy(np.asarray(s_np, np.float32)).to(self.device)
            a = torch.tensor(a_np, dtype=torch.float32).to(self.device)
            r = r_list[0]
            d = d_list[0]

            # Compute old log probability and value under current policy without tracking gradient
            dist = self.actor.get_dist(s.unsqueeze(0))
            with torch.no_grad():
                logp = dist.log_prob(a.unsqueeze(0)).sum(-1)
                value = self.critic(s.unsqueeze(0))
            # Detach to treat as fixed targets
            logp = logp.detach()
            value = value.detach()

            # Store for GAE and policy update
            self.ep_states.append(s)
            self.ep_actions.append(a)
            self.ep_logp.append(logp)
            self.ep_values.append(value)
            self.ep_rewards.append(r)
            self.ep_dones.append(d)

        # Clear buffer to avoid reusing data
        replaybuffer.buffer.clear()
        # Perform PPO update
        self._update()
        # Return critic and actor losses for logging
        return self.last_vloss.item(), self.last_ploss.item()

    def train(self, *args, **kwargs):
        raise NotImplementedError("Use _train() for on-policy update.")


# -------------------------
# Debugging Reports:
# 1. 在 get_action() 中打印 raw_a 与 action：
#    raw_a, _ = self._sample(s, is_training)
#    print(f"raw_a: {raw_a.cpu().numpy()}, action: {torch.tanh(raw_a).cpu().numpy()}")
# 2. 使用 get_action_random() 验证环境响应：
#    action = self.get_action_random()
#    print(f"random action: {action}")
# 3. 检查动作尺度和 step_time 参数：
#    动作范围[-1,1]是否映射正确，step_time 是否过小导致看不见移动
# -------------------------
