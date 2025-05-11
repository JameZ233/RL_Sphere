import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common.settings import ENABLE_STACKING
from .off_policy_agent import OffPolicyAgent, Network


class ActorPG(Network):
    """Gaussian policy with learnable log_std."""

    def __init__(self, name, state_size, action_size, hidden_size):
        super().__init__(name)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu_head = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))
        self.apply(super().init_weights)

    def forward(self, states):
        x = torch.tanh(self.fc1(states))
        x = torch.tanh(self.fc2(x))
        mu = self.mu_head(x)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mu, std


class CriticV(Network):
    """State‑value baseline V(s)."""

    def __init__(self, name, state_size, action_size, hidden_size):
        super().__init__(name)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.v_head = nn.Linear(hidden_size, 1)
        self.apply(super().init_weights)

    def forward(self, states):
        x = torch.tanh(self.fc1(states))
        x = torch.tanh(self.fc2(x))
        return self.v_head(x).squeeze(-1)  # shape [B]


class VPG(OffPolicyAgent):
    """Vanilla Policy Gradient with TD3‑compatible I/O."""

    def __init__(self, device, sim_speed, hidden_size=256, gamma=0.99,
                 entropy_coef=1e-3, value_coef=0.5):
        super().__init__(device, sim_speed)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.actor = self.create_network(ActorPG, 'actor')
        self.critic = self.create_network(CriticV, 'critic')

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=1e-3)

        self.reset_buffer()

    # ---------- action ----------
    def _sample(self, s: torch.Tensor, train: bool):
        mu, std = self.actor(s)
        dist = torch.distributions.Normal(mu, std)
        raw = dist.rsample() if train else mu
        tanh_a = torch.tanh(raw)
        logp = dist.log_prob(raw).sum(-1)
        return raw, tanh_a, logp

    def get_action(self, state, is_training, step, visualize=False):
        s = torch.from_numpy(np.asarray(state, np.float32)).to(self.device)
        with torch.no_grad():
            _, a, logp = self._sample(s, is_training)
        
        return a.cpu().numpy().tolist()

    def get_action_random(self):
        return np.random.uniform(-1.0, 1.0, self.action_size).tolist()

    # ---------- train ----------
    def train(self, s, a, r, ns, d):
        # tensorize
        if not isinstance(s, torch.Tensor):
            s = torch.from_numpy(np.asarray(s, np.float32)).to(self.device)
            a = torch.from_numpy(np.asarray(a, np.float32)).to(self.device)
            r = torch.tensor(r, dtype=torch.float32, device=self.device)
            d = torch.tensor(d, dtype=torch.bool, device=self.device)
        else:
            s, a = s.to(self.device), a.to(self.device)
            r, d = r.to(self.device).float(), d.to(self.device).bool()

        if s.ndim == 1:
            s, a, r, d = s.unsqueeze(0), a.unsqueeze(0), r.unsqueeze(0), d.unsqueeze(0)

        eps = 1e-6
        for si, ai, ri, di in zip(s, a, r, d):
            ai_clip = torch.clamp(ai, -1 + eps, 1 - eps)
            raw_ai = 0.5 * torch.log((1 + ai_clip) / (1 - ai_clip))

            mu, std = self.actor(si.unsqueeze(0))
            dist = torch.distributions.Normal(mu, std)
            logp_raw = dist.log_prob(raw_ai).sum(-1)
            log_det = torch.log(1 - ai_clip ** 2 + eps).sum(-1)
            logp = (logp_raw - log_det).squeeze()

            self.ep_states.append(si)
            self.ep_rewards.append(ri.squeeze())
            self.ep_logp.append(logp)

            if di:
                self._update()
                self.reset_buffer()

        return [self.last_vloss, self.last_ploss]

    # ---------- utils ----------
    def reset_buffer(self):
        self.ep_states, self.ep_rewards, self.ep_logp = [], [], []
        self.last_ploss = torch.tensor(0.0)
        self.last_vloss = torch.tensor(0.0)

    def _returns(self):
        G = 0.0
        rets = []
        for r in reversed(self.ep_rewards):
            G = r + self.gamma * G
            rets.insert(0, G)
        rets = torch.stack(rets)
        rets -= rets.mean()
        std = rets.std()
        if std > 1e-6:
            rets /= std
        return rets

    def _update(self):
        states = torch.stack(self.ep_states)
        returns = self._returns().detach()
        values = self.critic(states)
        adv = returns - values.detach()

        # actor
        logp = torch.stack(self.ep_logp)
        loss_actor = -(logp * adv).mean() - self.entropy_coef * (-logp).mean()
        self.actor_opt.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 2.0)
        self.actor_opt.step()

        # critic
        loss_value = self.value_coef * F.mse_loss(values, returns)
        self.critic_opt.zero_grad()
        loss_value.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 2.0)
        self.critic_opt.step()

        self.last_ploss = loss_actor.detach().cpu()
        self.last_vloss = loss_value.detach().cpu()