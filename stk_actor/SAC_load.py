

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# -------------------------------
# Flatten observation
# -------------------------------
def flatten_obs(obs, obs_space):
    """
    Flatten Dict observation:
    - continuous: raw
    - discrete (MultiDiscrete): one-hot using obs_space
    """
    cont = np.asarray(obs["continuous"], dtype=np.float32).reshape(1, -1)
    disc = np.asarray(obs["discrete"], dtype=np.int64).squeeze()
    nvec = obs_space.spaces["discrete"].nvec

    onehot = []
    for i, n in enumerate(nvec):
        v = np.zeros(n, dtype=np.float32)
        idx = int(np.clip(disc[i], 0, n - 1))  # safe clipping
        v[idx] = 1.0
        onehot.append(v)

    disc_flat = np.concatenate(onehot, axis=0).reshape(1, -1)
    return np.concatenate([cont, disc_flat], axis=1).squeeze(0)

# -------------------------------
# Replay Buffer
# -------------------------------
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size = size
        self.ptr, self.size = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.act_buf[idxs],
            rews=self.rew_buf[idxs],
            done=self.done_buf[idxs]
        )
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in batch.items()}

# -------------------------------
# Actor
# -------------------------------
LOG_STD_MIN = -20
LOG_STD_MAX = 2

class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


# -------------------------------
# Critic
# -------------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


# -------------------------------
# SAC Agent
# -------------------------------
class SACAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_size=int(1e6),
        batch_size=256,
        update_every=4,
        updates_per_step=1,
        policy_delay=2, 
        writer=None,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.updates_per_step = updates_per_step
        self.policy_delay = policy_delay
        self.writer = writer

        # -------- entropy --------
        self.target_entropy = -act_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=3e-4)

        # -------- actor --------
        self.actor = GaussianActor(obs_dim, act_dim).to(self.device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)

        # -------- critics --------
        self.q1 = QNetwork(obs_dim, act_dim).to(self.device)
        self.q2 = QNetwork(obs_dim, act_dim).to(self.device)
        self.q1_target = QNetwork(obs_dim, act_dim).to(self.device)
        self.q2_target = QNetwork(obs_dim, act_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.q_opt = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

        self.total_steps = 0
        self.update_steps = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, deterministic=False):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(obs)
                action = torch.tanh(mu)
            else:
                action, _ = self.actor.sample(obs)
        return action.cpu().numpy()[0]

    def step(self):
        self.total_steps += 1

        if (
            self.replay_buffer.size >= self.batch_size
            and self.total_steps % self.update_every == 0
        ):
            for _ in range(self.updates_per_step):
                self.update()

    def update(self):
        batch = self.replay_buffer.sample_batch(self.batch_size)
        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        done = batch["done"].to(self.device)

        # -------- critic update --------
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            target_q = torch.min(
                self.q1_target(next_obs, next_action),
                self.q2_target(next_obs, next_action),
            ) - self.alpha * next_log_prob

            q_target = rews.unsqueeze(1) + self.gamma * (1 - done.unsqueeze(1)) * target_q

        q1_loss = F.mse_loss(self.q1(obs, acts), q_target)
        q2_loss = F.mse_loss(self.q2(obs, acts), q_target)

        self.q_opt.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q_opt.step()

        # -------- delayed actor + alpha --------
        if self.update_steps % self.policy_delay == 0:
            action_new, log_prob = self.actor.sample(obs)
            q_min = torch.min(
                self.q1(obs, action_new),
                self.q2(obs, action_new),
            )

            actor_loss = (self.alpha * log_prob - q_min).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        # -------- target update --------
        for tp, p in zip(self.q1_target.parameters(), self.q1.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for tp, p in zip(self.q2_target.parameters(), self.q2.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        self.update_steps += 1

        if self.writer and self.update_steps % 100 == 0:
            self.writer.add_scalar("loss/q1", q1_loss.item(), self.update_steps)
            self.writer.add_scalar("loss/q2", q2_loss.item(), self.update_steps)
            self.writer.add_scalar("alpha", self.alpha.item(), self.update_steps)
