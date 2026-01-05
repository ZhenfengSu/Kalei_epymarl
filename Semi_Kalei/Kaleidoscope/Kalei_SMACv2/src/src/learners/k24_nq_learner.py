"""
K-2:4 NQ-Learner with Semi-structured Sparsity for SMACv2

Implements the complete K-2:4 algorithm for StarCraft II multi-agent scenarios.
Adapted from the epymarl implementation with SMACv2-specific adjustments.
"""

import torch as th
from torch.optim import RMSprop, Adam
from components.episode_buffer import EpisodeBatch
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets

from .nq_learner import NQLearner
import sys
# 获取当前文件的目录
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录
parent_dir = os.path.dirname(current_dir)
# 获得modules.layer目录路径
layer_dir = os.path.join(parent_dir, 'modules', 'layer')
# 将modules.layer目录添加到sys.path
sys.path.append(layer_dir)
from k24_diversity import K24DiversityManager
# from ..modules.layer.k24_diversity import K24DiversityManager


class K24_NQLearner(NQLearner):
    """
    K-2:4 NQ-Learner for SMACv2 with Semi-structured Sparsity.

    Implements training with:
    - TD loss for Q-learning (inherited)
    - Pattern orthogonality loss for unit type heterogeneity
    - Temperature annealing
    - Adaptive mask resetting

    Args:
        mac: Multi-agent controller with K24_type_NRNNAgent_1R3
        scheme: Data scheme
        logger: Logger
        args: Configuration with K24_args
    """

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)

        self.K24_args = getattr(args, 'K24_args', {})
        self.n_agents = args.n_agents
        self.t_max = args.t_max

        # Diversity loss manager
        base_div_coef = self.K24_args.get("div_coef", 0.1)
        deque_len = self.K24_args.get("deque_len", 100)
        self.diversity_manager = K24DiversityManager(
            n_agents=self.n_agents,
            base_div_coef=base_div_coef,
            deque_len=deque_len
        )

        # Training dynamics
        self.reset_interval = self.K24_args.get("reset_interval", 10000)
        self.reset_ratio = self.K24_args.get("reset_ratio", 0.1)
        self.last_reset_t = 0
        self.anneal_end_step = self.K24_args.get("anneal_end_step", int(0.8 * args.t_max))

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """Train the K-2:4 agent for SMACv2."""
        self.mac.agent.set_require_grads(mode=True)

        # Periodic reset
        if (
            t_env - self.last_reset_t > self.reset_interval
            and self.t_max - t_env > self.reset_interval
        ):
            self.mac.agent._reset_all_masks_weights(self.reset_ratio)
            self.last_reset_t = t_env

        # Calculate training progress
        progress = min(t_env / self.anneal_end_step, 1.0)

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1].to(self.device)
        actions = batch["actions"][:, :-1].to(self.device)
        terminated = batch["terminated"][:, :-1].float().to(self.device)
        mask = batch["filled"][:, :-1].float().to(self.device)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1]).to(self.device)
        avail_actions = batch["avail_actions"].to(self.device)

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)

        # Collect pattern probabilities
        pattern_probs_list = self.mac.agent.get_pattern_probs()

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            target_mac_out = th.stack(target_mac_out, dim=1)

            # Double Q-learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            if getattr(self.args, "q_lambda", False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])
                targets = build_q_lambda_targets(
                    rewards, terminated, mask, target_max_qvals, qvals,
                    self.args.gamma, self.args.td_lambda
                )
            else:
                targets = build_td_lambda_targets(
                    rewards, terminated, mask, target_max_qvals,
                    self.args.n_agents, self.args.gamma, self.args.td_lambda
                )

        # Mixer
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = chosen_action_qvals - targets.detach()
        td_error = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        td_loss = masked_td_error.sum() / mask.sum()

        # Pattern orthogonality diversity loss
        div_loss, div_coef, div_stats = self.diversity_manager.compute_loss(
            pattern_probs_list, td_loss
        )

        loss = td_loss + div_coef * div_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # Update target network
        if (
            episode_num - self.last_target_update_episode
        ) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # Temperature annealing
        self.mac.agent.anneal_temperature(progress)

        # Logging
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", td_loss.item(), t_env)
            self.logger.log_stat("div_loss", div_loss.item(), t_env)
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("div_coef", div_coef, t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("temperature", self.mac.agent.fc1.temperature.item(), t_env)
            self.logger.log_stat("progress", progress, t_env)

            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )

            # Diversity statistics
            if 'mean_similarity' in div_stats:
                self.logger.log_stat("pattern_mean_similarity", div_stats['mean_similarity'], t_env)
                self.logger.log_stat("pattern_max_similarity", div_stats['max_similarity'], t_env)
                self.logger.log_stat("pattern_entropy", div_stats.get('pattern_entropy', 0.0), t_env)

            # Sparsity statistics
            sparsities, sparsities_var, overall_sparsity = self.mac.agent.get_sparsities()
            for i, s in enumerate(sparsities):
                self.logger.log_stat(f"sparsity_layer_{i}", s.item(), t_env)
            self.logger.log_stat("overall_sparsity", overall_sparsity, t_env)

            self.log_stats_t = t_env
