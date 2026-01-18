"""
K-2:4 Q-Learner with Semi-structured Sparsity for SMACv2

Implements the complete K-2:4 algorithm for SMACv2 environments.
Adapted from MPE implementation with SMACv2-specific adjustments.
"""

import copy

import torch as th
from torch.optim import Adam
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer

from .nq_learner import NQLearner

# Import K24 diversity manager
import sys
import os
paraten_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(paraten_dir, 'modules', 'layer'))
from k24_diversity import K24DiversityManager


class K24_NQLearner(NQLearner):
    """
    K-2:4 Q-Learner for SMACv2 with Semi-structured Sparsity.

    Implements training with:
    - TD loss for Q-learning (inherited)
    - Pattern orthogonality loss for unit type heterogeneity
    - Temperature annealing
    - Adaptive mask resetting
    - Rewind & Finetune mechanism

    Args:
        mac: Multi-agent controller with K24_type_NRNNAgent_1R3
        scheme: Data scheme
        logger: Logger
        args: Configuration with K24_args
    """

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)

        self.K24_args = getattr(args, 'K24_args', {})
        self.n_unit_types = args.n_unit_types
        self.t_max = args.t_max

        # Diversity loss manager
        base_div_coef = self.K24_args.get("div_coef", 0.1)
        deque_len = self.K24_args.get("deque_len", 100)
        self.diversity_manager = K24DiversityManager(
            n_unit_types=self.n_unit_types,
            base_div_coef=base_div_coef,
            deque_len=deque_len
        )

        # Training dynamics
        self.reset_interval = self.K24_args.get("reset_interval", 10000)
        self.reset_ratio = self.K24_args.get("reset_ratio", 0.1)
        self.last_reset_t = 0
        self.anneal_end_step = self.K24_args.get("anneal_end_step", int(0.8 * args.t_max))

        # ========== Rewind & Finetune Mechanism ==========
        self.finetune_start_ratio = self.K24_args.get("finetune_start_ratio", 0.8)
        self.finetune_lr_decay = self.K24_args.get("finetune_lr_decay", 0.1)
        self.finetune_started = False
        self.base_lr = args.lr

        # Calculate finetune start step
        self.finetune_start_step = int(self.finetune_start_ratio * args.t_max)

        # Device
        self.device = "cuda" if args.use_cuda else "cpu"
        self.target_mac = copy.deepcopy(mac)

    def _update_targets(self):
        """Update target networks (hard update)."""
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """Train the K-2:4 agent for SMACv2."""
        self.mac.agent.set_require_grads(mode=True)

        # ========== Rewind & Finetune Check ==========
        # Check if we should start finetuning (freeze masks, decay LR)
        if not self.finetune_started and t_env >= self.finetune_start_step:
            self._start_finetune(t_env)

        # Periodic reset - only if not in finetune mode
        if (
            not self.finetune_started and
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
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Collect pattern probabilities
        pattern_probs_list = self.mac.agent.get_pattern_probs()

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(
            3
        )  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])
                targets = build_q_lambda_targets(
                    rewards,
                    terminated,
                    mask,
                    target_max_qvals,
                    qvals,
                    self.args.gamma,
                    self.args.td_lambda,
                )
            else:
                targets = build_td_lambda_targets(
                    rewards,
                    terminated,
                    mask,
                    target_max_qvals,
                    self.args.n_agents,
                    self.args.gamma,
                    self.args.td_lambda,
                )

        # Mixer
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        # Td-error
        td_error = chosen_action_qvals - targets.detach()
        td_error = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        td_loss = masked_td_error.sum() / mask.sum()

        # Pattern orthogonality diversity loss - skip during finetune (masks are frozen)
        if self.finetune_started:
            # During finetune, diversity loss is not computed (masks are fixed)
            div_loss = th.tensor(0.0, device=self.device)
            div_coef = 0.0
            div_stats = {'mean_similarity': 0.0, 'max_similarity': 0.0, 'pattern_entropy': 0.0}
        else:
            div_loss, div_coef, div_stats = self.diversity_manager.compute_loss(
                pattern_probs_list, td_loss
            )

        loss = td_loss + div_coef * div_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
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
            self.logger.log_stat("finetune_mode", 1.0 if self.finetune_started else 0.0, t_env)

            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item()
                / (mask_elems * self.args.n_agents),
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

            # Pattern usage statistics
            pattern_stats = self.mac.agent.get_pattern_stats()
            if 'pattern_mean' in pattern_stats:
                for i, p in enumerate(pattern_stats['pattern_mean']):
                    self.logger.log_stat(f"pattern_{i}_prob", p, t_env)

            self.log_stats_t = t_env

    def _start_finetune(self, t_env):
        """
        Start the finetuning phase (Rewind & Finetune).
        Fixed: Syncs target network and resets optimizer to prevent explosion.
        """
        print(f"\n{'='*60}")
        print(f"[Rewind & Finetune] Starting at t_env={t_env}")

        # 1. Freeze masks in Online Network (MAC)
        # -------------------------------------------------
        for layer in self.mac.agent.mask_layers:
            layer.freeze_mask()
        print(f"  - Online Network masks frozen")

        # 2. Synchronize Target Network (CRITICAL FIX)
        # -------------------------------------------------
        # We must ensure Target Network also enters frozen state with identical masks
        # First, perform a Hard Update to sync weights and buffers (frozen_mask)
        self._update_targets()

        # Second, manually set Target Network's mask_frozen flag
        # because load_state_dict usually doesn't include python boolean properties
        for target_layer, online_layer in zip(self.target_mac.agent.mask_layers, self.mac.agent.mask_layers):
            target_layer.mask_frozen = True
            # Ensure mask content is consistent (double insurance)
            if online_layer.frozen_mask is not None:
                target_layer.frozen_mask = online_layer.frozen_mask.clone()

        print(f"  - Target Network synced and frozen (Identical Masks)")

        # 3. Reduce Learning Rate & Reset Optimizer (CRITICAL FIX)
        # -------------------------------------------------
        # Merely modifying param_group['lr'] is insufficient, must clear Adam's state
        # Otherwise old momentum causes weights to fly wildly on new loss landscape

        new_lr = self.base_lr * self.finetune_lr_decay

        # Rebuild optimizer (most thorough method to clear state)
        self.optimiser = Adam(params=self.params, lr=new_lr)

        print(f"  - Optimizer reset. Learning rate: {self.base_lr} -> {new_lr}")

        # 4. Disable Heterogeneity Gradients
        # -------------------------------------------------
        self.mac.agent.set_require_grads(mode=False)
        print(f"  - Heterogeneity coefficients (alpha) frozen")
        print(f"  - Only shared weights (W) will be optimized")

        self.finetune_started = True
        print(f"{'='*60}\n")
        print(f"[Rewind & Finetune] Starting at t_env={t_env}")
        print(f"  - Masks frozen (Gumbel-Softmax disabled)")
        print(f"  - Learning rate: {self.base_lr} -> {self.base_lr * self.finetune_lr_decay}")
        print(f"  - Heterogeneity coefficients (alpha) frozen")
        print(f"  - Only shared weights (W) will be optimized")
        print(f"{'='*60}\n")
