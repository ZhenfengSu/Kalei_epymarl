"""
K-2:4 Q-Learner with Semi-structured Sparsity for MPE

Implements the complete K-2:4 algorithm for Multi-Agent Particle Environments.
Adapted from SMACv2 implementation with MPE-specific adjustments.
"""

import copy

import torch as th
from torch.optim import Adam
from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.layer.k24_diversity import K24DiversityManager

from .q_learner import QLearner


class K24_QLearner(QLearner):
    """
    K-2:4 Q-Learner for MPE with Semi-structured Sparsity.

    Implements training with:
    - TD loss for Q-learning (inherited)
    - Pattern orthogonality loss for agent heterogeneity
    - Temperature annealing
    - Adaptive mask resetting

    Args:
        mac: Multi-agent controller with K24_RNNAgent_1R3
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


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """Train the K-2:4 agent for MPE."""
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
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert (
                rewards.size(2) == 1
            ), "Expected singular agent dimension for common rewards"
            # reshape rewards to be of shape (batch_size, episode_length, n_agents)
            rewards = rewards.expand(-1, -1, self.n_agents)

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
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(
                chosen_action_qvals, batch["state"][:, :-1]
            )
            target_max_qvals = self.target_mixer(
                target_max_qvals, batch["state"][:, 1:]
            )

        if self.args.standardise_returns:
            target_max_qvals = (
                target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
            )

        # Calculate 1-step Q-Learning targets
        targets = (
            rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()
        )

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = chosen_action_qvals - targets.detach()

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error**2).sum() / mask.sum()

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

        self.training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

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
        # 我们必须确保 Target Network 也进入冻结状态，且使用完全相同的 Mask
        # 首先，进行一次 Hard Update，把权重和 buffer (frozen_mask) 同步过去
        self._update_targets_hard()
        
        # 其次，手动设置 Target Network 的 mask_frozen 标志
        # 因为 load_state_dict 通常不包含 python 的布尔属性
        for target_layer, online_layer in zip(self.target_mac.agent.mask_layers, self.mac.agent.mask_layers):
            target_layer.mask_frozen = True
            # 再次确保 mask 内容一致 (双重保险)
            if online_layer.frozen_mask is not None:
                target_layer.frozen_mask = online_layer.frozen_mask.clone()
        
        print(f"  - Target Network synced and frozen (Identical Masks)")

        # 3. Reduce Learning Rate & Reset Optimizer (CRITICAL FIX)
        # -------------------------------------------------
        # 仅仅修改 param_group['lr'] 是不够的，必须清除 Adam 的 state
        # 否则旧的动量(momentum)会导致权重在新的 Loss 地形上乱飞
        
        new_lr = self.base_lr * self.finetune_lr_decay
        
        # 重建优化器 (最彻底的清除状态方法)
        # 注意：这里假设 self.params 是在 __init__ 中定义的
        self.optimiser = Adam(params=self.params, lr=new_lr)
        
        # 或者，如果你不想重建对象，可以手动清除状态：
        # self.optimiser.state.clear()
        # for group in self.optimiser.param_groups:
        #     group['lr'] = new_lr
            
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
