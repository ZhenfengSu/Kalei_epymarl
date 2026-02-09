"""
LDSA Learner with K24 2:4 Sparsity Support.

This learner extends the base LDSA learner with:
1. Temperature annealing for K24 Gumbel-Softmax
2. Pattern orthogonality diversity loss
3. Adaptive diversity coefficient adjustment
4. Mask freezing and finetuning mechanism

References:
- K24: /mnt/lc_gpu_test/Semi_Kaleidoscope/Kalei_epymarl/Semi_Kalei/Kaleidoscope/Kalei_SMACv2
"""

import copy
import torch as th
from torch.optim import RMSprop
import torch.nn.functional as F

from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from modules.layer.k24_diversity import PatternOrthogonalityLoss, K24DiversityManager


class LDSAK24Learner:
    """
    LDSA Learner with K24 2:4 Sparsity and Attention Support.

    Features:
    - Standard TD loss
    - Subtask representation diversity loss
    - Subtask probability KL loss
    - K24 pattern orthogonality diversity loss (optional)
    - Temperature annealing for K24 layers
    - Mask freezing and finetuning mechanism

    Args:
        mac: Multi-agent controller
        scheme: Data scheme
        logger: Logger instance
        args: Configuration arguments
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        # Check if using K24
        self.use_k24 = getattr(args, 'use_k24', False)
        self.use_attention = getattr(args, 'use_attention', True)

        # Collect parameters
        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        # Mixer setup
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # Target network
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        # K24-specific setup
        if self.use_k24:
            # Temperature annealing
            self.temperature_init = getattr(args, 'K24_temperature_init', 5.0)
            self.temperature_min = getattr(args, 'K24_temperature_min', 0.1)
            self.anneal_end_step = getattr(args, 'K24_anneal_end_step', 100000)
            self.current_step = 0

            # Diversity loss
            self.div_coef = getattr(args, 'K24_div_coef', 0.1)
            self.pattern_loss_fn = PatternOrthogonalityLoss(args.n_agents)
            self.div_manager = K24DiversityManager(
                base_coef=self.div_coef,
                target_ratio=getattr(args, 'K24_div_target_ratio', 1.0),
                min_coef=getattr(args, 'K24_div_min_coef', 0.01),
                max_coef=getattr(args, 'K24_div_max_coef', 1.0)
            )

            # Mask reset
            self.reset_interval = getattr(args, 'K24_reset_interval', 10000)
            self.reset_ratio = getattr(args, 'K24_reset_ratio', 0.1)

            # Finetune
            self.finetune_start_ratio = getattr(args, 'K24_finetune_start_ratio', 0.8)
            self.finetune_lr_decay = getattr(args, 'K24_finetune_lr_decay', 0.9)
            self.finetune_started = False

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """
        Training step with K24 support.

        Args:
            batch: Episode batch
            t_env: Current environment step
            episode_num: Current episode number
        """
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]  # [bs, eplen, 1]
        actions = batch["actions"][:, :-1] # [bs, eplen, n_agents, 1]
        terminated = batch["terminated"][:, :-1].float() # [bs, eplen, 1]
        mask = batch["filled"][:, :-1].float() # [bs, eplen, 1]
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"] # [bs, eplen+1, n_agents, n_actions]

        # Calculate estimated Q-Values
        mac_out = []
        subtask_prob_logits = []
        subtask_prob_logits_last = []
        subtask_embeds = []

        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            # Controller will extract state from batch internally
            agent_outs, subtask_prob_logit, subtask_embed = self.mac.forward(
                batch, t=t
            )
            mac_out.append(agent_outs)
            if t > 0:
                subtask_prob_logits.append(subtask_prob_logit)
            if t < batch.max_seq_length - 1:
                subtask_prob_logits_last.append(subtask_prob_logit)
                subtask_embeds.append(subtask_embed)

        mac_out = th.stack(mac_out, dim=1)  # [bs, eplen+1, n_agents, n_actions]
        subtask_prob_logits = th.stack(subtask_prob_logits, dim=1) # [bs, eplen, n_agents, n_subtasks]
        subtask_prob_logits_last = th.stack(subtask_prob_logits_last, dim=1) # [bs, eplen, n_agents, n_subtasks]
        subtask_embeds = th.stack(subtask_embeds, dim=1) # [bs, eplen, n_subtasks, embed_dim]

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # [bs, eplen, n_agents]

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            # Controller will extract state from batch internally
            target_agent_outs, _, _ = self.target_mac.forward(
                batch, t=t
            )
            target_mac_out.append(target_agent_outs)

        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # [bs, eplen, n_agents, n_actions]

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        # MSE loss of representation between two different subtasks
        subtask_embeds1 = subtask_embeds.unsqueeze(3) # [bs, eplen, n_subtasks, 1, embed_dim]
        subtask_embeds2 = subtask_embeds.unsqueeze(2).clone().detach() # [bs, eplen, 1, n_subtasks, embed_dim]
        subtask_dis = ((subtask_embeds1 - subtask_embeds2) ** 2).sum(dim=4, keepdim=True) # [bs, eplen, n_subtasks, n_subtasks, 1]
        subtask_dis = subtask_dis.sum([4, 3, 2]).unsqueeze(-1) # [bs, eplen, 1]
        subtask_dis = subtask_dis / (self.args.n_subtasks * (self.args.n_subtasks - 1)) # [bs, eplen, 1]
        masked_subtask_dis = subtask_dis * mask
        subtask_dis_loss = masked_subtask_dis.sum() / mask.sum()

        # KL loss of subtask prob between two adjacent frames
        subtask_probs = F.softmax(subtask_prob_logits, dim=-1) # [bs, eplen, n_agents, n_subtasks]
        subtask_probs_last = F.softmax(subtask_prob_logits_last, dim=-1) # [bs, eplen, n_agents, n_subtasks]
        subtask_prob_kl = th.sum(subtask_probs_last.detach() * ( - th.log(subtask_probs + 1e-8)), dim=[3, 2]).unsqueeze(-1) / self.args.n_agents #[bs, eplen, 1]
        mask_ = mask[:, 1:] # [bs, eplen-1, 1]
        mask_ = th.cat([mask_, th.zeros(mask_.shape[0], 1, 1, device=mask_.device)], dim=1) # [bs, eplen, 1]
        subtask_prob_kl_loss = subtask_prob_kl.sum() / mask_.sum()

        # Base loss
        loss = td_loss - self.args.lambda_subtask_repr * subtask_dis_loss + self.args.lambda_subtask_prob * subtask_prob_kl_loss

        # K24 diversity loss
        div_loss = th.tensor(0.0)
        if self.use_k24:
            pattern_probs_dict = self.mac.get_pattern_probs()
            if pattern_probs_dict:
                # Extract pattern probabilities from the first K24 layer
                # pattern_probs_dict format: {'layer_0': [batch, out_features, n_groups, 6], ...}
                # We need to reshape to [n_agents, batch//n_agents, n_groups, 6] for diversity loss
                first_layer_probs = None
                for layer_name, probs in pattern_probs_dict.items():
                    if probs is not None:
                        first_layer_probs = probs
                        break

                if first_layer_probs is not None:
                    # probs shape: [batch_total, out_features, n_groups, 6]
                    # We want to organize by agents for diversity computation
                    # Assuming batch_total = batch_size * n_agents
                    bs_total, out_features, n_groups, n_patterns = first_layer_probs.shape
                    n_agents = self.args.n_agents

                    if bs_total >= n_agents and bs_total % n_agents == 0:
                        # Reshape to [n_agents, batch_total//n_agents, out_features*n_groups, 6]
                        # Average over batches and features to get per-agent pattern distribution
                        probs_reshaped = first_layer_probs.reshape(n_agents, bs_total // n_agents, -1, n_patterns)
                        # Average over batch and feature dimensions: [n_agents, 6]
                        agent_pattern_probs = probs_reshaped.mean(dim=1).mean(dim=1)

                        # Add dummy dimensions to match expected format: [n_agents, 1, 1, 6]
                        agent_pattern_probs = agent_pattern_probs.unsqueeze(1).unsqueeze(2)

                        # Calculate diversity loss directly (expects [batch, n_agents, n_groups, 6])
                        div_loss = self.pattern_loss_fn(agent_pattern_probs)
                        loss += self.div_manager.get_coef() * div_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # Update target network
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # K24-specific updates
        if self.use_k24:
            self._k24_step_update(t_env, episode_num, td_loss.item(), div_loss.item() if isinstance(div_loss, th.Tensor) else 0.0)

        # Logging
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            self.logger.log_stat("subtask_dis_loss", subtask_dis_loss.item(), t_env)
            self.logger.log_stat("subtask_prob_kl_loss", subtask_prob_kl_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)

            if self.use_k24:
                self.logger.log_stat("div_loss", div_loss.item() if isinstance(div_loss, th.Tensor) else 0.0, t_env)
                self.logger.log_stat("div_coef", self.div_manager.get_coef(), t_env)

                # Log temperature
                k24_layers = self.mac.get_k24_layers()
                if k24_layers:
                    self.logger.log_stat("temperature", k24_layers[0].temperature.item(), t_env)

                # Log sparsity
                for i, layer in enumerate(k24_layers):
                    sparsity = layer.get_sparsity()
                    self.logger.log_stat(f"sparsity_layer_{i}", sparsity, t_env)

            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _k24_step_update(self, t_env, episode_num, td_loss, div_loss):
        """
        K24-specific updates per training step.

        Args:
            t_env: Current environment step
            episode_num: Current episode number
            td_loss: Current TD loss value
            div_loss: Current diversity loss value
        """
        self.current_step = t_env

        # Temperature annealing
        progress = min(t_env / self.anneal_end_step, 1.0)
        temperature = self.temperature_init - (self.temperature_init - self.temperature_min) * progress
        self.mac.set_temperature(temperature)

        # Adaptive diversity coefficient
        if div_loss > 0:
            self.div_manager.update_coef(td_loss, div_loss)

        # Periodic mask reset
        if t_env > 0 and t_env % self.reset_interval == 0:
            self._reset_hetero_alpha()

        # Finetune mechanism
        if not self.finetune_started:
            t_max = getattr(self.args, 't_max', 500000)
            if t_env >= t_max * self.finetune_start_ratio:
                self._start_finetune()

    def _reset_hetero_alpha(self):
        """Reset heterogeneity coefficients for all K24 layers."""
        for layer in self.mac.get_k24_layers():
            layer.reset_hetero_alpha(reset_mask=self.reset_ratio)

    def _start_finetune(self):
        """
        Start finetuning phase:
        1. Freeze masks
        2. Decay learning rate
        3. Reset optimizer state
        """
        self.logger.console_logger.info(f"Starting K24 finetuning at step {self.current_step}")

        # Freeze masks
        self.mac.freeze_masks()
        self.finetune_started = True

        # Decay learning rate
        for param_group in self.optimiser.param_groups:
            param_group['lr'] *= self.finetune_lr_decay

        # Reset optimizer state for stability
        self.optimiser.state = {}

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

        # Save K24-specific state
        if self.use_k24:
            k24_state = {
                'finetune_started': self.finetune_started,
                'current_step': self.current_step,
                'div_coef': self.div_manager.get_coef(),
            }
            th.save(k24_state, "{}/k24_state.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

        # Load K24-specific state
        if self.use_k24:
            try:
                k24_state = th.load("{}/k24_state.th".format(path), map_location=lambda storage, loc: storage)
                self.finetune_started = k24_state.get('finetune_started', False)
                self.current_step = k24_state.get('current_step', 0)
                self.div_manager.current_coef = k24_state.get('div_coef', self.div_coef)
            except FileNotFoundError:
                self.logger.console_logger.warning("K24 state file not found, using defaults")
