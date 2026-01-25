from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_registry
import torch as th

from .basic_controller import BasicMAC


class SNP_MAC(BasicMAC):
    """
    Multi-Agent Controller for SNP (Structured Network Pruning) algorithm.

    Extends BasicMAC to support:
    - Agent-specific fixed random masks for structured pruning
    - Heterogeneous agent architectures through parameter sharing with masks
    """
    def forward(self, ep_batch, t, test_mode=False):
        """
        Forward pass through agents with their structured pruning masks.

        Args:
            ep_batch: Episode batch
            t: Timestep
            test_mode: Whether in test mode

        Returns:
            agent_outs: Agent outputs (Q-values or actions)
        """
        agent_inputs, agent_ids = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        # Forward with agent IDs for mask selection
        agent_outs, self.hidden_states = self.agent(
            agent_inputs, self.hidden_states, agent_ids
        )

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def _build_inputs(self, batch, t):
        """
        Build agent inputs and agent IDs.

        Args:
            batch: Episode batch
            t: Timestep

        Returns:
            inputs: Agent inputs
            agent_ids: Agent IDs for mask selection
        """
        # Use parent class method to build inputs
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])

        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=2)

        # Generate agent IDs: [0, 1, 2, ..., n_agents-1] for each batch
        agent_ids = th.arange(self.n_agents, device=batch.device).expand(bs, -1).long()

        return inputs, agent_ids
