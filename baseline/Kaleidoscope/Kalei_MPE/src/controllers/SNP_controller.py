from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_registry
from .basic_controller import BasicMAC
import torch as th


class SNP_MAC(BasicMAC):
    """
    SNP Multi-Agent Controller for MPE.

    This controller extends the BasicMAC to work with SNP agents.
    It handles agent-specific processing similar to Kalei_MAC,
    allowing different agents to use different masks in the SNP agent.

    Key features:
    1. Supports agent ID-based agent identification (0 to n_agents-1)
    2. Passes agent IDs to SNP agent for mask selection
    3. Compatible with MPE environments

    Differences from SMACv2 SNP_type_NMAC:
    - Uses agent IDs directly (0 to n_agents-1) instead of unit type extraction
    - Simpler _build_inputs method since MPE doesn't use unit types
    """

    def forward(self, ep_batch, t, test_mode=False):
        """
        Forward pass through the SNP agent network.

        Args:
            ep_batch: Episode batch data
            t: Current timestep
            test_mode: Whether in test mode

        Returns:
            agent_outs: Q-values for each agent [batch_size, n_agents, n_actions]
        """
        agent_inputs, agent_ids = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        if test_mode:
            self.agent.eval()
        else:
            self.agent.train()

        # Pass agent_ids to agent for SNP mask selection
        agent_outs, self.hidden_states = self.agent(
            agent_inputs, self.hidden_states, agent_ids
        )

        self.agent.train()

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def _build_inputs(self, batch, t):
        """
        Build input features for the agent network.

        This method extracts observations and generates agent IDs.
        Agent IDs are used to select appropriate masks in SNP agents.

        Args:
            batch: Episode batch
            t: Timestep

        Returns:
            inputs: Concatenated input features [batch_size, n_agents, input_dim]
            agent_ids: Agent IDs [batch_size, n_agents]
        """
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av

        # Optionally add last action
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])

        # Optionally add agent IDs
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )

        # Concatenate all inputs
        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)

        # Generate agent IDs (0 to n_agents-1)
        agent_ids = th.arange(self.n_agents, device=batch.device).expand(bs, -1).long()

        return inputs, agent_ids
