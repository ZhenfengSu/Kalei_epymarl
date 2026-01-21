from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_registry
from .n_controller import NMAC
import torch as th


class SNP_type_NMAC(NMAC):
    """
    SNP-type Multi-Agent Controller.

    This controller extends the basic NMAC to work with SNP agents.
    It handles unit type-specific processing similar to Kalei_type_NMAC,
    allowing different unit types to use different masks in the SNP agent.

    Key features:
    1. Supports unit type-based agent identification
    2. Passes unit IDs to SNP agent for mask selection
    3. Compatible with SMAC environments that use unit types
    """

    def __init__(self, scheme, groups, args):
        super(SNP_type_NMAC, self).__init__(scheme, groups, args)
        self.n_unit_types = args.n_unit_types
        assert (
            args.env_args["state_timestep_number"] is False
        ), "the unit type slicing is wrong if otherwise"

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """Select actions for agents using epsilon-greedy or other action selectors."""
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            qvals[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

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
        agent_inputs, unit_ids = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        if test_mode:
            self.agent.eval()
        else:
            self.agent.train()

        # Pass unit_ids to agent for SNP mask selection
        agent_outs, self.hidden_states = self.agent(
            agent_inputs, self.hidden_states, unit_ids
        )

        self.agent.train()
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def _build_inputs(self, batch, t):
        """
        Build input features for the agent network.

        This method extracts observations and unit type IDs from the batch.
        Unit IDs are used to select appropriate masks in SNP agents.

        Args:
            batch: Episode batch
            t: Timestep

        Returns:
            inputs: Concatenated input features [batch_size, n_agents, input_dim]
            unit_ids: Unit type IDs [batch_size, n_agents]
        """
        bs = batch.batch_size
        inputs = []

        # Add observations
        inputs.append(batch["obs"][:, t])  # b1av

        # Extract unit type IDs from one-hot encoding (last n_unit_types dimensions)
        unit_ids_onehot = batch["obs"][:, t, :, -self.n_unit_types:]

        # The agent is either dead (all zeros) or alive (one-hot)
        assert (
            th.logical_or(
                unit_ids_onehot.sum(dim=-1) == 1, unit_ids_onehot.sum(dim=-1) == 0
            )
        ).all(), "recheck codes for selecting unit types"

        # Convert one-hot to unit IDs
        unit_ids = th.argmax(unit_ids_onehot, dim=-1)

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
        unit_ids = unit_ids.reshape(bs, self.n_agents)

        return inputs, unit_ids
