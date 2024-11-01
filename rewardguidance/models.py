"""
This code will contains the networks architecture for the reward guidance model.

Basicly the idea is that the model will take a time series compose of two things :
- the init states of the system
- the future states of the system (where the diffusion process will be applied)

The model will then output a reward for the whole trajectory, indicating how good the trajectory is.
"""


from typing import Tuple
import einops
import torch
import torch.nn as nn

from rewardguidance.mamba2 import Mamba2, Mamba2Config, RMSNorm


class RewardGuidanceModel(nn.Module):
    """
    Reward guidance model with a time flags and a shortcut value.

    Shortcut value is used to indicate if the trajectory is a shortcut or not.
    Time flags is used to indicate the time of the trajectory.

    For shortcut value see paper : https://arxiv.org/pdf/2410.12557

    Time is the classical time value in diffusion models.

    """

    def __init__(
        self,
        nb_future_states: int = 16,
        nb_init_states: int = 1,
        nb_hidden_dim: int = 128,
        nb_input_dim: int = 9*6*6,
        nb_output_dim: int = 9*6*6,
        chunk_size: int = 8,
        device = None,
    ):
        super().__init__()

        self.nb_future_states = nb_future_states
        self.nb_init_states = nb_init_states
        self.nb_hidden_dim = nb_hidden_dim
        self.nb_input_dim = nb_input_dim
        self.nb_output_dim = nb_output_dim

        config = Mamba2Config(
                d_model=nb_hidden_dim,
                n_layer=8,
                d_state=32,
                d_conv=4,
                expand=2,
                headdim=8,
                chunk_size=chunk_size,
        )

        self.mamba2_layers = nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer=Mamba2(config, device=device),
                                norm=RMSNorm(config.d_model, device=device),
                            )
                        )
                        for _ in range(config.n_layer)
                    ]
                )


        self.input_layer = nn.Linear(nb_input_dim, nb_hidden_dim // 2)
        self.output_layer = nn.Linear(nb_hidden_dim, nb_output_dim)

        # embedding for the states
        self.states_embedding = nn.Parameter(
            torch.randn(nb_future_states + nb_init_states, nb_hidden_dim // 2)
        )

        # embedding for the time flags
        self.time_flags_embedding = nn.Linear(1, nb_hidden_dim * 4)

        # embedding for the shortcut value
        self.shortcut_value_embedding = nn.Linear(1, nb_hidden_dim * 4)

        # linear layer for reward
        self.reward_layer = nn.Linear(nb_hidden_dim, 1)

    def forward(
        self,
        init_states: torch.Tensor,
        future_states: torch.Tensor,
        time_flags: torch.Tensor,
        shortcut_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            init_states: (batch_size, seq_len, 6*9, 6)
            future_states: (batch_size, seq_len, 6*9, 6)
            time_flags: (batch_size, 1) value between 0 and 1 (for the time of the trajectory)
            shortcut_value: (batch_size, 1) value between 0 and 1 (for the shortcut value)
        Returns:
            (batch_size, seq_len, nb_output_dim)
        """
        # we retrieve the dimensions of the inputs (batch_size, seq_len, dim)
        batch_size, seq_init_states, _, _ = init_states.shape
        batch_size, seq_future_states, _, _ = future_states.shape

        assert (
            seq_init_states == self.nb_init_states
        ), f"seq_init_states {seq_init_states} != nb_init_states {self.nb_init_states}"
        assert (
            seq_future_states == self.nb_future_states
        ), f"seq_future_states {seq_future_states} != nb_future_states {self.nb_future_states}"

        init_states = einops.rearrange(init_states, "b s h w -> b s (h w)")
        future_states = einops.rearrange(future_states, "b s h w -> b s (h w)")
        
        # we create the embeddings for the future states and the init states
        states_embeddings = self.states_embedding.repeat(batch_size, 1, 1)

        # we create the embeddings for the time flags
        time_flags_embeddings = self.time_flags_embedding(time_flags).unsqueeze(1)
        shortcut_value_embeddings = self.shortcut_value_embedding(
            shortcut_value
        ).unsqueeze(1)

        # we concatenate the init states and the future states embeddings
        x = torch.cat([init_states, future_states], dim=1)

        # we pass the input through the input layer
        x = self.input_layer(x)

        # we concatenate the states embeddings and the input embeddings
        x = torch.cat([states_embeddings, x], dim=2)

        # modification of the input to add the time flags embeddings [:, :, :self.nb_hidden_dim]
        x = (
            x
            * time_flags_embeddings[:, :, : self.nb_hidden_dim]
            * shortcut_value_embeddings[:, :, : self.nb_hidden_dim]
            + time_flags_embeddings[:, :, self.nb_hidden_dim : (self.nb_hidden_dim * 2)]
            + shortcut_value_embeddings[
                :, :, self.nb_hidden_dim : (self.nb_hidden_dim * 2)
            ]
        )

        # we pass the input through the mamba2 model
        for i, layer in enumerate(self.mamba2_layers):
            y, _ = layer.mixer(layer.norm(x), None)
            x = y + x

        # we add another embedding for the time flags
        x = (
            x
            * time_flags_embeddings[
                :, :, (self.nb_hidden_dim * 2) : (self.nb_hidden_dim * 3)
            ]
            * shortcut_value_embeddings[
                :, :, (self.nb_hidden_dim * 2) : (self.nb_hidden_dim * 3)
            ]
            + time_flags_embeddings[:, :, (self.nb_hidden_dim * 3) :]
            + shortcut_value_embeddings[:, :, (self.nb_hidden_dim * 3) :]
        )

        # we pass the output through the output layer
        state_final = self.output_layer(x)

        # # we pass the output through the reward layer
        # reward = self.reward_layer(state_final)

        # # sum for the reward over the time
        # reward = reward.max(dim=1)

        # we take only the seq_future_states last states
        state_final = state_final[:, -self.nb_future_states:, :]

        # we want to modify the state_final to be of shape (batch_size, seq_len, 6*9, 6)
        state_final = state_final.view(batch_size, seq_future_states, -1, 6)

        return state_final #, reward
