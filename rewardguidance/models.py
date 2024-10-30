"""
This code will contains the networks architecture for the reward guidance model.

Basicly the idea is that the model will take a time series compose of two things :
- the init states of the system
- the future states of the system (where the diffusion process will be applied)

The model will then output a reward for the whole trajectory, indicating how good the trajectory is.
"""


from typing import Tuple
import torch
import torch.nn as nn

from rewardguidance.mamba2 import Mamba2, Mamba2Config


class RewardGuidanceModel(nn.Module):
    """
    Reward guidance model.
    """

    def __init__(
        self,
        nb_future_states: int = 16,
        nb_init_states: int = 1,
        nb_hidden_dim: int = 128,
        nb_input_dim: int = 392,
        nb_output_dim: int = 32,
    ):
        super().__init__()

        self.nb_future_states = nb_future_states
        self.nb_init_states = nb_init_states
        self.nb_hidden_dim = nb_hidden_dim
        self.nb_input_dim = nb_input_dim
        self.nb_output_dim = nb_output_dim

        self.mamba2 = Mamba2(
            Mamba2Config(
                d_model=nb_hidden_dim,
                n_layer=8,
                d_state=32,
                d_conv=4,
                expand=2,
                headdim=8,
            )
        )

        self.input_layer = nn.Linear(nb_input_dim, nb_hidden_dim // 2)
        self.output_layer = nn.Linear(nb_hidden_dim, nb_output_dim)

        # embedding for the states
        self.states_embedding = nn.Parameter(
            torch.randn(nb_future_states + nb_init_states, nb_hidden_dim // 2)
        )

        # embedding for the time flags
        self.time_flags_embedding = nn.Linear(1, nb_hidden_dim * 4)

    def forward(
        self,
        init_states: torch.Tensor,
        future_states: torch.Tensor,
        time_flags: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            init_states: (batch_size, seq_len, dim)
            future_states: (batch_size, seq_len, dim)
            time_flags: (batch_size, 1) value between 0 and 1
        Returns:
            (batch_size, seq_len, nb_output_dim)
        """
        # we retrieve the dimensions of the inputs (batch_size, seq_len, dim)
        batch_size, seq_init_states, _ = init_states.shape
        batch_size, seq_future_states, _ = future_states.shape

        assert seq_init_states == self.nb_init_states
        assert seq_future_states == self.nb_future_states

        # we create the embeddings for the future states and the init states
        states_embeddings = self.states_embedding.repeat(batch_size, 1, 1)

        # we create the embeddings for the time flags
        time_flags_embeddings = self.time_flags_embedding(time_flags).unsqueeze(1)

        # we concatenate the init states and the future states embeddings
        x = torch.cat([init_states, future_states], dim=1)

        # we pass the input through the input layer
        x = self.input_layer(x)

        # we concatenate the states embeddings and the input embeddings
        x = torch.cat([states_embeddings, x], dim=2)

        # modification of the input to add the time flags embeddings [:, :, :self.nb_hidden_dim]
        x = (
            x * time_flags_embeddings[:, :, : self.nb_hidden_dim]
            + time_flags_embeddings[:, :, self.nb_hidden_dim : (self.nb_hidden_dim * 2)]
        )

        # we pass the input through the mamba2 model
        x = self.mamba2(x)

        # we add another embedding for the time flags
        x = (
            x
            * time_flags_embeddings[
                :, :, (self.nb_hidden_dim * 2) : (self.nb_hidden_dim * 3)
            ]
            + time_flags_embeddings[:, :, (self.nb_hidden_dim * 3) :]
        )

        # we pass the output through the output layer
        x = self.output_layer(x)

        return x
