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
import numpy as np

from rewardguidance.mamba2 import Mamba2, Mamba2Config, RMSNorm
from x_transformers import TransformerWrapper, Encoder

class RewardGuidanceModel(nn.Module):
    """
    Reward guidance model with a time flags and a reward value.

    Shortcut value is used to indicate if the trajectory is a shortcut or not.
    Time flags is used to indicate the time of the trajectory.

    For shortcut value see paper : https://arxiv.org/pdf/2410.12557

    Time is the classical time value in diffusion models.

    """

    def __init__(
        self,
        nb_future_states: int = 16,
        nb_init_states: int = 1,
        nb_hidden_dim: int = 512,
        nb_input_dim: int = 9*6*6,
        nb_output_dim: int = 9*6*6,
        chunk_size: int = 32,
        device = None,
    ):
        super().__init__()

        self.nb_future_states = nb_future_states
        self.nb_init_states = nb_init_states
        self.nb_hidden_dim = nb_hidden_dim
        self.nb_input_dim = nb_input_dim
        self.nb_output_dim = nb_output_dim


        # Initialize the transformer encoder
        self.transformer_encoder = Encoder(
            dim = nb_hidden_dim,
            depth = 6,
            heads = 8,
            rotary_pos_emb=True,
        )

        # Initialize linear layers with Xavier/Glorot initialization
        self.input_layer = nn.Linear(nb_input_dim, nb_hidden_dim)
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        
        self.output_layer = nn.Linear(nb_hidden_dim, nb_output_dim)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        # Initialize states embedding with a smaller standard deviation
        self.states_embedding = nn.Parameter(
            torch.randn(nb_future_states + nb_init_states, nb_hidden_dim) * 0.02
        )

        # Initialize time flags embedding
        self.time_flags_embedding = nn.Linear(1, nb_hidden_dim * 4)
        nn.init.xavier_uniform_(self.time_flags_embedding.weight)
        nn.init.zeros_(self.time_flags_embedding.bias)

        # Initialize reward value embedding
        self.reward_value_embedding = nn.Linear(1, nb_hidden_dim * 4)
        nn.init.xavier_uniform_(self.reward_value_embedding.weight)
        nn.init.zeros_(self.reward_value_embedding.bias)

        # Initialize reward layer
        self.reward_layer = nn.Linear(nb_hidden_dim, 1)
        nn.init.xavier_uniform_(self.reward_layer.weight)
        nn.init.zeros_(self.reward_layer.bias)

        # Add layer normalization before applying embeddings
        self.pre_norm = nn.LayerNorm(nb_hidden_dim)
        self.post_norm = nn.LayerNorm(nb_hidden_dim)

    def forward(
        self,
        init_states: torch.Tensor,
        future_states: torch.Tensor,
        time_flags: torch.Tensor,
        reward_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            init_states: (batch_size, seq_len, 6*9, 6)
            future_states: (batch_size, seq_len, 6*9, 6) noisy futur state (diffusion setup)
            time_flags: (batch_size, 1) value between 0 and 1 (for the time of the trajectory)
            reward_value: (batch_size, 1) value between 0 and 1 (for the reward value)
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
        reward_value_embeddings = self.reward_value_embedding(
            reward_value
        ).unsqueeze(1)

        # we concatenate the init states and the future states embeddings
        x = torch.cat([init_states, future_states], dim=1)

        # we pass the input through the input layer
        x = self.input_layer(x)

        # we concatenate the states embeddings and the input embeddings
        x = x + states_embeddings

        # modification of the input to add the time flags embeddings [:, :, :self.nb_hidden_dim]
        x = (
            x
            * (1. + time_flags_embeddings[:, :, : self.nb_hidden_dim])
            * (1. + reward_value_embeddings[:, :, : self.nb_hidden_dim])
            + time_flags_embeddings[:, :, self.nb_hidden_dim : (self.nb_hidden_dim * 2)]
            + reward_value_embeddings[
                :, :, self.nb_hidden_dim : (self.nb_hidden_dim * 2)
            ]
        )

        # Add residual connection around transformer
        residual = x
        x = self.transformer_encoder(x)
        x = x + residual


        # Add layer normalization before applying embeddings
        x = self.pre_norm(x)
        x = (
            x
            * (1. + time_flags_embeddings[:, :, :self.nb_hidden_dim])  # Add 1 to prevent zeroing
            * (1. + reward_value_embeddings[:, :, :self.nb_hidden_dim])
            + time_flags_embeddings[:, :, self.nb_hidden_dim:(self.nb_hidden_dim * 2)]
            + reward_value_embeddings[:, :, self.nb_hidden_dim:(self.nb_hidden_dim * 2)]
        )
        x = self.post_norm(x)

        # we pass the output through the output layer
        state_final = self.output_layer(x)

        # Instead of mean pooling, consider using the final state or attention pooling
        # reward = x.mean(dim=1)  # old code
        
        # Option 1: Use the final state
        reward = x[:, -1, :]  # Use the last state
        reward = self.reward_layer(reward)


        # Reshape more explicitly
        state_final = state_final[:, -self.nb_future_states:, :]  # Take only future states
        state_final = einops.rearrange(state_final, 'b s (h w) -> b s h w', h=9*6, w=6)

        return state_final, reward
