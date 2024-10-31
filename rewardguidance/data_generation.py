# -*- coding: utf-8 -*-
"""
Created by Adrien Bufort
Created on 2024-10-31
This code is the property of Orange Innovation
All rights reserved.
"""
"""
Data generation for the reward guidance model.

Set of tools to generate data for the reward guidance model.
Basicly the data part.

We have two part here :

- The totally random generation of data.
- The generation of data from a pre-trained model trying to maximize the reward.

"""

import torch
import pytorch_lightning as pl
from typing import List, Tuple, Dict

import jax
import jax.numpy as jnp

import flashbax  # replay buffer tool
import jumanji

# GOAL OBSERVATION
GOAL_OBSERVATION = jnp.zeros((6, 3, 3))
for i in range(6):
    GOAL_OBSERVATION = GOAL_OBSERVATION.at[i, :, :].set(i)

ENV = jumanji.make("RubiksCube-v0")
jit_step = jax.jit(ENV.step)

def step_fn(state, key):
    """
    Simple step function
    We choose a random action
    """
    action = jax.random.randint(
        key=key,
        minval=ENV.action_spec.minimum,
        maxval=ENV.action_spec.maximum,
        shape=(3,),
    )

    new_state, timestep = jit_step(state, action)
    timestep.extras["action"] = action

    return new_state, timestep


def run_n_steps(state, key, n):
    random_keys = jax.random.split(key, n)
    state, rollout = jax.lax.scan(step_fn, state, random_keys)
    return rollout


def generate_random_data(
    batch_size: int, global_batch_size: int, nb_init_seq: int, nb_future_seq: int, key: jax.random.PRNGKey,
    buffer_max_batch_size: int = 1024 * 100
):
    env, buffer = init_env_buffer(sample_batch_size=batch_size, max_length=buffer_max_batch_size)

    nb_games = global_batch_size
    len_seq = nb_init_seq + nb_future_seq

    state_first = jnp.zeros((6, 3, 3))
    state_next = jnp.zeros((len_seq, 6, 3, 3))
    action = jnp.zeros((len_seq, 3))

    # transform state to int8 type
    state_first = state_first.astype(jnp.int8)
    state_next = state_next.astype(jnp.int8)

    # action to int32 type
    action = action.astype(jnp.int32)

    reward = jnp.zeros((1))

    vmap_reset = jax.vmap(jax.jit(env.reset))

    buffer_list = buffer.init(
        {
            "action": action,
            "reward": reward,
            "state_histo": state_next,
        }
    )

    vmap_step = jax.vmap(run_n_steps, in_axes=(0, 0, None))

    buffer, buffer_list = fast_gathering_data_diffusion(
        env,
        vmap_reset,
        vmap_step,
        nb_games, 
        len_seq,
        buffer,
        buffer_list,
        key,
    )

    return buffer, buffer_list



def reshape_sample(sample):
    sample.experience["state_histo"] = sample.experience["state_histo"].reshape(
        (
            sample.experience["state_histo"].shape[0],
            sample.experience["state_histo"].shape[1],
            54,
        )
    )

    # one hot encoding for state_histo
    sample.experience["state_histo"] = jax.nn.one_hot(
        sample.experience["state_histo"],
        num_classes=6,
        axis=-1,
    )

    # batch creation
    batch = sample.experience
    len_seq = batch["state_histo"].shape[1]

    batch["state_past"] = batch["state_histo"][:, : len_seq // 4, :, :]
    batch["state_future"] = batch["state_histo"][:, len_seq // 4 :, :, :]

    batch["action_inverse"] = sample.experience["action"][:, 1:, :]

    # flatten the action_inverse to only have batch data
    batch["action_inverse"] = jnp.reshape(
        batch["action_inverse"],
        (batch["action_inverse"].shape[0] * batch["action_inverse"].shape[1], -1),
    )

    # now we can one hot encode the action_inverse
    action_inverse_0 = jax.nn.one_hot(
        batch["action_inverse"][:, 0], num_classes=6, axis=-1
    )
    action_inverse_1 = jax.nn.one_hot(
        batch["action_inverse"][:, 2], num_classes=3, axis=-1
    )

    batch["action_inverse"] = jnp.concatenate(
        [action_inverse_0, action_inverse_1], axis=1
    )

    state_histo_inverse_t = sample.experience["state_histo"][:, :-1, :, :]
    state_histo_inverse_td1 = sample.experience["state_histo"][:, 1:, :, :]

    batch["state_histo_inverse_t"] = state_histo_inverse_t
    batch["state_histo_inverse_td1"] = state_histo_inverse_td1

    # we flatten the two state_histo_inverse
    batch["state_histo_inverse_t"] = jnp.reshape(
        batch["state_histo_inverse_t"],
        (
            batch["state_histo_inverse_t"].shape[0]
            * batch["state_histo_inverse_t"].shape[1],
            -1,
        ),
    )
    batch["state_histo_inverse_td1"] = jnp.reshape(
        batch["state_histo_inverse_td1"],
        (
            batch["state_histo_inverse_td1"].shape[0]
            * batch["state_histo_inverse_td1"].shape[1],
            -1,
        ),
    )

    return batch

class RewardGuidanceBuffer:
    def __init__(self, buffer, buffer_list, nb_games, key):
        self.buffer = buffer
        self.buffer_list = buffer_list
        self.key = key
        self.nb_games = nb_games
    def sample(self):
        self.key, subkey = jax.random.split(self.key)
        samples = self.buffer.sample(self.buffer_list, subkey)
        return reshape_sample(samples)


    def add(self, new_data: List[Dict]):
        """
        Add new data to the replay buffer
        
        new_data is a list of dictionary with the following keys :
            - state_first
            - action
            - reward
            - state_next
            - action_pred
        """
        for data in new_data:
            self.buffer.add(self.buffer_list, data)


def fast_gathering_data_diffusion(
    env, vmap_reset, vmap_step, batch_size, rollout_length, buffer, buffer_list, key
):
    key1, key2 = jax.random.split(key)

    keys = jax.random.split(key1, batch_size)
    state, timestep = vmap_reset(keys)

    # Collect a batch of rollouts
    keys = jax.random.split(key2, batch_size)
    rollout = vmap_step(state, keys, rollout_length)

    # we retrieve the information from the state_first (state), state_next,
    #  the action and the reward
    state_histo = rollout.observation.cube
    action = rollout.extras["action"]

    # now we compute the reward :
    reward = jnp.zeros((batch_size, rollout_length))

    # for each batch / rollout we compute the mean difference between the
    # observation and the goal
    # we repeat the goal_observation to match the shape of the observation
    goal_observation = jnp.repeat(
        GOAL_OBSERVATION[None, None, :, :, :], batch_size, axis=0
    )
    goal_observation = jnp.repeat(goal_observation, rollout_length, axis=1)
    reward = jnp.where(state_histo != goal_observation, -1.0, 1.0)

    reward = reward.mean(axis=[2, 3, 4])
    reward = reward[:, -1] - reward[:, rollout_length//4]

    for idx_batch in range(batch_size):
        buffer_list = buffer.add(
            buffer_list,
            {
                "action": action[idx_batch],
                "reward": reward[idx_batch],
                "state_histo": state_histo[idx_batch],
            },
        )

    return buffer, buffer_list


def generate_data_from_model(
    pl_model: pl.LightningModule, batch_size: int, nb_init_seq: int, nb_future_seq: int
):
    pass


def init_replay_buffer():
    pass


def update_replay_buffer(
    replay_buffer: RewardGuidanceBuffer, new_data: List[Tuple[torch.Tensor, torch.Tensor]]
):
    pass


def init_env_buffer(max_length=1024 * 100, sample_batch_size=32):
    """
    Initializes the environment and buffer for the Rubik's Cube game.

    Returns:
        env (jumanji.Environment): The initialized environment.
        buffer (flashbax.Buffer): The initialized buffer.
    """
    env = jumanji.make("RubiksCube-v0")

    buffer = flashbax.make_item_buffer(
        max_length=max_length, min_length=2, sample_batch_size=sample_batch_size
    )

    return env, buffer
