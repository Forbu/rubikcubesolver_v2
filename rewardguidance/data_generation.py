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

import random

import jax
import jax.numpy as jnp

import flashbax  # replay buffer tool
import jumanji

# GOAL OBSERVATION
GOAL_OBSERVATION = jnp.zeros((6, 3, 3))
for i in range(6):
    GOAL_OBSERVATION = GOAL_OBSERVATION.at[i, :, :].set(i)


def generate_random_data(
    batch_size: int, nb_init_seq: int, nb_future_seq: int, key: jax.random.PRNGKey
):
    env, buffer = init_env_buffer(sample_batch_size=batch_size)

    nb_games = batch_size
    len_seq = nb_init_seq + nb_future_seq

    state_first = jnp.zeros((6, 3, 3))
    state_next = jnp.zeros((len_seq, 6, 3, 3))
    action = jnp.zeros((len_seq, 3))
    action_proba = jnp.zeros((len_seq, 9))

    # transform state to int8 type
    state_first = state_first.astype(jnp.int8)
    state_next = state_next.astype(jnp.int8)

    # action to int32 type
    action = action.astype(jnp.int32)

    reward = jnp.zeros((len_seq))

    jit_step = jax.jit(env.step)

    buffer_list = buffer.init(
        {
            "state_first": state_first,
            "action": action,
            "reward": reward,
            "state_next": state_next,
            "action_pred": action_proba,
        }
    )

    def step_fn(state, key):
        """
        Simple step function
        We choose a random action
        """

        action = jax.random.randint(
            key=key,
            minval=env.action_spec.minimum,
            maxval=env.action_spec.maximum,
            shape=(3,),
        )

        new_state, timestep = jit_step(state, action)
        timestep.extras["action"] = action
        timestep.extras["action_pred"] = jnp.zeros((9,))

        return new_state, timestep

    def run_n_steps(state, key, n):
        random_keys = jax.random.split(key, n)
        state, rollout = jax.lax.scan(step_fn, state, random_keys)

        return rollout

    vmap_reset = jax.vmap(jax.jit(env.reset))
    vmap_step = jax.vmap(run_n_steps, in_axes=(0, 0, None))

    key, subkey = jax.random.split(key)

    buffer, buffer_list = fast_gathering_data(
        env,
        vmap_reset,
        vmap_step,
        nb_games,
        len_seq,
        buffer,
        buffer_list,
        subkey,
    )

    return buffer, buffer_list


class ReplayBuffer:
    def __init__(self, buffer, buffer_list, key):
        self.buffer = buffer
        self.buffer_list = buffer_list
        self.key = key

    def sample(self):
        return self.buffer.sample(self.buffer_list)

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


def fast_gathering_data(
    env, vmap_reset, vmap_step, batch_size, rollout_length, buffer, buffer_list, key
):
    """
    Fast gathering data for the Rubik's Cube game.
    Params :
        env : the environment (coming from jumanji.make)
        vmap_reset : the reset function (coming from jax.vmap(jax.jit(env.reset)))
        vmap_step : the step function (coming from jax.vmap(run_n_steps, in_axes=(0, 0, None)))
        batch_size : the batch size
        rollout_length : the length of the rollout
        buffer : the buffer (coming from flashbax.make_item_buffer)
        buffer_list : the buffer list (empty at the beginning)
        key : the key
    """
    key1, key2 = jax.random.split(key)

    keys = jax.random.split(key1, batch_size)
    state, timestep = vmap_reset(keys)

    # Collect a batch of rollouts
    keys = jax.random.split(key2, batch_size)
    rollout = vmap_step(state, keys, rollout_length)

    # we retrieve the information from the state_first (state), state_next,
    #  the action and the reward
    state_first = timestep.observation.cube
    state_next = rollout.observation.cube
    action = rollout.extras["action"]
    action_pred = rollout.extras["action_pred"]

    # now we compute the reward :
    reward = jnp.zeros((batch_size, rollout_length))

    # for each batch / rollout we compute the mean difference between the
    # observation and the goal
    # we repeat the goal_observation to match the shape of the observation
    goal_observation = jnp.repeat(
        GOAL_OBSERVATION[None, None, :, :, :], batch_size, axis=0
    )
    goal_observation = jnp.repeat(goal_observation, rollout_length, axis=1)
    reward = jnp.where(state_next != goal_observation, -1.0, 1.0)

    reward = reward.mean(axis=[2, 3, 4])

    for idx_batch in range(batch_size):
        buffer_list = buffer.add(
            buffer_list,
            {
                "state_first": state_first[idx_batch],
                "action": action[idx_batch],
                "reward": reward[idx_batch],
                "state_next": state_next[idx_batch],
                "action_pred": action_pred[
                    idx_batch
                ],  # here we could manage the action prediction
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
    replay_buffer: ReplayBuffer, new_data: List[Tuple[torch.Tensor, torch.Tensor]]
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
