"""
Training script for the reward guidance model.
"""

from tqdm import tqdm

from rewardguidance.pl_models import PLRewardGuidanceModel
from rewardguidance.models import RewardGuidanceModel
from rewardguidance.data_generation import generate_random_data, RewardGuidanceBuffer

import numpy as np
import jax
import torch

# variable to check if cuda is available
cuda_available = torch.cuda.is_available()


def main():
    # first init random jax key
    key = jax.random.PRNGKey(42)

    batch_size = 10
    global_batch_size = 100
    nb_init_seq = 1
    nb_future_seq = 11

    print("Generating data...")
    buffer, buffer_list = generate_random_data(
        key=key,
        batch_size=batch_size,
        global_batch_size=global_batch_size,
        nb_init_seq=nb_init_seq,
        nb_future_seq=nb_future_seq,
    )

    key, subkey = jax.random.split(key)

    replay_buffer = RewardGuidanceBuffer(
        buffer,
        buffer_list,
        global_batch_size // batch_size,
        nb_init_seq,
        nb_future_seq,
        subkey,
    )

    print("Data generated. Now training...")

    # init model
    model = RewardGuidanceModel(
        nb_init_states=nb_init_seq,
        nb_future_states=nb_future_seq,
        nb_hidden_dim=128,
        nb_input_dim=9 * 6 * 6,
        nb_output_dim=9 * 6 * 6,
        chunk_size=2,
    )

    pl_model = PLRewardGuidanceModel(model=model)

    optimizer = torch.optim.Adam(pl_model.parameters(), lr=1e-3)

    nb_epochs = 10
    # here we need to create a custom loop for training
    for epoch in range(nb_epochs):
        for i in tqdm(range(replay_buffer.nb_games)):
            batch = replay_buffer.sample()

            # convert batch (dict) to torch tensors
            batch_torch = {}

            for k in batch.keys():
                if k in ["state_past", "state_future", "reward"]:
                    batch_torch[k] = torch.from_numpy(np.array(batch[k]))

                    #print("batch_torch[k].shape for k", k, batch_torch[k].shape)

            # move tensors to cuda if available
            if cuda_available:
                batch_torch = {k: batch_torch[k].cuda() for k in batch_torch.keys()}

            # train model
            loss = pl_model.training_step(batch_torch, i)

            # backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log loss
            pl_model.log("loss", loss, on_step=True, on_epoch=True)


if __name__ == "__main__":
    main()
