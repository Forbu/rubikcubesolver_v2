"""
Training script for the reward guidance model.
"""

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

    print("Generating data...")
    buffer, buffer_list = generate_random_data(
        key=key,
        batch_size=batch_size,
        global_batch_size=global_batch_size,
        nb_init_seq=1,
        nb_future_seq=10,
    )

    key, subkey = jax.random.split(key)

    replay_buffer = RewardGuidanceBuffer(buffer, buffer_list, global_batch_size // batch_size, subkey)

    print("Data generated. Now training...")

    # init model
    model = RewardGuidanceModel()
    pl_model = PLRewardGuidanceModel(model=model)

    optimizer = torch.optim.Adam(pl_model.parameters(), lr=1e-3)

    nb_epochs = 10
    # here we need to create a custom loop for training
    for epoch in range(nb_epochs):
        for i in range(replay_buffer.nb_games):
            batch = replay_buffer.sample()

            print(batch)

            # convert batch (dict) to torch tensors
            batch_torch = {}

            for k in batch.keys():
                batch_torch[k] = torch.from_numpy(np.array(batch[k]))

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
