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

import wandb
from prettytable import PrettyTable

# variable to check if cuda is available
cuda_available = torch.cuda.is_available()
DEVICE = "cuda" if cuda_available else "cpu"


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def init_replay_buffer(key, batch_size, global_batch_size, nb_init_seq, nb_future_seq):
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

    return replay_buffer

def convert_batch_to_proper_device(batch):
    # convert batch (dict) to torch tensors
    batch_torch = {}

    for k in batch.keys():
        if k in ["state_past", "state_future", "reward"]:
            batch_torch[k] = torch.from_numpy(np.array(batch[k]))

    # move tensors to cuda if available
    if cuda_available:
        batch_torch = {k: batch_torch[k].cuda() for k in batch_torch.keys()}

    return batch_torch

def main():
    # init wandb
    # init with key
    # load key in wandb_key.txt
    with open("wandb_key.txt", "r") as f:
        wandb_key = f.read().strip()

    wandb.login(key=wandb_key)
    wandb.init(project="reward-guidance-rubiks", entity="forbu14")

    # first init random jax key
    key = jax.random.PRNGKey(42)
    lr = 0.001

    batch_size = 64
    global_batch_size = 100000
    nb_init_seq = 1
    nb_future_seq = 11

    replay_buffer = init_replay_buffer(key, batch_size, global_batch_size, nb_init_seq, nb_future_seq)

    key = jax.random.PRNGKey(155)

    batch_size = 64
    global_batch_size_valid = 1000

    replay_buffer_valid = init_replay_buffer(key, batch_size, global_batch_size_valid, nb_init_seq, nb_future_seq)

    print("Data generated. Now training...")

    # init model
    model = RewardGuidanceModel(
        nb_init_states=nb_init_seq,
        nb_future_states=nb_future_seq,
        nb_hidden_dim=512,
        nb_input_dim=9 * 6 * 6,
        nb_output_dim=9 * 6 * 6,
        chunk_size=2,
        device=DEVICE,
    )

    pl_model = PLRewardGuidanceModel(model=model)

    print("Model created")
    count_parameters(pl_model.model)

    if cuda_available:
        pl_model.model = pl_model.model.to("cuda")
        pl_model = pl_model.to("cuda")

    optimizer = torch.optim.AdamW(pl_model.parameters(), lr=lr)

    nb_epochs = 100
    # here we need to create a custom loop for training
    for epoch in range(nb_epochs):
        for i in tqdm(range(replay_buffer.nb_games)):
            batch = replay_buffer.sample()

            # convert batch (dict) to torch tensors
            batch_torch = convert_batch_to_proper_device(batch) 
            
            # train model
            loss, (loss_worldmodel, loss_reward) = pl_model.training_step(batch_torch, i)

            # backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log loss
            # pl_model.log("loss", loss, on_step=True, on_epoch=True)

            # log metrics
            if i % 100 == 0:
                wandb.log({"loss": loss.cpu().item()})
                wandb.log({"loss_worldmodel": loss_worldmodel.cpu().item()})
                wandb.log({"loss_reward": loss_reward.cpu().item()})

                with torch.inference_mode():
                    batch = replay_buffer_valid.sample()

                    # convert batch (dict) to torch tensors
                    batch_torch = convert_batch_to_proper_device(batch) 
                    
                    # train model
                    loss, _ = pl_model.training_step(batch_torch, i)

                    wandb.log({"loss_valid": loss.cpu().item()})


        # save model
        torch.save(pl_model.state_dict(), f"models/model_{epoch}.pt")

if __name__ == "__main__":
    main()
