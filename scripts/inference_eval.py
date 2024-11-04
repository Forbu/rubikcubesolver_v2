"""
Inference script for the reward guidance model.
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


def main():
    # init wandb
    # init with key
    # wandb.init(project="reward-guidance-rubiks", entity="forbu14")

    # first init random jax key
    key = jax.random.PRNGKey(142)

    batch_size = 64
    global_batch_size = 1000
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
        nb_hidden_dim=512,
        nb_input_dim=9 * 6 * 6,
        nb_output_dim=9 * 6 * 6,
        chunk_size=2,
        device=DEVICE,
    )

    pl_model = PLRewardGuidanceModel(model=model)

    print("Model created")
    count_parameters(pl_model.model)

    # load model from models/ folder
    path_model = "models/model_99.pt"
    pl_model.load_state_dict(torch.load(path_model))

    if cuda_available:
        pl_model.model = pl_model.model.to("cuda")
        pl_model = pl_model.to("cuda")

    # set inference mode
    pl_model.eval()

    # generate data
    batch = replay_buffer.sample()

    # get the init states, future states and rewards
    init_states, future_states, rewards = (
        torch.from_numpy(np.array(batch["state_past"])).to(DEVICE),
        torch.from_numpy(np.array(batch["state_future"])).to(DEVICE),
        torch.from_numpy(np.array(batch["reward"])).to(DEVICE),
    )

    # now we can generate data
    result_generation, reward_gen = pl_model.generate(
        nb_batch=64, nb_iter=100.0, init_states=init_states
    )

    print(result_generation[0, :, :, :])

    # now we want to retrieve the data
    # and plot the results
    init_value = np.argmax(np.array(batch["state_past"]), axis=-1)

    result_generation = np.argmax(result_generation.detach().cpu().numpy(), axis=-1)

    # now we want to plot the results
    # we can use matplotlib or plotly
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # for the first batches we plot the init value
    for idx in range(4):
        init_value_tmp = init_value[idx, 0, :]

        # reshape to get image
        init_value_tmp = init_value_tmp.reshape(6, 3, 3)
        init_value_tmp = init_value_tmp.reshape(6* 3, 3)
        
        # same thing but for the generated value
        result_generation_tmp = result_generation[idx, 0, :].reshape(6, 3, 3).reshape(6 * 3, 3)

        axs[0].imshow(init_value_tmp)
        axs[0].set_title("Init value")
        axs[1].imshow(result_generation_tmp)
        axs[1].set_title("Generated value")

        # save images in images/
        plt.savefig(f"images/image_{idx}.png")


if __name__ == "__main__":
    main()
