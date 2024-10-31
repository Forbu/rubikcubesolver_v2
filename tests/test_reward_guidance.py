"""
Test the reward guidance model.
"""

import torch

from rewardguidance.models import RewardGuidanceModel

def test_reward_guidance_model():
    """
    Test the reward guidance model.
    """

    nb_batch = 7
    nb_init_states = 2
    nb_future_states = 6
    nb_hidden_dim = 128

    model = RewardGuidanceModel(
        nb_future_states=nb_future_states,
        nb_init_states=nb_init_states,
        nb_hidden_dim=nb_hidden_dim,
        nb_input_dim=9*6*6,
        nb_output_dim=9*6*6
    )
    assert model is not None


    # we create a dummy input for the init states
    input_init_dummy = torch.randn(nb_batch, nb_init_states, 9*6, 6)

    # we create a dummy input for the future states
    input_future_dummy = torch.randn(nb_batch, nb_future_states, 9*6, 6)

    # init time_flags
    time_flags = torch.zeros(nb_batch, 1)

    # shortcut_value
    shortcut_value = torch.zeros(nb_batch, 1)

    # we pass the inputs through the model
    output = model(input_init_dummy, input_future_dummy, time_flags, shortcut_value)
    assert output is not None