"""
Test the reward guidance model.
"""

import torch

from rewardguidance.models import RewardGuidanceModel

def test_reward_guidance_model():
    """
    Test the reward guidance model.
    """
    model = RewardGuidanceModel()
    assert model is not None

    nb_batch = 7
    nb_init_states = 2
    nb_future_states = 5

    # we create a dummy input for the init states
    input_init_dummy = torch.randn(nb_batch, nb_init_states, 3)

    # we create a dummy input for the future states
    input_future_dummy = torch.randn(nb_batch, nb_future_states, 3)

    # we pass the inputs through the model
    output = model(input_init_dummy, input_future_dummy)
    assert output is not None