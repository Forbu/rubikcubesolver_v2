import pytest
import torch
from rewardguidance.pl_models import PLRewardGuidanceModel

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nb_future_states = 5
        self.linear = torch.nn.Linear(10, 6)
        
    def forward(self, init_states, future_states, time_flags, shortcut_value):
        batch_size = init_states.shape[0]
        # Return mock state predictions and reward predictions
        state_preds = torch.randn(batch_size, future_states.shape[1], 9*6, 6)
        reward_preds = torch.randn(batch_size, 1)
        return state_preds, reward_preds

@pytest.fixture
def model():
    mock_model = MockModel()
    return PLRewardGuidanceModel(model=mock_model, lr=1e-3)

def test_model_initialization(model):
    assert model.lr == 1e-3
    assert model.dirichlet_alpha == 0.5
    assert model.nb_dim_dirichlet == 6

def test_training_step(model):
    batch_size = 4
    nb_init_seq = 3
    nb_future_seq = 5
    
    # Create mock batch
    batch = {
        "state_past": torch.randn(batch_size, nb_init_seq, 9*6, 6),
        "state_future": torch.randn(batch_size, nb_future_seq, 9*6, 6),
        "reward": torch.randn(batch_size, 1)
    }
    
    # Run training step
    loss, (loss_reg, loss_reward) = model.training_step(batch, 0)
    
    # Check outputs
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert isinstance(loss_reg, torch.Tensor)
    assert isinstance(loss_reward, torch.Tensor)

def test_generate(model):
    batch_size = 4
    nb_init_seq = 3
    
    # Create mock initial states
    init_states = torch.randn(batch_size, nb_init_seq, 9*6, 6)
    
    # Test generation
    future_states, rewards = model.generate(
        nb_batch=batch_size,
        nb_iter=10,
        init_states=init_states
    )
    
    # Check outputs
    assert future_states.shape == (batch_size, model.model.nb_future_states, 9*6, 6)
    assert rewards.shape == (batch_size, 1)
    assert not future_states.requires_grad
    assert not rewards.requires_grad

def test_forward(model):
    batch_size = 4
    nb_init_seq = 3
    nb_future_seq = 5
    
    # Create mock inputs
    init_states = torch.randn(batch_size, nb_init_seq, 9*6, 6)
    future_states = torch.randn(batch_size, nb_future_seq, 9*6, 6)
    time_flags = torch.rand(batch_size, 1)
    shortcut_value = torch.rand(batch_size, 1)
    
    # Test forward pass
    state_preds, reward_preds = model(
        init_states, 
        future_states,
        time_flags,
        shortcut_value
    )
    
    # Check outputs
    assert state_preds.shape == (batch_size, nb_future_seq, 9*6, 6)
    assert reward_preds.shape == (batch_size, 1) 