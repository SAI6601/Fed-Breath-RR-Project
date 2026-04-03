import torch
from model import AttentionBiLSTM
from opacus.validators import ModuleValidator

m = AttentionBiLSTM()
m = ModuleValidator.fix(m)
try:
    m.load_state_dict(torch.load("centralized_model.pth", weights_only=True))
    print("Load succeeded after fix!")
except Exception as e:
    print(f"Load failed: {e}")
