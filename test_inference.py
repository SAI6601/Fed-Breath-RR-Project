import torch
import numpy as np
from model import AttentionBiLSTM
from opacus.validators import ModuleValidator

m = AttentionBiLSTM()
m = ModuleValidator.fix(m)
m.load_state_dict(torch.load("centralized_model.pth", weights_only=True))
m.eval()

x = torch.randn(1, 1, 300)
try:
    with torch.no_grad():
        out = m(x, return_attention=True, return_anomaly=True)
    print("Inference succeeded!")
except Exception as e:
    print(f"Inference failed: {e}")
    import traceback
    traceback.print_exc()
