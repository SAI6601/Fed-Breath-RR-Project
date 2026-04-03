import torch
from model import AttentionBiLSTM
from opacus.validators import ModuleValidator

m = AttentionBiLSTM()
from opacus.layers import DPLSTM
dp_lstm = DPLSTM(
    input_size=m.lstm.input_size, hidden_size=m.lstm.hidden_size,
    num_layers=m.lstm.num_layers, batch_first=m.lstm.batch_first,
    bidirectional=m.lstm.bidirectional, dropout=m.lstm.dropout
)
m.lstm = dp_lstm

m.load_state_dict(torch.load("centralized_model.pth", weights_only=True))

m = ModuleValidator.unfix(m)

print("SUCCESS: Model unfixed! LSTM layer is now:", type(m.lstm))
print("State dict keys sample:", list(m.state_dict().keys())[:5])
