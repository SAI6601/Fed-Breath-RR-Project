from model import AttentionBiLSTM
from opacus.validators import ModuleValidator

m = AttentionBiLSTM()
try:
    m = ModuleValidator.fix(m)
    print("Fix succeeded!")
except Exception as e:
    print(f"Fix failed: {e}")
