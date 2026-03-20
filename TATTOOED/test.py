import numpy as np
import torch
from Tattooed_Watermark import TattooedWatermarker
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class TinyDNN(nn.Module):
    def __init__(self):
        super(TinyDNN, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def main():
    # Model is trained, ready to use. For now, we use dummy.
    model = TinyDNN()

    # Generate a random watermark payload
    P = 256  # payload length
    rng = np.random.default_rng(42) # for reproducibility
    bits = rng.integers(0, 2, size=P, dtype=np.uint8) 
    #-----------------------------------#

    # Create the watermark
    wm = TattooedWatermarker(b"my-secret-key", ratio=0.2, gamma=9e-4)
    vec = parameters_to_vector(model.parameters()).detach().cpu().numpy()
    vec_wtm = wm.embed_watermark(vec, bits) # Embed watermark

    # Write back the parameters to the model
    param_ref = next(model.parameters())
    vec_wtm_torch = torch.from_numpy(vec_wtm).to(device=param_ref.device, dtype=param_ref.dtype)
    vector_to_parameters(vec_wtm_torch, model.parameters())
    #-----------------------------------#

    # Extract watermark and test
    vec_after = parameters_to_vector(model.parameters()).detach().cpu().numpy()
    bits_extracted = wm.extract_watermark(vec_after, P)
    ok, ber = wm.verify_watermark(bits, bits_extracted, threshold=0)

    print(f"Watermark verified={ok},  BER={ber:.0e}")  # expect: True, BER = 0
    #-----------------------------------#


if __name__ == "__main__":
    main()


