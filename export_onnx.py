"""
export_onnx.py -- export the three trained DMCNet fc_layers to ONNX.

Run once:
    .venv/bin/python export_onnx.py

Produces:
    docs/models/net_0.onnx
    docs/models/net_1.onnx
    docs/models/net_2.onnx

The ONNX net takes a single batched input of shape (N, 3148)
(obs_dim=335 concatenated with a one-hot action vector of dim=2813)
and outputs Q-values of shape (N,).

(using docs instead of web so that Github Pages is happy)
"""
import os, sys

if len(sys.argv) > 1 and sys.argv[1] == '-m':
    sys.exit(0)

import torch
import torch.nn as nn

OBS_DIM = 335       # 6*52 + 4 + 5*3 + 4  (3-player obs)
ACTION_DIM = 2813   # NUM_ACTIONS from tan_rlcard
INPUT_DIM = OBS_DIM + ACTION_DIM  # 3148

MODEL_DIR = "experiments/tan_dmc/tan"
OUT_DIR = "docs/models"
os.makedirs(OUT_DIR, exist_ok=True)

for seat in range(3):
    pth = os.path.join(MODEL_DIR, f"{seat}_2081369600.pth") # change this later
    out = os.path.join(OUT_DIR, f"net_{seat}.onnx")

    agent = torch.load(pth, map_location="cpu", weights_only=False)
    net: nn.Module = agent.net
    net.eval()

    # Wrap so we export only the sequential (obs+action already concatenated)
    class FlatNet(nn.Module):
        def __init__(self, fc):
            super().__init__()
            self.fc = fc
        def forward(self, x):          # x: (N, 3148)
            return self.fc(x).flatten()

    flat = FlatNet(net.fc_layers)
    dummy = torch.zeros(1, INPUT_DIM)

    torch.onnx.export(
        flat, dummy, out,
        input_names=["input"],
        output_names=["q_values"],
        dynamic_axes={"input": {0: "batch"}, "q_values": {0: "batch"}},
        opset_version=17,
    )

    # Inline any external data so the browser gets a single self-contained file
    import onnx
    model = onnx.load(out)
    onnx.save_model(model, out, save_as_external_data=False)
    # Remove the stale .data file if present
    data_file = out + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
    print(f"Exported seat {seat} -> {out} ({os.path.getsize(out)//1024} KB)")

print("Done. Copy docs/ to any static host.")
