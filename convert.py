import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

scripted_model = torch.jit.load("manually_added_output/best.torchscript")
optimized = optimize_for_mobile(scripted_model)
optimized._save_for_lite_interpreter("best.ptl")