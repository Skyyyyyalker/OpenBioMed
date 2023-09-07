import sys
import os.path as osp
path = osp.dirname(osp.abspath(''))
sys.path.append(path)
sys.path.append(osp.join(path, "open_biomed"))
print(path)

import json
import torch
from open_biomed.utils import fix_path_in_config
from open_biomed.models.multimodal import BioMedGPTV

config = json.load(open("../configs/encoders/multimodal/biomedgptv.json", 'r'))
fix_path_in_config(config, path)
print("Config: ", config)

device = torch.device("cuda:0")
config["network"]["device"] = device
model = BioMedGPTV(config["network"])
ckpt = torch.load("../ckpts/fusion_ckpts/biomedgpt_10b.pth")    # i dont have this ckpt!
model.load_state_dict(ckpt)
model = model.to(device)
model.eval()
print("Finish loading model")