import jittor as jt
from jittor.dataset import DataLoader
from model import TCP
from config import GlobalConfig
from data import CARLA_Data
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

jt.flags.use_cuda = jt.has_cuda 

config = GlobalConfig()
net = TCP(config)
ckpt = jt.load('tcp_b2d_jittor.pkl') 
new_state_dict = OrderedDict()

for key, value in ckpt.items():
    new_key = key.replace("model.", "")
    new_state_dict[new_key] = value

net.load_state_dict(new_state_dict)
# use cuda
net.cuda()
# set to eval mode
net.eval()

config.val_data = 'tcp_bench2drive-val.npy'
batch_size=100

val_set = CARLA_Data(root=config.root_dir_all, data_path=config.val_data, img_aug=False)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=16)

# Iterate over the validation set
l2_05 = []
l2_1 = []
l2_15 = []
l2_2 = []
length = len(val_set.front_img) # samples size not batch length

with jt.no_grad():
    for index, batch in enumerate(tqdm(val_loader)):
        front_img = jt.Var(batch['front_img']).to('cuda')
        speed = jt.Var(batch['speed']).to(dtype=jt.float32).view(-1, 1) / 12.
        target_point = jt.Var(batch['target_point']).to(dtype=jt.float32)
        command = jt.Var(batch['target_command']).permute(1, 0)
        state = jt.concat([speed, target_point, command], 1).to('cuda')
        gt_waypoints = jt.Var(batch['waypoints']).permute(2, 0, 1)

        pred = net(front_img, state, target_point.to('cuda'))

        l2_05.extend(np.linalg.norm(pred['pred_wp'][:, 0].detach().cpu().numpy() - gt_waypoints[:, 0].numpy(), axis=1).tolist())
        l2_1.extend(np.linalg.norm(pred['pred_wp'][:, 1].detach().cpu().numpy() - gt_waypoints[:, 1].numpy(), axis=1).tolist())
        l2_15.extend(np.linalg.norm(pred['pred_wp'][:, 2].detach().cpu().numpy() - gt_waypoints[:, 2].numpy(), axis=1).tolist())
        l2_2.extend(np.linalg.norm(pred['pred_wp'][:, 3].detach().cpu().numpy() - gt_waypoints[:, 3].numpy(), axis=1).tolist())

print("L2:", (sum(l2_05)/length + sum(l2_1)/length + sum(l2_15)/length + sum(l2_2)/length)/4)