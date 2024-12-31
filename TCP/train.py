import jittor as jt
from jittor import init
from jittor import nn
from jittor import optim
from jittor.dataset import DataLoader
from tqdm import tqdm
import argparse
import os
from collections import OrderedDict
import sys
import numpy as np
from TCP.model import TCP
from TCP.data import CARLA_Data
from TCP.config import GlobalConfig

# Model definition
class TCP_planner(nn.Module):

    def __init__(self, config):
        super(TCP_planner, self).__init__()
        self.config = config
        self.model = TCP(config)

    def _load_state_dict(self, il_net, rl_state_dict, key_word):
        rl_keys = [k for k in rl_state_dict.keys() if (key_word in k)]
        il_keys = il_net.state_dict().keys()
        assert (len(rl_keys) == len(il_net.state_dict().keys())), f'mismatch number of layers loading {key_word}'
        new_state_dict = OrderedDict()
        for (k_il, k_rl) in zip(il_keys, rl_keys):
            new_state_dict[k_il] = rl_state_dict[k_rl]
        il_net.load_parameters(new_state_dict)

    def execute(self, front_img, state, target_point):
        return self.model(front_img, state, target_point)

def train(model, dataloader, optimizer, config, device, epoch):
    # set to train mode
    model.train()

    running_loss = 0.0
    total_loss = 0.0

    for (batch_idx, batch) in enumerate(tqdm(dataloader, desc=f'Training Epoch {(epoch + 1)}')):
        front_img = jt.Var(batch['front_img']).to(device)
        speed = jt.Var(batch['speed']).to(dtype=jt.float32).view(-1, 1).to(device) / 12.
        target_point = jt.Var(batch['target_point']).to(dtype=jt.float32).to(device)
        command = jt.Var(batch['target_command']).permute(1, 0).to(device)
        state = jt.contrib.concat([speed, target_point, command], dim=1)
        value = jt.Var(batch['value']).view(((- 1), 1)).to(device)
        feature = jt.Var(batch['feature']).to(device)
        gt_waypoint = jt.Var(batch['waypoints']).to(device)

        optimizer.zero_grad()

        # forward pass
        pred = model(front_img, state, target_point)

        # ------------------
        # Trajectory Branch Loss
        # ------------------
        action_loss = nn.nll_loss(nn.log_softmax(pred['action_index'], dim=1), batch['action_index'])
        speed_loss = nn.l1_loss(pred['pred_speed'], speed) * config.speed_weight
        value_loss = (nn.mse_loss(pred['pred_value_traj'], value) + nn.mse_loss(pred['pred_value_ctrl'], value)) * config.value_weight
        feature_loss = (nn.mse_loss(pred['pred_features_traj'], feature) + nn.mse_loss(pred['pred_features_ctrl'], feature)) * config.features_weight

        # ------------------
        # MultiStep Control Branch Loss
        # ------------------
        future_feature_loss = 0
        future_action_loss = 0
        
        for i in range(config.pred_len):
            action_loss = nn.nll_loss(nn.log_softmax(pred['future_action_index'][i], dim=1), batch['future_action_index'][i])
            future_action_loss += nn.nll_loss(nn.log_softmax(pred['future_action_index'][i], dim=1), batch['future_action_index'][i])
            future_feature_loss += nn.mse_loss(pred['future_feature'][i], batch['future_feature'][i]) * config.features_weight
        
        future_action_loss /= config.pred_len
        future_feature_loss /= config.pred_len

        wp_loss = nn.smooth_l1_loss(pred['pred_wp'].permute(1, 2, 0).to(dtype=jt.float32), gt_waypoint, reduction='none').mean() 

        loss = action_loss + speed_loss + value_loss + feature_loss + future_action_loss + future_feature_loss + wp_loss

        print(f'action_loss: {action_loss.item()}')
        print(f'speed_loss: {speed_loss.item()}')
        print(f'value_loss: {value_loss.item()}')
        print(f'feature_loss: {feature_loss.item()}')
        print(f'future_action_loss: {future_action_loss.item()}')
        print(f'future_feature_loss: {future_feature_loss.item()}')
        print(f'wp_loss: {wp_loss.item()}')
        print(f'loss: {loss.item()}')

        total_loss += loss.item()
        optimizer.zero_grad()
        # Back propogation
        optimizer.step(loss)
        
    average_loss = (total_loss / len(dataloader))
    print(f'Training Loss: {average_loss}')
    return average_loss

def validate(model, dataloader, config, device, epoch):
    model.eval()
    running_loss = 0.0

    with jt.no_grad():
        for batch in tqdm(dataloader, desc=f'Validation Epoch {(epoch + 1)}'):
            front_img = jt.Var(batch['front_img']).to(device)
            speed = jt.Var(batch['speed']).to(dtype=jt.float32).view(-1, 1).to(device) / 12.
            target_point = jt.Var(batch['target_point']).to(device)
            command = jt.Var(batch['target_command']).permute(1, 0).to(device)
            state = jt.contrib.concat([speed, target_point, command], dim=1)
            value = jt.Var(batch['value']).view(((- 1), 1)).to(device)
            feature = jt.Var(batch['feature']).to(device)
            gt_waypoint = jt.Var(batch['waypoints']).to(device)

            pred = model(front_img, state, target_point)

            # Calculate loss
            action_loss = nn.nll_loss(nn.log_softmax(pred['action_index'], dim=1), batch['action_index'].to(device))
            speed_loss = nn.l1_loss(pred['pred_speed'], speed) * config.speed_weight
            value_loss = (nn.mse_loss(pred['pred_value_traj'], value) + nn.mse_loss(pred['pred_value_ctrl'], value)) * config.value_weight
            feature_loss = (nn.mse_loss(pred['pred_features_traj'], feature) + nn.mse_loss(pred['pred_features_ctrl'], feature)) * config.features_weight
            wp_loss = nn.smooth_l1_loss(pred['pred_wp'].permute(1, 2, 0), gt_waypoint, reduction='none').mean()
            
            future_feature_loss = 0
            future_action_loss = 0

            for i in range(config.pred_len):
                action_loss_i = (nn.nll_loss(nn.log_softmax(pred['future_action_index'][i], dim=1), batch['future_action_index'][i])).to(device)
                future_action_loss += action_loss_i
                feature_loss_i = nn.mse_loss(pred['future_feature'][i], batch['future_feature'][i]).to(device) * config.features_weight
                future_feature_loss += feature_loss_i
            
            future_action_loss /= config.pred_len
            future_feature_loss /= config.pred_len

            loss = action_loss + speed_loss + value_loss + feature_loss + future_action_loss + future_feature_loss + wp_loss
            
            running_loss += loss.item()
            
            fb_error_mean = jt.abs_((pred['pred_wp'].permute(1, 2, 0)[:, :, 0] - gt_waypoint[:, :, 0])).mean().item()
            lr_error_mean = jt.abs_((pred['pred_wp'].permute(1, 2, 0)[:, :, 1] - gt_waypoint[:, :, 1])).mean().item()
            
            (predicted_indices, _) = jt.argmax(nn.log_softmax(pred['action_index'], dim=1), 1)
            
            correct = (predicted_indices == jt.Var(batch['action_index']).to(device)).float()
            accuracy = (correct.sum() / len(correct))
        
        avg_loss = (running_loss / len(dataloader))
        print(f'Epoch {(epoch + 1)} Validation Loss: {avg_loss}')

        return avg_loss

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    jt.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model.pth')
        jt.save(state, best_filepath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='TCP', help='Unique experiment identifier.')
    parser.add_argument('--epochs', type=int, default=27, help='Number of train epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--val_every', type=int, default=2, help='Validation frequency (epochs).')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus')
    
    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)

    config = GlobalConfig()

    device = ''
    if jt.has_cuda:
        device  = 'cuda'
        jt.flags.use_cuda = 1
    else:
        device = 'cpu'
        jt.flags.use_cuda = 0

    train_set = CARLA_Data(
        root=config.root_dir_all, 
        data_path=config.train_data, 
        img_aug=config.img_aug
        )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)

    print(f'Training samples: {len(train_set.front_img)}')

    val_set = CARLA_Data(
        root=config.root_dir_all, 
        data_path=config.val_data, 
        img_aug=False,
        )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=16)
    
    print(f'Validation samples: {len(val_set.front_img)}')

    model = TCP_planner(config).cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)
    scheduler = jt.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, config, device, epoch)
        scheduler.step()
        if ((epoch + 1) % args.val_every) == 0:
            val_loss = validate(model, val_loader, config, device, epoch)
            is_best = (val_loss < best_val_loss)
            if is_best:
                best_val_loss = val_loss
            save_checkpoint({'epoch': (epoch + 1), 'state_dict': model.state_dict(), 'best_val_loss': best_val_loss, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, is_best, checkpoint_dir=args.logdir, filename=f'checkpoint_{(epoch + 1)}.pth')
    

if (__name__ == '__main__'):
    main()
