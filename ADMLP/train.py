import argparse
import os
import jittor.nn as nn
import jittor as jt
from jittor.dataset import DataLoader
from ADMLP.model import ADMLP
from ADMLP.data import CARLA_Data
from ADMLP.config import GlobalConfig
from tqdm import tqdm
import sys

class ADMLP_planner(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.model = ADMLP(config)
	
	def forward(self, batch):
		pass

def train(model, dataloader, optimizer, config, device, epoch):
	model.train()

	total_loss = 0.0

	for (batch_idx, batch) in enumerate(tqdm(dataloader, desc=f'Training Epoch {epoch+1}')):

		predict = model.model(batch)
		waypoints = batch['waypoints'].to(device)
		theta = batch['thetas'].to(device)

		x_loss = nn.smooth_l1_loss(predict[:,:,0], waypoints[:,:,0], reduction='mean')
		y_loss = nn.smooth_l1_loss(predict[:,:,1], waypoints[:,:,1], reduction='mean')
		theta  = nn.smooth_l1_loss(predict[:,:,2], theta, reduction='mean')

		loss = x_loss + y_loss + theta	
		total_loss += loss.item()

		optimizer.zero_grad()
		# Backpropagation
		optimizer.step(loss)

		if batch_idx % 10 == 0:
			print(f'Epoch {epoch+1}, batch {batch_idx}, loss: {loss.item()}')

	average_loss = total_loss / len(dataloader)
	print(f'Epoch {epoch+1}, average loss: {average_loss}')
	return average_loss

def validate(model, dataloader, config, device, epoch):
	model.eval()

	total_loss = 0.0

	with jt.no_grad():
		for batch in tqdm(dataloader, desc=f'Validation Epoch {epoch+1}'):
			predict = model.model(batch)
			waypoints = batch['waypoints'].to(device)
			theta = batch['thetas'].to(device)

			x_loss = nn.smooth_l1_loss(predict[:,:,0], waypoints[:,:,0], reduction='mean')
			y_loss = nn.smooth_l1_loss(predict[:,:,1], waypoints[:,:,1], reduction='mean')
			theta  = nn.smooth_l1_loss(predict[:,:,2], theta, reduction='mean')

			loss = x_loss + y_loss + theta	
			total_loss += loss.item()

	average_loss = total_loss / len(dataloader)
	print(f'Epoch {epoch+1}, Validation average loss: {average_loss}')
	return average_loss

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    jt.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model.pth')
        jt.save(state, best_filepath)
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--id', type=str, default='ADMLP', help='Unique experiment identifier.')
	parser.add_argument('--epochs', type=int, default=60, help='Number of train epochs.')
	parser.add_argument('--val_every', type=int, default=3, help='Validation frequency (epochs).')
	parser.add_argument('--batch_size', type=int, default=3200, help='Batch size')
	parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
	parser.add_argument('--gpus', type=int, default=1, help='number of gpus')
	parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
	parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')

	args = parser.parse_args()
	args.logdir = os.path.join(args.logdir, args.id)

	# Config
	config = GlobalConfig()

	# Data
	train_set = CARLA_Data(data_path=config.train_data)
	print(f"train set samples: {len(train_set.x)}")
	val_set = CARLA_Data(data_path=config.val_data)
	print(f"validation set samples: {len(val_set.x)}")

	dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
	dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=16)

	ADMLP_model = ADMLP_planner(config)

	# OPTIMIZER
	optimizer = jt.optim.AdamW(ADMLP_model.parameters(), lr=4e-6, weight_decay=1e-2)
	lr_scheduler = jt.lr_scheduler.MultiStepLR(optimizer,[2,4],gamma=0.2)

	# DEVICE
	device = 'cuda' if jt.has_cuda else 'cpu'
	jt.flags.use_cuda = jt.has_cuda

	best_val_loss = float('inf')

	for epoch in range(args.epochs):
		train(ADMLP_model, dataloader_train, optimizer, config, 'cuda', epoch)
		lr_scheduler.step()
		if ((epoch+1) % args.val_every) == 0:
			# TODO: validation
			val_loss = validate(ADMLP_model, dataloader_val, config, 'cuda', epoch)
			print(f'Epoch {epoch+1}, val_loss: {val_loss}')
			is_best = val_loss < best_val_loss
			best_val_loss = min(val_loss, best_val_loss)
			save_checkpoint({'epoch': (epoch + 1), 'state_dict': ADMLP_model.state_dict()}, is_best, checkpoint_dir=args.logdir, filename=f'checkpoint_{(epoch + 1)}.pth')
	