import os
from PIL import Image
import numpy as np
import jittor as jt
from jittor.dataset.dataset import Dataset
from jittor import transform as T
from collections.abc import Mapping, Sequence

from TCP.augment import hard as augmenter
import threading
from tqdm import tqdm
import matplotlib.pyplot as plt

class CARLA_Data(Dataset):

	def __init__(self, root, data_path, img_aug=False):	
		super().__init__()
		self.root = root
		self.img_aug = img_aug
		self._batch_read_number = 0

		self.front_img = []
		self.x = []
		self.y = []
		self.command = []
		self.target_command = []
		self.target_gps = []
		self.theta = []
		self.speed = []


		self.value = []
		self.feature = []
		self.action = []
		self.action_index = []

		self.future_x = []
		self.future_y = []
		self.future_theta = []

		self.future_feature = []
		self.future_action = []
		self.future_action_index = []
		self.future_only_ap_brake = []

		self.x_command = []
		self.y_command = []
		self.command = []
		self.only_ap_brake = []

		# for sub_root in data_folders:
		print(f'load data from {data_path}')
		data = np.load(data_path, allow_pickle=True).item()

		#self.total_len = len(data['front_img'])

		self.load_to_memory = False
		if self.load_to_memory:
			print(f'load data to memory begin')
			self.progress_bar = tqdm(total=len(data['front_img']), desc="Loading Images")
			threads = []
			self.front_img_dict = {}
			self.front_left_img_dict = {}
			self.front_right_img_dict = {}
			self.lock = threading.Lock()
			for img_path in data['front_img']:
				while threading.active_count() >= 64:
					for t in threads:
						t.join()
					threads = [t for t in threads if t.is_alive()]
				thread = threading.Thread(target=self.load_image, args=(img_path))
				thread.start()
				threads.append(thread)

			for t in threads:
				t.join()
			self.progress_bar.close()
			print(f'load data to memory end')

		self.x_command += data['x_target']
		self.y_command += data['y_target']
		self.command += data['target_command']

		self.front_img += data['front_img']
		self.x += data['input_x']
		self.y += data['input_y']
		self.theta += data['input_theta']
		self.speed += data['speed']

		self.future_x += data['future_x']
		self.future_y += data['future_y']
		self.future_theta += data['future_theta']

		self.future_feature += data['future_feature']
		self.future_action += data['future_action']
		self.future_action_index += data['future_action_index']
		self.future_only_ap_brake += data['future_only_ap_brake']

		self.value += data['value']
		self.feature += data['feature']
		self.action += data['action']
		self.action_index += data['action_index']
		self.only_ap_brake += data['only_ap_brake']
		self._im_transform = T.Compose([T.ToTensor(), T.ImageNormalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

		# must set
		self.set_attrs(total_len=len(self.front_img))

	def load_image(self, img_path):
		front_img = np.array(Image.open(img_path))
		front_left_img = np.array(Image.open(img_path.replace('rgb_front', 'rgb_front_left')))
		front_right_img = np.array(Image.open(img_path.replace('rgb_front', 'rgb_front_right')))
		if img_path not in self.front_img_dict.keys():
			self.front_img_dict[img_path] = front_img
			self.front_left_img_dict[img_path.replace('rgb_front', 'rgb_front_left')] = front_left_img
			self.front_right_img_dict[img_path.replace('rgb_front', 'rgb_front_right')] = front_right_img
		with self.lock:
			self.front_img_dict[img_path] = front_img
			self.front_left_img_dict[img_path.replace('rgb_front', 'rgb_front_left')] = front_left_img
			self.front_right_img_dict[img_path.replace('rgb_front', 'rgb_front_right')] = front_right_img
			self.progress_bar.update(1) 


	def __getitem__(self, index):
		"""Returns the item at index idx. """
		data = dict()
		img_path = self.front_img[index][0]

		if not os.path.exists(img_path):
			img_path = img_path.replace('v2', 'v2-216')
		if self.load_to_memory:
			front_img = self.front_img_dict[img_path]
			front_left_img = self.front_left_img_dict[img_path.replace('rgb_front', 'rgb_front_left')]
			front_right_img = self.front_right_img_dict[img_path.replace('rgb_front', 'rgb_front_right')]
		else:
			front_img = np.array(Image.open(img_path))
			front_left_img = np.array(Image.open(img_path.replace('rgb_front', 'rgb_front_left')))
			front_right_img = np.array(Image.open(img_path.replace('rgb_front', 'rgb_front_right')))

		front_img = front_img[:, 200:1400, :]
		front_left_img = front_left_img[:, :1400, :]
		front_right_img = front_right_img[:, 200:, :]

		front_img = np.concatenate((front_left_img, front_img, front_right_img), axis=1)
		front_img = jt.unsqueeze(jt.Var(front_img).permute(2, 0, 1), 0).float()
		front_img = jt.nn.interpolate(front_img, size=(256, 900), mode='bilinear', align_corners=False)
		#front_img = front_img.squeeze(0).permute(1, 2, 0).byte().numpy()
		front_img = jt.squeeze(front_img, 0).permute(1, 2, 0).uint8().numpy()

		debug_concat_img = False
		if debug_concat_img:
			import time
			image = Image.fromarray(front_img)
			image.save(f'./{time.time()}.jpg')

		
		if self.img_aug:
			front_img = augmenter(self._batch_read_number).augment_image(front_img)
			# Transpose image format to (C, H, W)
			front_img = np.transpose(front_img, (2, 0, 1))
			front_img = self._im_transform(front_img)
			data['front_img'] = front_img
		else:
			front_img = np.transpose(np.array(front_img), (2, 0, 1))
			front_img = self._im_transform(front_img)
			data['front_img'] = front_img

		# fix for theta=nan in some measurements
		if np.isnan(self.theta[index][0]):
			self.theta[index][0] = 0.

		ego_x = self.x[index][0]
		ego_y = self.y[index][0]
		ego_theta = self.theta[index][0] - np.pi/2 # compass on left hand (0, -1)

		waypoints = []
		for i in range(4):
			R = np.array([
				[np.cos(ego_theta), np.sin(ego_theta)],
				[-np.sin(ego_theta),  np.cos(ego_theta)]
			])
			local_command_point = np.array([self.future_x[index][i]-ego_x, self.future_y[index][i]-ego_y])
			local_command_point = R.dot(local_command_point) # left hand
			waypoints.append([local_command_point[0], local_command_point[1]])

		data['waypoints'] = waypoints

		data['action'] = self.action[index]
		data['action_index'] = self.action_index[index]

		data['future_action_index'] = self.future_action_index[index]
		data['future_feature'] = self.future_feature[index]
		
		R = np.array([
			[np.cos(ego_theta), np.sin(ego_theta)],
			[-np.sin(ego_theta),  np.cos(ego_theta)]
		])

		local_command_point_aim = np.array([(self.x_command[index]-ego_x), self.y_command[index]-ego_y])
		local_command_point_aim = R.dot(local_command_point_aim)
		data['target_point'] = local_command_point_aim[:2]

		debug_plot_local = debug_plot_world = False
		if debug_plot_world:
			plt.figure(figsize=(10, 10))
			x = self.x[index] + self.future_x[index]
			y = self.y[index] + self.future_y[index]
			plt.scatter(x, y, color='red', zorder=5)
			for i, (px, py) in enumerate(zip(x,y)):
				plt.text(px, py, f'P{i}', fontsize=12, ha='right' if i % 2 == 0 else 'left', va='top' if i % 2 == 0 else 'bottom')
			plt.plot(x, y, 'b--')
			plt.title(img_path.split('/')[-4:])
			plt.xlabel('X coordinate')
			plt.ylabel('Y coordinate')
			plt.grid(True)
			plt.axis('equal') 
			plt.savefig(f"{index}_world.png")

		if debug_plot_local:
			plt.figure(figsize=(10, 10))
			x, y = zip(*data['waypoints'])
			x = (0, ) + x
			y = (0, ) + y
			plt.scatter(x, y, color='red', zorder=5)
			for i, (px, py) in enumerate(zip(x,y)):
				plt.text(px, py, f'P{i}', fontsize=12, ha='right' if i % 2 == 0 else 'left', va='top' if i % 2 == 0 else 'bottom')
			plt.plot(x, y, 'b--')
			plt.title(img_path.split('/')[-4:])
			plt.xlabel('X coordinate')
			plt.ylabel('Y coordinate')
			plt.grid(True)
			plt.axis('equal') 
			plt.savefig(f"{index}_local.png")

		data['speed'] = self.speed[index]
		data['feature'] = self.feature[index]
		data['value'] = self.value[index]
		command = self.command[index]

		# VOID = -1
		# LEFT = 1
		# RIGHT = 2
		# STRAIGHT = 3
		# LANEFOLLOW = 4
		# CHANGELANELEFT = 5
		# CHANGELANERIGHT = 6
		if command < 0:
			command = 4
		command -= 1
		assert command in [0, 1, 2, 3, 4, 5]
		cmd_one_hot = [0] * 6
		cmd_one_hot[command] = 1
		#data['target_command'] = jt.Var(cmd_one_hot)	
		# MARK: return numpy array instead of jt.Var
		data['target_command'] = cmd_one_hot

		self._batch_read_number += 1

		return data

	def collate_batch(self, batch):
		r"""Puts each data field into a tensor with outer dimension batch size"""
		real_size = len(batch)
		elem = batch[0]
		elem_type = type(elem)
		
		if isinstance(elem, jt.Var):
			temp_data = jt.stack([data for data in batch], 0)
			return temp_data
		if elem_type is np.ndarray:
			temp_data = np.stack([data for data in batch], 0)
			return temp_data
		elif np.issubdtype(elem_type, np.integer):
			return np.int32(batch)
		elif isinstance(elem, int):
			return np.int32(batch)
		elif isinstance(elem, np.float32):  # add np.float32 type
			return np.float32(batch)
		elif isinstance(elem, float):
			return np.float32(batch) 
		elif isinstance(elem, str):
			return batch
		elif isinstance(elem, Mapping):
			return {key: self.collate_batch([d[key] for d in batch]) for key in elem}
		elif isinstance(elem, tuple):
			transposed = zip(*batch)
			return tuple(self.collate_batch(samples) for samples in transposed)
		elif isinstance(elem, Sequence):
			transposed = zip(*batch)
			return [self.collate_batch(samples) for samples in transposed]
		elif isinstance(elem, Image.Image):
			temp_data = np.stack([np.array(data) for data in batch], 0)
			return temp_data
		else:
			raise TypeError(f"Not support type <{elem_type.__name__}>")


def scale_and_crop_image(image, scale=1, crop_w=256, crop_h=256):
	"""
	Scale and crop a PIL image
	"""
	(width, height) = (int(image.width // scale), int(image.height // scale))
	im_resized = image.resize((width, height))
	start_x = height//2 - crop_h//2
	start_y = width//2 - crop_w//2
	cropped_image = im_resized.crop((start_y, start_x, start_y+crop_w, start_x+crop_h))

	# cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
	# cropped_image = np.transpose(cropped_image, (2,0,1))
	return cropped_image


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
	"""
	Build a rotation matrix and take the dot product.
	"""
	# z value to 1 for rotation
	xy1 = xyz.copy()
	xy1[:,2] = 1

	c, s = np.cos(r1), np.sin(r1)
	r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

	# np.dot converts to a matrix, so we explicitly change it back to an array
	world = np.asarray(r1_to_world @ xy1.T)

	c, s = np.cos(r2), np.sin(r2)
	r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
	world_to_r2 = np.linalg.inv(r2_to_world)

	out = np.asarray(world_to_r2 @ world).T
	
	# reset z-coordinate
	out[:,2] = xyz[:,2]

	return out

def rot_to_mat(roll, pitch, yaw):
	roll = np.deg2rad(roll)
	pitch = np.deg2rad(pitch)
	yaw = np.deg2rad(yaw)

	yaw_matrix = np.array([
		[np.cos(yaw), -np.sin(yaw), 0],
		[np.sin(yaw), np.cos(yaw), 0],
		[0, 0, 1]
	])
	pitch_matrix = np.array([
		[np.cos(pitch), 0, -np.sin(pitch)],
		[0, 1, 0],
		[np.sin(pitch), 0, np.cos(pitch)]
	])
	roll_matrix = np.array([
		[1, 0, 0],
		[0, np.cos(roll), np.sin(roll)],
		[0, -np.sin(roll), np.cos(roll)]
	])

	rotation_matrix = yaw_matrix.dot(pitch_matrix).dot(roll_matrix)
	return rotation_matrix


def vec_global_to_ref(target_vec_in_global, ref_rot_in_global):
	R = rot_to_mat(ref_rot_in_global['roll'], ref_rot_in_global['pitch'], ref_rot_in_global['yaw'])
	np_vec_in_global = np.array([[target_vec_in_global[0]],
								 [target_vec_in_global[1]],
								 [target_vec_in_global[2]]])
	np_vec_in_ref = R.T.dot(np_vec_in_global)
	return np_vec_in_ref[:,0]
	