import tensorlayer as tl
import tensorflow as tf
import numpy as np
import warnings
from tensorlayer.layers import (Input, DeConv2d, BatchNorm)
import os
import matplotlib.pyplot as plt
from dci import DCI

warnings.filterwarnings('ignore')

import collections
Hyperparams = collections.namedtuple('Hyperarams', 'base_lr batch_size num_epochs decay_step decay_rate staleness num_samples_factor num_outputs')
Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None, None)

def mnist_save_img(img, path, name):
	if not os.path.exists(path):
		os.mkdir(path)
	(rows, cols) = img.shape
	fig = plt.figure()
	plt.gray()
	plt.imshow(img)
	fig.savefig(path + name)

def save_img(n, data):
	for i in range(n):
		path = "pic/"
		name = str(i) + ".png"
		mnist_save_img(data[i], path, name)

def get_data(num_data = 128):
	x_train, y_train, x_val, y_val, x_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28))
	row_set = np.random.choice(x_train.shape[0], num_data)
	data = x_train[row_set, :].reshape((num_data, 28, 28, 1))
	return data

def get_model(input_shape):
	ni = Input(shape=input_shape)
	nn = DeConv2d(1024, (1, 1), (1, 1), in_channels=64)(ni)
	nn = BatchNorm(decay=0.99, act=tf.nn.relu)(nn)
	nn = DeConv2d(128, (7, 7), (7, 7), in_channels=1024)(nn)
	nn = BatchNorm(decay=0.99, act=tf.nn.relu)(nn)
	nn = DeConv2d(64, (4, 4), (2, 2), in_channels=128)(nn)
	nn = BatchNorm(decay=0.99, act=tf.nn.relu)(nn)
	nn = DeConv2d(1, (4, 4), (2, 2), in_channels=64, act=tf.nn.sigmoid)(nn)
	return tl.models.Model(inputs=ni, outputs=nn, name='cnn')

class IMLE():
	def __init__(self, z_dim):
		self.z_dim = z_dim
		self.dci_db = None

	def train(self, data_np, hyperparams, shuffle_data=True):

		batch_size = hyperparams.batch_size
		num_batches = data_np.shape[0] // batch_size
		num_samples = num_batches * hyperparams.num_samples_factor
		num_outputs = hyperparams.num_outputs

		if shuffle_data:
			data_ordering = np.random.permutation(data_np.shape[0])
			data_np = data_np[data_ordering]

		data_flat_np = np.reshape(data_np, (data_np.shape[0], np.prod(data_np.shape[1:])))

		input_shape = [64, 1, 1, self.z_dim]
		net = get_model(input_shape)
		train_weights = net.trainable_weights

		if self.dci_db is None:
			self.dci_db = DCI(np.prod(data_np.shape[1:]), num_comp_indices=2, num_simp_indices=7)

		for epoch in range(hyperparams.num_epochs):

			if epoch % hyperparams.decay_step == 0:
				lr = hyperparams.base_lr * hyperparams.decay_rate ** (epoch // hyperparams.decay_step)
				optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
				# Optimizer API changes. betas->beta_1, beta_2 & remove weight_decay=1e-5

			if epoch % hyperparams.staleness == 0:
				net.eval() #declare now in eval mode.
				z_np = np.empty((num_samples * batch_size, 1, 1, self.z_dim))
				samples_np = np.empty((num_samples * batch_size,) + data_np.shape[1:])
				for i in range(num_samples):
					z = tf.random.normal([batch_size, 1, 1, self.z_dim])
					samples = net(z)
					z_np[i * batch_size:(i + 1) * batch_size] = z.numpy()
					samples_np[i * batch_size:(i + 1) * batch_size] = samples.numpy()

				samples_flat_np = np.reshape(samples_np, (samples_np.shape[0], np.prod(samples_np.shape[1:])))

				self.dci_db.reset()
				self.dci_db.add(samples_flat_np, num_levels=2, field_of_view=10, prop_to_retrieve=0.002)
				nearest_indices, _ = self.dci_db.query(data_flat_np, num_neighbours=1, field_of_view=20, prop_to_retrieve=0.02)
				nearest_indices = np.array(nearest_indices)[:, 0]

				z_np = z_np[nearest_indices]
				z_np += 0.01 * np.random.randn(*z_np.shape)

				del samples_np, samples_flat_np

			err = 0.
			for i in range(num_batches):
				net.eval()
				cur_z = tf.convert_to_tensor(z_np[i * batch_size:(i + 1) * batch_size], dtype = tf.float32)
				cur_data = tf.convert_to_tensor(data_np[i * batch_size:(i + 1) * batch_size], dtype = tf.float32)
				cur_samples = net(cur_z)
				_loss = tl.cost.mean_squared_error(cur_data, cur_samples, is_mean=False)
				err += _loss

			print("Epoch %d: Error: %f" % (epoch, err / num_batches))

			for i in range(num_batches):
				net.train()
				with tf.GradientTape() as tape:
					cur_z = tf.convert_to_tensor(z_np[i * batch_size:(i + 1) * batch_size], dtype = tf.float32)
					cur_data = tf.convert_to_tensor(data_np[i * batch_size:(i + 1) * batch_size], dtype = tf.float32)
					cur_samples = net(cur_z)
					_loss = tl.cost.mean_squared_error(cur_data, cur_samples, is_mean=False)
				grad = tape.gradient(_loss, train_weights)
				optimizer.apply_gradients(zip(grad, train_weights))

			#debug. extreamly lower the speed. but worth doing to visualize the process. (following 4 lines can be deleted directly)
			#cur_z = tf.convert_to_tensor(z_np[0: batch_size], dtype = tf.float32)
			#cur_samples = net(cur_z)
			#pics = np.reshape(cur_samples.numpy(), (batch_size, 28, 28))
			#save_img(1, pics)
			cur_z = tf.random.normal([batch_size, 1, 1, self.z_dim])
			cur_samples = net(cur_z)
			pics = np.reshape(cur_samples.numpy(), (batch_size, 28, 28))
			save_img(2, pics)

		#cur_z = tf.convert_to_tensor(z_np[0: batch_size], dtype = tf.float32)
		cur_z = tf.random.normal([batch_size, 1, 1, self.z_dim])
		cur_samples = net(cur_z)
		pics = np.reshape(cur_samples.numpy(), (batch_size, 28, 28))

		save_img(batch_size, pics)

# train_data is of shape N x C x H x W, where N is the number of examples, C is the number of channels, H is the height and W is the width
def main():
	z_dim = 64
	train_data = get_data()

	imle = IMLE(z_dim)

	imle.train(train_data, Hyperparams(base_lr=1e-3, batch_size=z_dim, num_epochs=100, decay_step=25, decay_rate=0.5, staleness=5, num_samples_factor=10, num_outputs = 5))
    
if __name__ == '__main__':
	main()