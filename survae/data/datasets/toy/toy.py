import numpy as np

class CheckerBoard:
	def __init__(self, num_points):
		self.num_points = num_points

	def get_data(self):
		x1 = np.random.rand(self.num_points) * 4 - 2
		x2_ = np.random.rand(self.num_points) - np.random.randint(0, 2, [self.num_points]) * 2
		x2 = x2_ + np.floor(x1) % 2
		data = 2 * np.stack([x1, x2]).T
		return data

class Corners:
	def __init__(self, num_points):
		self.num_points = num_points

	def get_data(self):
		n = int(self.num_points / 4)
		x1_1 = np.random.rand(n) + 1
		x2_1 = np.random.rand(n) * 0.3 + 1

		x1_2 = np.random.rand(n) * 0.3 + 1
		x2_2 = np.random.rand(n) + 1

		x1_3 = np.random.rand(n) - 2
		x2_3 = np.random.rand(n) * 0.3 + 1

		x1_4 = np.random.rand(n) * (-0.3) - 1
		x2_4 = np.random.rand(n) + 1

		x1_5 = np.random.rand(n) - 2
		x2_5 = np.random.rand(n) * (-0.3) - 1

		x1_6 = np.random.rand(n) * (-0.3) - 1
		x2_6 = np.random.rand(n) - 2

		x1_7 = np.random.rand(n) + 1
		x2_7 = np.random.rand(n) * (-0.3) - 1

		x1_8 = np.random.rand(n) * 0.3 + 1
		x2_8 = np.random.rand(n) - 2


		data = np.stack([np.concatenate((x1_1, x1_2, x1_3, x1_4, x1_5, x1_6, x1_7, x1_8)), np.concatenate((x2_1, x2_2, x2_3, x2_4, x2_5, x2_6, x2_7, x2_8))]).T
		return data

class FourCirclesDataset:
	def __init__(self, num_points):
		self.num_points = num_points

		if num_points % 4 != 0:
			raise ValueError('Number of data points must be a multiple of four')


	@staticmethod
	def create_circle(num_per_circle, std=0.1):
		u = np.random.rand(num_per_circle)
		x1 = np.cos(2 * np.pi * u)
		x2 = np.sin(2 * np.pi * u)
		data = 2 * np.stack((x1, x2)).T
		data += std * np.random.normal(size=data.shape)
		return data

	def get_data(self):
		num_per_circle = self.num_points // 4
		centers = [
			[-1, -1],
			[-1, 1],
			[1, -1],
			[1, 1]
		]
		data = np.concatenate(
			[self.create_circle(num_per_circle) - np.array(center)
			for center in centers]
		)

		return data

class GaussianDataset:
	def __init__(self, num_points):
		self.num_points = num_points

	def get_data(self):
		x1 = np.random.normal(size=self.num_points)
		x2 = 0.5 * np.random.normal(size=self.num_points)
		data = np.stack((x1, x2)).T

		return data

class EightGaussiansDataset:
	'''Adapted from https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py'''
	def __init__(self, num_points):
		self.num_points = num_points

	def get_data(self):
		scale = 4.
		bias = np.pi / 8
		step = np.pi / 4
		centers = [(np.cos(bias + 0*step), np.sin(bias + 0*step)),
					(np.cos(bias + 1*step), np.sin(bias + 1*step)),
					(np.cos(bias + 2*step), np.sin(bias + 2*step)),
					(np.cos(bias + 3*step), np.sin(bias + 3*step)),
					(np.cos(bias + 4*step), np.sin(bias + 4*step)),
					(np.cos(bias + 5*step), np.sin(bias + 5*step)),
					(np.cos(bias + 6*step), np.sin(bias + 6*step)),
					(np.cos(bias + 7*step), np.sin(bias + 7*step))]
		centers = [(scale * x, scale * y) for x, y in centers]

		dataset = []
		for i in range(self.num_points):
			point = np.random.randn(2) * 0.5
			idx = np.random.randint(8)
			center = centers[idx]
			point[0] += center[0]
			point[1] += center[1]
			dataset.append(point)
		dataset = np.array(dataset, dtype="float32")
		dataset /= 1.414

		return dataset

def test():
	num_points = 1000
	gaussian = EightGaussiansDataset(num_points)

	data = gaussian.get_data()
	from matplotlib import pyplot as plt
	plt.scatter(data[:, 0], data[:, 1], cmap="bwr", alpha=0.5, )
	plt.savefig('./unit_test/US1.03/samples/test/test-{}.png'.format(0))


if __name__ == '__main__':
	test()