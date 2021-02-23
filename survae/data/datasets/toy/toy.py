import numpy as np

class CheckerBoard:
	def __init__(self, num_points):
		self.num_points = num_points

	def get_data(self):
		x1 = np.random.rand(self.num_points) * 4 - 2
		x2_ = np.random.rand(self.num_points) - np.random.randint(0, 2, [self.num_points]) * 2
		x2 = x2_ + np.floor(x1) % 2
		data = np.stack([x1, x2]).T
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

def test():
	num_points = 1000
	checkerboard = CheckerBoard(num_points)

	data = checkerboard.get_data()
	from matplotlib import pyplot as plt
	plt.scatter(data[:, 0], data[:, 1], cmap="bwr", alpha=0.5, )
	plt.show()


if __name__ == '__main__':
	test()