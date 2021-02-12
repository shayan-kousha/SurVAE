import numpy as np

class CheckerBoard:
	def __init__(self, num_points):
		self.num_points = num_points

	def get_data():
		x1 = np.random.rand(self.num_points) * 4 - 2
		x2_ = np.random.rand(self.num_points) - np.random.randint(0, 2, [self.num_points]) * 2
		x2 = x2_ + np.floor(x1) % 2
		data = np.stack([x1, x2]).T
		return data

def test():
	num_points = 1000
	checkerboard = CheckerBoard(num_points)

	data = checkerboard.get_data()
	from matplotlib import pyplot as plt
	plt.scatter(data[:, 0], data[:, 1], cmap="bwr", alpha=0.5, )
	plt.show()

if __name__ == '__main__':
	test()