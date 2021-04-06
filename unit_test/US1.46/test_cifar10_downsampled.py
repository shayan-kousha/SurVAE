import sys
sys.path.append(".")

from survae.data.loaders import CIFAR10_resized

cifar = CIFAR10_resized()

print(cifar.train[0].shape, " should be (3, 32, 32)")

cifar = CIFAR10_resized(size=(8, 8))

print(cifar.train[0].shape, " should be (3, 8, 8)")

cifar = CIFAR10_resized(size=(2, 2), interpolation='bilinear')

print(cifar.train[0].shape, " should be (3, 2, 2)")

try:
	cifar = CIFAR10_resized(size=(4, 4), interpolation='sldgkflkjg')
except ValueError:
	print("Successfully raised ValueError for incorrect interpolation mode.")
