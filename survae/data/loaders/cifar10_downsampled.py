from survae.data.loaders import CIFAR10SURVAE
from torchvision.transforms.functional import resize
from PIL import Image

str2interpolation = {
	'nearest' : Image.NEAREST,
	'lanczos' : Image.LANCZOS,
	'bilinear' : Image.BILINEAR,
	'bicubic' : Image.BICUBIC,
	'box' : Image.BOX,
	'hamming' : Image.HAMMING,
}

class Resize():

    def __init__(self, size=(32, 32), interpolation='nearest'):
        self.size = size
        self.interpolation = str2interpolation.get(interpolation, None)

        if self.interpolation is None:
        	raise ValueError('Interpolation mode not recognized. Use one of these options: \'nearest\', \'lanczos\', \'bilinear\', \'bicubic\', \'box\', \'hamming\'')

    def __call__(self, image):
        return resize(image, size=self.size, interpolation=self.interpolation)


def CIFAR10_resized(size=(32, 32), interpolation='bicubic'):
	cifar = CIFAR10SURVAE(pil_transforms=[Resize(size=size, interpolation=interpolation)])
	return cifar
