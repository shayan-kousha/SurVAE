from survae.data.loaders import *
from torchvision.transforms.functional import resize
from torch.utils.data import Dataset
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

    def __init__(self, size=(32, 32), interpolation='bicubic'):
        self.size = size
        self.interpolation = str2interpolation.get(interpolation, None)

        if self.interpolation is None:
        	raise ValueError('Interpolation mode not recognized. Use one of these options: \'nearest\', \'lanczos\', \'bilinear\', \'bicubic\', \'box\', \'hamming\'')

    def __call__(self, image):
        return resize(image, size=self.size, interpolation=self.interpolation)


class CIFAR10Downsampled(Dataset):
    def __init__(self, highres_size=(32, 32), lowres_size=(16, 16), interpolation='bicubic', mode='train', root=root, download=True, num_bits=8, pil_transforms=[], train_pil_transforms=[]):
    	highres_resize = Resize(size=highres_size, interpolation=interpolation)
    	lowres_resize = Resize(size=lowres_size, interpolation=interpolation)

    	if mode == 'train':
    		highres_trans_train = train_pil_transforms + [highres_resize] + pil_transforms + [ToTensor(), Quantize(num_bits)]
    		lowres_trans_train = train_pil_transforms + [lowres_resize] + pil_transforms + [ToTensor(), Quantize(num_bits)]

    		self.highres = UnsupervisedCIFAR10(root, train=True, transform=Compose(highres_trans_train), download=download)
    		self.lowres = UnsupervisedCIFAR10(root, train=True, transform=Compose(lowres_trans_train), download=download)

    	elif mode == 'test':
    		highres_trans_test = [highres_resize] + pil_transforms + [ToTensor(), Quantize(num_bits)]
    		lowres_trans_test = [lowres_resize] + pil_transforms + [ToTensor(), Quantize(num_bits)]

    		self.highres = UnsupervisedCIFAR10(root, train=False, transform=Compose(highres_trans_test), download=download)
    		self.lowres = UnsupervisedCIFAR10(root, train=False, transform=Compose(lowres_trans_test), download=download)

    def __len__(self):
    	return self.highres.shape[0]

    def __getitem__(self, idx):
    	return (self.highres[idx], self.lowres[idx])
        

def CIFAR10_resized(size=(32, 32), interpolation='bicubic', train_pil_transforms=[]):
	cifar = CIFAR10SURVAE(pil_transforms=[Resize(size=size, interpolation=interpolation)], train_pil_transforms=train_pil_transforms)
	return cifar
