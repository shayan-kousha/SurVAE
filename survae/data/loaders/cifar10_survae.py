from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import CIFAR10
from survae.data.loaders import root

class Quantize():
    '''
    Assumes input takes values in {0,1,...255}/255, i.e. in [0,1].
    Note: This corresponds to the output of ToTensor().
    '''

    def __init__(self, num_bits=8):
        self.num_bits = num_bits

    def __call__(self, image):
        image = image * 255 # [0, 1] -> [0, 255]
        if self.num_bits != 8:
            image = torch.floor(image / 2 ** (8 - self.num_bits)) # [0, 255] -> [0, 2**num_bits - 1]
        return image

class UnsupervisedCIFAR10(CIFAR10):
    def __init__(self, root=root, train=True, transform=None, download=False):
        super(UnsupervisedCIFAR10, self).__init__(root,
                                                  train=train,
                                                  transform=transform,
                                                  download=download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return super(UnsupervisedCIFAR10, self).__getitem__(index)[0]

class PresplitLoader():

    @property
    def num_splits(self):
        return len(self.splits)

    def get_data_loader(self, split, batch_size, shuffle=True, pin_memory=True, num_workers=4):
        return DataLoader(getattr(self, split), batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)

    def get_data_loaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=4):
        data_loaders = [self.get_data_loader(split=split,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             pin_memory=pin_memory,
                                             num_workers=num_workers) for split in self.splits]
        return data_loaders


class TrainTestLoader(PresplitLoader):
    splits = ['train', 'test']

class CIFAR10SURVAE(TrainTestLoader):
    '''
    The CIFAR10 dataset of (Krizhevsky & Hinton, 2009):
    https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    '''

    def __init__(self, root=root, download=True, num_bits=8, pil_transforms=[]):

        self.root = root

        # Define transformations
        trans_train = pil_transforms + [ToTensor(), Quantize(num_bits)]
        trans_test = pil_transforms + [ToTensor(), Quantize(num_bits)]

        # Load data
        self.train = UnsupervisedCIFAR10(root, train=True, transform=Compose(trans_train), download=download)
        self.test = UnsupervisedCIFAR10(root, train=False, transform=Compose(trans_test))