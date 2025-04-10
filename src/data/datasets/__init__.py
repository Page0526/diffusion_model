from .cifar10 import Cifar10Dataset
from .fashion import FashionDataset
from .mnist import MnistDataset
from .mri import MRIDataset

__datasets = {
    'cifar10': Cifar10Dataset,
    'fashion': FashionDataset,
    'mnist': MnistDataset,
    'mri': MRIDataset,
}


def init_dataset(name, **kwargs):
    """Initializes a dataset."""
    avai_datasets = list(__datasets.keys())
    if name not in avai_datasets:
        raise ValueError('Invalid dataset name. Received "{}", '
                         'but expected to be one of {}'.format(
                             name, avai_datasets))
    return __datasets[name](**kwargs)