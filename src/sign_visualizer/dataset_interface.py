import re
import torchvision
from torchvision import transforms

from src.datasets.data_loader import get_datasets


class ContinualDatasetConfig:
    def __init__(self, dataset_name, num_tasks, **kwargs) -> None:
        _get_before_underscore = lambda str: str.split("_")[0]
        _get_digits = lambda str: int(re.sub(r"\D", "", str))

        self.num_tasks = num_tasks
        self.dataset = dataset_name
        self.validation = kwargs.get("nc_first_task", 0.0)
        self.nc_first_task = kwargs.get("nc_first_task", None)
        self.nc_per_task = kwargs.get("nc_per_task", None)
        transforms = self._setup_transforms()
        self.transforms = kwargs.get("transforms", transforms)
        self.path = kwargs.get("path", "data/" + _get_before_underscore(dataset_name))
        self.class_order = kwargs.get(
            "class_order", [i for i in range(_get_digits(dataset_name))]
        )

    def _setup_transforms(self):
        mean, std = (0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
        return transform


class ContinualDataset:
    def __init__(self, config) -> None:
        self.config = config
        self._setup()

    def _setup(self):
        trn_dset, val_dset, tst_dset, taskcla = get_datasets(
            dataset=self.config.dataset,
            num_tasks=self.config.num_tasks,
            validation=self.config.validation,
            nc_first_task=self.config.nc_first_task,
            nc_per_task=self.config.nc_per_task,
            path=self.config.path,
            trn_transform=self.config.transforms,
            tst_transform=self.config.transforms,
            class_order=self.config.class_order,
        )

        self.name_dataset = {
            "train": trn_dset,
            "val": val_dset,
            "test": tst_dset,
        }

    def __getitem__(self, args):
        task, type = args
        return self.name_dataset[type][task]
