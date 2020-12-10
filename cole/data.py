import torchvision
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np

base_path = "../data"


class XYDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None, **kwargs):
        super(XYDataset, self).__init__()
        self.x, self.y = x, y
        self.transform = transform

        for name, value in kwargs.items():
            setattr(self, name, value)

    def get_labels(self):
        return set(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        if self.transform is not None:
            x = self.transform(self.x[item])
        else:
            x = self.x[item]
        return x, self.y[item]


# TODO add support for adaptive lenght sampler
class CLDataLoader:
    def __init__(self, datasets_per_task, bs=10, train=True, sampler=None):
        self.bs = bs
        self.datasets = datasets_per_task
        if sampler is None:
            self.data_loaders = [torch.utils.data.DataLoader(x, self.bs, shuffle=train) for x in self.datasets]
        else:
            self.data_loaders = [torch.utils.data.DataLoader(x, batch_sampler=sampler) for x in self.datasets]

    def __getitem__(self, item):
        return self.data_loaders[item]

    def __len__(self):
        return len(self.data_loaders)


# TODO: bad name
class AdaptiveLengthSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, size, batch_size=10, shuffle=True):
        super(AdaptiveLengthSampler, self).__init__(data_source)
        self.bs = batch_size
        self.size = size if len(data_source) > size else len(data_source)

        if shuffle:
            self.order = torch.randperm(len(data_source)).tolist()[:self.size]
        else:
            self.order = range(self.size)

    def __iter__(self):
        batch = []
        for idx in self.order:
            batch.append(idx)
            if len(batch) == self.bs:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return int(np.ceil(self.size / self.bs))


# TODO: remove args
# TODO: make possible to give int in tasks instead of list
def get_split_mnist(args, joint=False, tasks=None):

    if tasks is None:
        tasks = [i for i in range(1, 6)]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train = torchvision.datasets.MNIST(base_path,  train=True, download=True)
    test = torchvision.datasets.MNIST(base_path, train=False, download=True)

    return make_split_dataset(train, test, joint, tasks, transform)


# TODO: remove args
def get_split_cifar10(args, joint=False, tasks=None):

    if tasks is None:
        tasks = [i for i in range(5)]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train = torchvision.datasets.CIFAR10(base_path,  train=True, download=True)
    test = torchvision.datasets.CIFAR10(base_path,  train=False, download=True)

    return make_split_dataset(train, test, joint, tasks, transform)


# TODO: let this return an object with train, validate and test members
def make_split_dataset(train, test, joint=False, tasks=None, transform=None):
    train_x, train_y = train.data, train.targets
    test_x, test_y = test.data, test.targets

    # Sort all samples based on targets
    out_train = [(x, y) for (x, y) in sorted(zip(train_x, train_y), key=lambda v: v[1])]
    out_test = [(x, y) for (x, y) in sorted(zip(test_x, test_y), key=lambda v: v[1])]

    # Create tensor of samples and targets
    train_x, train_y = [np.stack([elem[i] for elem in out_train]) for i in [0, 1]]
    test_x, test_y = [np.stack([elem[i] for elem in out_test]) for i in [0, 1]]

    # Get max idx of each target label
    train_idx = [((train_y + i) % 10).argmax() for i in range(10)]
    train_idx = sorted(train_idx) + [len(train_x)]

    test_idx = [((test_y + i) % 10).argmax() for i in range(10)]
    test_idx = sorted(test_idx) + [len(test_x)]

    labels_per_task = 2
    train_ds, test_ds = [], []
    for i in tasks:
        task_st_label = (i - 1) * 2
        tr_s, tr_e = train_idx[task_st_label], train_idx[task_st_label + labels_per_task]
        te_s, te_e = test_idx[task_st_label], test_idx[task_st_label + labels_per_task]

        train_ds += [(train_x[tr_s:tr_e], train_y[tr_s:tr_e])]
        test_ds += [(test_x[te_s:te_e], test_y[te_s:te_e])]

    if joint:
        train_ds = [(np.concatenate([task_ds[0] for task_ds in train_ds]),
                     np.concatenate([task_ds[1] for task_ds in train_ds]))]
        test_ds = [(np.concatenate([task_ds[0] for task_ds in test_ds]),
                    np.concatenate([task_ds[1] for task_ds in test_ds]))]

    train_ds, val_ds = make_valid_from_train(train_ds)

    train_ds = [XYDataset(x[0], x[1], transform=transform) for x in train_ds]
    val_ds = [XYDataset(x[0], x[1], transform=transform) for x in val_ds]
    test_ds = [XYDataset(x[0], x[1], transform=transform) for x in test_ds]

    return train_ds, val_ds, test_ds


def make_valid_from_train(dataset, cut=0.95):
    tr_ds, val_ds = [], []
    for task_ds in dataset:
        x_t, y_t = task_ds

        # shuffle before splitting
        perm = torch.randperm(len(x_t))
        x_t, y_t = x_t[perm], y_t[perm]

        split = int(len(x_t) * cut)
        x_tr, y_tr = x_t[:split], y_t[:split]
        x_val, y_val = x_t[split:], y_t[split:]

        tr_ds += [(x_tr, y_tr)]
        val_ds += [(x_val, y_val)]

    return tr_ds, val_ds
