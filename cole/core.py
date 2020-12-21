from abc import ABC

import torch
import torch.utils.data
import torchvision
import random
from cole.helper import *

__BASE_DATA_PATH = '../../data'


def set_data_path(path: str):
    global __BASE_DATA_PATH
    __BASE_DATA_PATH = path


class CLDataLoader:
    """
    Sequential dataloader for continual learning tasks.
    """

    def __init__(self, task_datasets, bs=10, shuffle=True, task_size=0):
        """
        :param task_datasets: Iterable datasets
        :param bs: batch size
        :param shuffle: shuffle each task independently
        :param task_size: Size of the task. If size is 0 or higher than samples in task all data is used.
        """
        self.bs = bs
        self.datasets = task_datasets
        self.data_loaders = []

        if task_size < 0:
            raise ValueError(f"Task size should be non-negative. Got {task_size}")

        for task in task_datasets:
            if task_size > len(task) or task_size == 0:
                self.data_loaders.append(torch.utils.data.DataLoader(task, self.bs, shuffle=shuffle))
            else:
                sampler = SizedSampler(task, task_size, batch_size=bs, shuffle=shuffle)
                self.data_loaders.append(torch.utils.data.DataLoader(task, batch_sampler=sampler))

    def __getitem__(self, item):
        return self.data_loaders[item]

    def __len__(self):
        return len(self.data_loaders)


def get_split_mnist(tasks=None, joint=False):
    """
    Get split version of the MNIST dataset.
    :param tasks: int or list with task indices. Task 1 has labels 0 and 1, etc.
    :param joint: Concatenate tasks in joint dataset
    :return: DataSplit object with train_set, test_set en validation members.
    """
    if tasks is None:
        tasks = [i for i in range(1, 6)]
    if type(tasks) is int:
        tasks = [tasks]

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.MNIST(__BASE_DATA_PATH, train=True, download=True)
    test_set = torchvision.datasets.MNIST(__BASE_DATA_PATH, train=False, download=True)

    return make_split_dataset(train_set, test_set, joint, tasks, transform)


def get_split_cifar10(tasks=None, joint=False):
    """
    Get split version of the CIFAR 10 dataset.
    :param tasks: int or list with task indices. Task 1 has labels 0 and 1, etc.
    :param joint: Concatenate tasks in joint dataset
    :return: DataSplit object with train, test en validation members.
    """
    if tasks is None:
        tasks = [i for i in range(5)]
    if type(tasks) is int:
        tasks = [tasks]

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.CIFAR10(__BASE_DATA_PATH, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(__BASE_DATA_PATH, train=False, download=True)

    return make_split_dataset(train_set, test_set, joint, tasks, transform)


class MLP(nn.Module, ABC):
    def __init__(self, nb_classes=10, hid_nodes=400, hid_layers=2, down_sample=1, input_size=28):
        """
        Simple 2-layer MLP with RELU activation
        :param nb_classes: nb of outputs nodes, i.e. classes
        :param hid_nodes: nb of hidden nodes per layer
        :param hid_layers: nb of hidden layers in model
        :param down_sample: down sample data before entering model with a factor down_sample
        :param input_size: data width/height before possible down_sampling
        """
        super(MLP, self).__init__()

        if down_sample < 1:
            raise ValueError(f"down_sample should be 1 or greater, got {down_sample}")
        if hid_layers < 1:
            raise ValueError(f"hid_layers should be 1 or greater, got {hid_layers}")

        self.input_size = (input_size // down_sample) * (input_size // down_sample)
        self.down_sample = nn.MaxPool2d(down_sample) if down_sample > 1 else None

        layers = [nn.Linear(self.input_size, hid_nodes), nn.ReLU(True)]
        for _ in range(1, hid_layers):
            layers.extend([nn.Linear(hid_nodes, hid_nodes), nn.ReLU(True)])
        self.hidden = nn.Sequential(*layers)

        self.output = nn.Linear(hid_nodes, nb_classes)

    def forward(self, x):
        if self.down_sample is not None:
            x = self.down_sample(x)
        x = x.view(-1, self.input_size)
        x = self.hidden(x)
        return self.output(x)


def get_resnet18(nb_classes=10):
    """
    :param nb_classes: nb classes or output nodes
    :return: ResNet18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nb_classes, 20, [3, 32, 32])


def step(model, optimizer, data, target):
    """
    Updates a model a single step, based on the cross entropy loss of the given and target
    """
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()


def test(model, loaders, avg=True, device='cpu'):
    """
    Returns (mean) loss and (mean) accuracy of all loaders in loaders.
    :param loaders: iterable of data loaders
    :param avg: if True, the average across all tasks is returned. The average is calculated with equal weight to all
    tasks, independent of the actual size of the task.
    :return: (int, int) or (arr, arr) if avg is False
    """
    model.eval()

    acc_arr = []
    loss_arr = []

    for loader in loaders:
        loss, acc = test_dataset(model, loader, device)
        acc_arr.append(acc)
        loss_arr.append(loss.item())

    model.train()

    if avg:
        return 100 * np.mean(acc_arr), np.mean(loss_arr)
    else:
        return acc_arr, loss_arr


# TODO: make sampler retriever abstract (?) such that user can implement own.
class Buffer(torch.utils.data.Dataset):

    def __init__(self, size, sampler='reservoir', retriever='random'):
        """
        Buffer class, to be used as ER buffer. Works as a torch dataset.
        :param size: Buffer size
        :param sampler: Sampler to use. Options: 'reservoir' (def) or 'first_in'
        :param retriever: Retriever to use. Options: 'random' (def)
        """
        super(Buffer, self).__init__()
        self.data = {}  # Dictionary, key is label value is x
        self.size = size
        self.sampler = _set_sampler(sampler)(self)
        self.retriever = _set_retriever(retriever)(self)

    def __getitem__(self, item):
        for label in self.data:
            if item >= len(self.data[label]):
                item -= len(self.data[label])
            else:
                return self.data[label][item], label

    def __len__(self):
        return sum([len(self.data[label]) for label in self.data])

    def __repr__(self):
        rep = ""
        for label in self.data:
            rep += f"{label}: {len(self.data[label])} \n"
        rep += f"Total: {len(self)}/{self.size}"
        return rep

    def pop(self, item):
        for label in self.data:
            if item >= len(self.data[label]):
                item -= len(self.data[label])
            else:
                self.data[label].pop(item)
                break

    def add_item(self, x, y):
        x = x.to('cpu')
        if isinstance(y, torch.Tensor):
            y = y.item()
        try:
            self.data[y].append(x)
        except KeyError:
            self.data.update([(y, [x])])

    def sample(self, data):
        self.sampler(data)

    def retrieve(self, data, size):
        return self.retriever(data, size)


class ReservoirSampler:

    def __init__(self, buffer: Buffer):
        self.buffer = buffer
        self.seen_samples = 0

    def __call__(self, data, **kwargs):
        for idx, (x, y) in enumerate(zip(*data)):
            if len(self.buffer) < self.buffer.size:
                self.buffer.add_item(x, y)
            else:
                r = np.random.randint(0, self.seen_samples + idx)
                if r < self.buffer.size:
                    self.buffer.pop(r)
                    self.buffer.add_item(x, y)
        self.seen_samples += len(data[0])


class FirstInSampler:

    def __init__(self, buffer: Buffer):
        """
        Samples only first buffer.size samples, all later samples are discarded.
        """
        self.buffer = buffer

    def __call__(self, data, **kwargs):
        free_space = self.buffer.size - len(self.buffer)

        if free_space > 0:
            for i, (x, y) in enumerate(zip(*data)):
                self.buffer.add_item(x, y)
                if i - 1 == free_space:
                    break


class RandomRetriever:

    def __init__(self, buffer: Buffer):
        """
        Samples at random samples. Only samples with labels not in the current batch are considered.
        If no samples are found None, None is returned.
        """
        self.buffer = buffer

    def __call__(self, data, size, **kwargs):
        _, labels = data

        y = set([label.item() for label in labels])
        allowed_labels = set(self.buffer.data.keys()) - set(y)

        if len(allowed_labels) == 0:
            return None, None
        else:
            max_size = sum([len(self.buffer.data[label]) for label in allowed_labels])

            try:
                idx = np.array(sorted(random.sample(range(max_size), size)))
            except ValueError:  # max_size is smaller than size
                idx = np.array(list(range(max_size)))

            data_iter = iter(allowed_labels)
            curr_label = next(data_iter)

            retrieved_x, retrieved_y = [], []
            for c, i in enumerate(idx):
                while i >= len(self.buffer.data[curr_label]):
                    i -= len(self.buffer.data[curr_label])
                    idx[c + 1:] -= len(self.buffer.data[curr_label])
                    curr_label = next(data_iter)
                retrieved_x.append(self.buffer.data[curr_label][i])
                retrieved_y.append(curr_label)

            return torch.stack(retrieved_x), torch.tensor(retrieved_y)


def _set_sampler(s):
    return {
        'reservoir': ReservoirSampler,
        'first_in': FirstInSampler
    }.get(s)


def _set_retriever(r):
    return {
        'random': RandomRetriever
    }.get(r)
