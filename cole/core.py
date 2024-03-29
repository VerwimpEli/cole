from abc import ABC
from collections import Counter
from typing import Union

import torch
import torch.utils.data
import torchvision
import pickle
import random
from cole.helper import *

__BASE_DATA_PATH = '../data'


def set_data_path(path: str):
    global __BASE_DATA_PATH
    __BASE_DATA_PATH = path


class CLDataLoader:
    """
    Sequential dataloader for continual learning tasks.
    """

    def __init__(self, task_datasets, bs=10, shuffle=True, num_workers=0, task_size=0):
        """
        CL Dataloader
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
                self.data_loaders.append(torch.utils.data.DataLoader(task, self.bs, shuffle=shuffle,
                                                                     num_workers=num_workers))
            else:
                sampler = SizedSampler(task, task_size, batch_size=bs, shuffle=shuffle)
                self.data_loaders.append(torch.utils.data.DataLoader(task, batch_sampler=sampler,
                                                                     num_workers=num_workers))

    def __getitem__(self, item):
        return self.data_loaders[item]

    def __len__(self):
        return len(self.data_loaders)


# TODO!! mean and std for mnist and cifar10 are the same?
# TODO: add support for single label dataset, should use different helper function
def get_split_mnist(tasks=None, joint=False, task_labels=None, transform=None):
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

    if transform is None:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.MNIST(__BASE_DATA_PATH, train=True, download=True)
    test_set = torchvision.datasets.MNIST(__BASE_DATA_PATH, train=False, download=True)

    return make_split_dataset(train_set, test_set, joint, tasks, train_transform=transform,
                              test_transform=transform, task_labels=task_labels)


def get_single_label_mnist(labels: Union[int, Sequence[int]]):
    if isinstance(labels, int):
        labels = [labels]

    train_set = torchvision.datasets.MNIST(__BASE_DATA_PATH, train=True, download=True)
    test_set = torchvision.datasets.MNIST(__BASE_DATA_PATH, train=False, download=True)

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    return make_split_label_set(train_set, test_set, labels, transform)


# TODO: add support for single label dataset, should use different helper function
def get_split_cifar10(tasks=None, joint=False, train_transform=None, test_transform=None, task_labels=None):
    """
    Get split version of the CIFAR 10 dataset.
    :param tasks: int or list with task indices. Task 1 has labels 0 and 1, etc.
    :param joint: Concatenate tasks in joint dataset
    :param train_transform: Train transform. If None, default normalization is used.
    :param test_transform: Test transform. If None, default normalization is used.
    :param task_labels: if not the default task, this can be an Iterable with the labels per task
    :return: DataSplit object with train, test en validation members.
    """
    if tasks is None:
        tasks = [i for i in range(6)]
    if type(tasks) is int:
        tasks = [tasks]

    if train_transform is None:
        train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    if test_transform is None:
        test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.CIFAR10(__BASE_DATA_PATH, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(__BASE_DATA_PATH, train=False, download=True)

    return make_split_dataset(train_set, test_set, joint, tasks, train_transform, test_transform, task_labels)


def get_single_label_cifar10(labels: Union[int, Sequence[int]]):
    if isinstance(labels, int):
        labels = [labels]

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.CIFAR10(__BASE_DATA_PATH, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(__BASE_DATA_PATH, train=False, download=True)

    return make_split_label_set(train_set, test_set, labels, transform)


def get_split_cifar100(task_labels: Sequence[Sequence[int]], joint=False, train_transform=None, test_transform=None):
    """
        Get split version of the CIFAR 100 dataset.
        :param task_labels: Sequence of integer sequences
        :param joint: Concatenate tasks in joint dataset
        :param train_transform: Transform for training, if None, basic is used
        :param test_transform: Transform for testing, if None, basic is used
        :return: DataSplit object with train, test en validation members.
        """

    if train_transform is None:
        train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize((0.5071, 0.4866, 0.4409),
                                                                                           (0.2009, 0.1984, 0.2023))])
    if test_transform is None:
        test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize((0.5071, 0.4866, 0.4409),
                                                                                          (0.2009, 0.1984, 0.2023))])

    train_set = torchvision.datasets.CIFAR100(__BASE_DATA_PATH, train=True, download=True)
    test_set = torchvision.datasets.CIFAR100(__BASE_DATA_PATH, train=False, download=True)

    return make_split_dataset(train_set, test_set, joint, None, train_transform, test_transform, task_labels)


# TODO: Merge with other datasets getters, use task label file for other datasets too. Refactor reader.

def get_split_mini_imagenet(task_labels: Sequence[Sequence[int]], joint=False, train_transform=None,
                            test_transform=None, path=None):
    if path is None:
        path = f"{__BASE_DATA_PATH}/miniImageNet/miniImageNet.pkl"

    with open(path, "rb") as f:
        dataset = pickle.load(f)

    if joint:
        task_labels = [[label for task in task_labels for label in task]]

    train_x, test_x = [], []
    train_y, test_y = [], []

    for i in range(0, len(dataset["labels"]), 600):
        train_x.extend(dataset["data"][i:i + 500])
        test_x.extend(dataset["data"][i + 500:i + 600])
        train_y.extend(dataset["labels"][i:i + 500])
        test_y.extend(dataset["labels"][i + 500:i + 600])

    train_x, test_x = np.array(train_x), np.array(test_x)
    train_y, test_y = np.array(train_y), np.array(test_y)

    train_ds, test_ds = [], []
    for labels in task_labels:
        train_label_idx = [y in labels for y in train_y]
        test_label_idx = [y in labels for y in test_y]
        train_ds.append((train_x[train_label_idx], train_y[train_label_idx]))
        test_ds.append((test_x[test_label_idx], test_y[test_label_idx]))

    default_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                                                                         (0.229, 0.224, 0.225))])
    if train_transform is None:
        train_transform = default_transform
    if test_transform is None:
        test_transform = default_transform

    train_ds = [XYDataset(x[0], x[1], transform=train_transform) for x in train_ds]
    test_ds = [XYDataset(x[0], x[1], transform=test_transform) for x in test_ds]

    return DataSplit(train_ds, None, test_ds)


class MultiHeadModel(nn.Module):

    def __init__(self, backbone: nn.Module, num_heads: int, classes_per_head: int, in_features: int):
        """
        :param backbone: backbone of the multi-head network. The backbone should not have a classification head,
        although in theory you could also use that as the representation.
        """
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList([nn.Linear(in_features, classes_per_head) for _ in range(num_heads)])

    def forward(self, x, task: int):
        x = self.backbone(x)
        x = self.heads[task](x)
        return x


class MLP(nn.Module, ABC):
    def __init__(self, nb_classes=10, hid_nodes=400, hid_layers=2, down_sample=1, input_size=28, in_channels=1):
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

        self.input_size = (input_size // down_sample) * (input_size // down_sample) * in_channels
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

    def feature(self, x):
        if self.down_sample is not None:
            x = self.down_sample(x)
        x = x.view(-1, self.input_size)
        return self.hidden(x)


def get_resnet18(nb_classes=10, input_size=None, nf: int = 20):
    """
    :param nb_classes: nb classes or output nodes
    :param input_size: defines size of input layer Resnet18
    :param nf: nb of feature maps in initial layer
    :return: ResNet18 object
    """
    input_size = [3, 32, 32] if input_size is None else input_size
    return ResNet(BasicBlock, [2, 2, 2, 2], nb_classes, nf, input_size)


def step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, data, target, loss_func=None):
    """
    Updates a model a single step, based on the cross entropy loss of the given and target.
    Loss func can be any function returning a loss and taking arguments (data, target, model).
    """

    if loss_func is None:
        loss_func = loss_wrapper("CE")

    optimizer.zero_grad()
    output = model(data)
    loss = loss_func(output, target, model)
    loss.backward()
    optimizer.step()


def test(model, loaders, avg=True, device='cpu', loss_func=None, print_result=False):
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
        loss, acc = test_dataset(model, loader, device, loss_func)
        acc_arr.append(acc)
        loss_arr.append(loss.item())

    model.train()

    if print_result:
        print(f'Acc : {np.mean(acc_arr) * 100:.2f}%  \t| {acc_arr}')
        print(f'Loss: {np.mean(loss_arr):.4f}  \t| {loss_arr}')

    if avg:
        return 100 * np.mean(acc_arr), np.mean(loss_arr)
    else:
        return acc_arr, loss_arr


# TODO: combine with normal test?
@torch.no_grad()
def test_multitask(model, loaders, real_labels, device='cpu', print_result=True):
    loss_results, acc_results = [], []
    for t, loader in enumerate(loaders):
        correct, length = 0, 0
        loss = 0.0

        for data, target in loader:
            data, target = data.to(device), target.to(device)
            target = label_convert(target, real_labels[t])

            output = model(data, t)
            loss += torch.nn.functional.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            length += len(target)

        loss_results.append(loss.item() / length)
        acc_results.append(correct / length)

    if print_result:
        for t, (loss, acc) in enumerate(zip(loss_results, acc_results)):
            print(f'Test {t} | Acc: {acc * 100:.2f}%  Loss: {loss:.5f}')

    return acc_results, loss_results


def test_per_class(model, loaders, device='cpu', print_result=False):
    if isinstance(loaders, torch.utils.data.DataLoader):
        loaders = [loaders]

    loss, acc = {}, {}
    model.eval()
    for load in loaders:
        loader_loss, loader_acc = test_dataset_per_class(model, load, device)
        loss.update(loader_loss)
        acc.update(loader_acc)
    model.train()

    if print_result:
        print(f'Acc: {np.mean(list(acc.values())) * 100:.2f}'.ljust(20), end='\t')
        fmt_str = '{}: {:.3f}  ' * len(acc)
        sorted_keys = sorted(list(acc.keys()))
        print(fmt_str.format(*[x for key in sorted_keys for x in [key, acc[key]]]))

        print(f'Loss: {np.mean(list(loss.values())):.5e}'.ljust(20), end='\t')
        print(fmt_str.format(*[x for key in sorted_keys for x in [key, loss[key]]]))

    return acc, loss


# TODO: although it acts as a dataset, transformed tensors are stored instead of raw images as in XYDataset
class Buffer(torch.utils.data.Dataset):

    def __init__(self, size, sampler='reservoir', retriever='random', **kwargs):
        """
        Buffer class, to be used as ER buffer. Works as a torch dataset.
        :param size: Buffer size
        :param sampler: Sampler to use. Options: 'reservoir' (def), 'first_in', 'balanced_res' or 'balanced'
        :param retriever: Retriever to use. Options: 'random' (def)
        """
        super(Buffer, self).__init__()
        self.data = {}  # Dictionary, key is label value is x
        self.size = size
        self.sampler = _set_sampler(sampler)(self)
        self.retriever = _set_retriever(retriever)(self, **kwargs)

    # TODO: None is returned during iteration
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

    # TODO Single sample sampling should be easier, now (itr, itr) is expected
    def sample(self, data):
        self.sampler(data)

    def sample_individual(self, x, y):
        """
        Version to add individual samples to buffer. Because it shouldn't be used during training,
        type checks can be performed, so any type that can be converted to a torch.Tensor is accepted.
        """
        if not type(y) == torch.Tensor:
            y = torch.tensor(y, dtype=torch.long)
        if not type(x) == torch.Tensor:
            x = torch.tensor(x)
        self.sampler(([x], [y]))

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


class BalancedReservoirSampler:

    def __init__(self, buffer: Buffer):
        self.buffer = buffer
        self.seen_samples = {}
        self.current_size = buffer.size

    def __call__(self, data, **kwargs):
        for idx, (x, y) in enumerate(zip(*data)):
            if y.item() in self.seen_samples.keys():
                if self.seen_samples[y.item()] < self.current_size:
                    self.buffer.add_item(x, y)
                    self.seen_samples[y.item()] += 1
                else:
                    r = np.random.randint(0, self.seen_samples[y.item()])
                    if r < self.current_size:
                        self.buffer.data[y.item()].pop(r)
                        self.buffer.add_item(x, y)
                        self.seen_samples[y.item()] += 1
            else:
                self.seen_samples[y.item()] = 1
                self.current_size = self.buffer.size // len(self.seen_samples)
                self.resize_buffer()
                self.buffer.add_item(x, y)

    def resize_buffer(self):
        for key in self.buffer.data.keys():
            self.buffer.data[key] = self.buffer.data[key][:self.current_size]


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


class BalancedSampler:

    def __init__(self, buffer: Buffer):
        """
        Samples only first buffer.size samples, but balances classes.
        """
        self.buffer = buffer
        self.keys = []
        self.current_size = buffer.size

    def __call__(self, data, **kwargs):

        for (x, y) in zip(*data):
            if y in self.keys:
                if len(self.buffer.data[y.item()]) < self.current_size:
                    self.buffer.add_item(x, y)
            else:
                self.keys.append(y.item())
                self.current_size = self.buffer.size // len(self.keys)
                self.resize_buffer()
                self.buffer.add_item(x, y)

    def resize_buffer(self):
        for key in self.buffer.data.keys():
            self.buffer.data[key] = self.buffer.data[key][:self.current_size]


class RandomRetriever:

    def __init__(self, buffer: Buffer, labels_of_current_batch: bool = False, **kwargs):
        """
        Samples at random samples. Only samples with labels not in the current batch are considered.
        If no samples are found None, None is returned.
        :param labels_of_current_batch if True, labels of the current batch are allowed in the returend samples.
        """
        self.buffer = buffer
        self.labels_of_current_batch = labels_of_current_batch

    def __call__(self, data, size, **kwargs):
        _, labels = data

        allowed_labels = set(self.buffer.data.keys())

        if not self.labels_of_current_batch:
            y = set([label.item() for label in labels])
            allowed_labels -= y

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


class BalancedRetriever:

    def __init__(self, buffer: Buffer, labels_of_current_batch: bool = False, **kwargs):
        """
        Samples at random samples. Only samples with labels not in the current batch are considered.
        If no samples are found None, None is returned.
        :param labels_of_current_batch if True, labels of the current batch are allowed in the returend samples.
        """
        self.buffer = buffer
        self.labels_of_current_batch = labels_of_current_batch

    def __call__(self, data, size, **kwargs):

        # TODO: currently this goes over max size if too many of one class is in the current data
        # but don't know whether that's really a problem

        _, labels = data
        labels = labels.tolist()

        counts = Counter(labels)
        for key in self.buffer.data.keys():
            if key not in counts:
                counts[key] = 0

        optimal_count = (len(labels) + size) // len(self.buffer.data.keys())
        counts = {k: optimal_count - v for k, v in counts.items()}

        retrieved_x, retrieved_y = [], []

        for k, v in counts.items():
            if v > 0:
                max_size = len(self.buffer.data[k])
                try:
                    idx = np.array(sorted(random.sample(range(max_size), v)))
                except ValueError:  # max_size is smaller than size
                    idx = np.array(list(range(max_size)))
                for i in idx:
                    retrieved_x.append(self.buffer.data[k][i])
                    retrieved_y.append(k)

        if len(retrieved_y) > 0:
            return torch.stack(retrieved_x), torch.tensor(retrieved_y)
        else:
            return None, None


def _set_sampler(s):
    return {
        'reservoir': ReservoirSampler,
        'first_in': FirstInSampler,
        'balanced': BalancedSampler,
        'balanced_res': BalancedReservoirSampler
    }.get(s)


def _set_retriever(r):
    return {
        'random': RandomRetriever,
        'balanced': BalancedRetriever
    }.get(r)
