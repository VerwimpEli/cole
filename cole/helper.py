from abc import ABC

import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


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


class SizedSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, size, batch_size=10, shuffle=True):
        super(SizedSampler, self).__init__(data_source)
        self.bs = batch_size
        self.size = size if len(data_source) > size else len(data_source)
        self.shuffle = shuffle

        if shuffle:
            self.order = torch.randperm(len(data_source)).tolist()[:self.size]
        else:
            self.order = list(range(self.size))

    def __iter__(self):
        batch = []
        for idx in self.order:
            batch.append(idx)
            if len(batch) == self.bs:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

        if self.shuffle:
            random.shuffle(self.order)

    def __len__(self):
        return np.ceil(self.size // self.bs)


# TODO: if only single task or joint, dont return array but single XY dataset. But, that would complicate loader?
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

    return DataSplit(train_ds, val_ds, test_ds)


class DataSplit:
    def __init__(self, train_ds, val_ds, test_ds):
        self.train = train_ds
        self.validation = val_ds
        self.test = test_ds


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


def test_dataset(model, loader, device='cpu'):
    """
    Return the loss and accuracy on a single dataset (which is provided through a loader)
    """
    loss, correct, length = 0, 0, 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum')
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            length += len(target)

    return loss / length, correct / length


# TODO: clean up code below
class ResBlock(nn.Module, ABC):
    def __init__(self, in_channels, channels, bn=False):
        super(ResBlock, self).__init__()

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(channels))
        self.convolutions = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convolutions(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module, ABC):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module, ABC):
    def __init__(self, block, num_blocks, num_classes, nf, input_size):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.input_size = input_size

        self.conv1 = conv3x3(input_size[0], nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        # hardcoded for now
        last_hid = nf * 8 * block.expansion if input_size[1] in [8, 16, 21, 32, 42] else 640
        self.linear = nn.Linear(last_hid, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def return_hidden(self, x):
        bsz = x.size(0)
        # pre_bn = self.conv1(x.view(bsz, 3, 32, 32))
        # post_bn = self.bn1(pre_bn, 1 if is_real else 0)
        # out = F.relu(post_bn)
        out = F.relu(self.bn1(self.conv1(x.view(bsz, *self.input_size))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.return_hidden(x)
        out = self.linear(out)
        return out
