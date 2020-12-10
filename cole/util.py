import numpy as np
import random
import torch
import torch.utils.data
import torch.nn.functional as F
from cllib.model import MLP, ResNet18
import torch.autograd.functional


def step(model, optimizer, data, target, args=None):
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()


def test(model, loaders, device):
    model.eval()
    total_correct = 0
    loss = 0
    length = 0

    for task, loader in enumerate(loaders):
        correct = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss += F.cross_entropy(output, target, reduction='sum')
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                length += len(target)

        total_correct += correct

    model.train()
    return 100 * total_correct / length, loss / length


def diff_matrix(a):
    n, m = a.shape
    result = np.zeros((n-2, m-2))
    for i in range(1, n-1):
        for j in range(1, m-1):
            result[i-1, j-1] = a[i-1, j] + a[i+1, j] + a[i, j+1] + a[i, j-1] - 4*a[i, j]
    return result


def build_model(path, data='mnist', device='cpu'):
    model = MLP(None) if data == 'mnist' else ResNet18(None)
    model.load_state_dict(torch.load(path))
    model.eval()
    model.to(device)
    return model


def flatten_parameters(model):
    with torch.no_grad():
        flat = torch.cat([
            p.view(-1) for p in model.parameters()
        ])
    return flat


def flatten_parameters_wg(model):
    return torch.cat([p.view(-1) for p in model.parameters()])


def assign_parameters(model, params):
    state_dict = model.state_dict(keep_vars=False)
    with torch.no_grad():
        total_count = 0
        for param in state_dict.keys():
            if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
            param_count = state_dict[param].numel()
            param_size = state_dict[param].shape
            state_dict[param] = params[total_count:total_count+param_count].view(param_size).clone()
            total_count += param_count
        model.load_state_dict(state_dict)
    return model


class WeightPlane:
    # Weights should be flattened before creating plane
    def __init__(self, w1, w2, w3):
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.u = w2 - w1
        self.v = w3 - w1
        self.v -= torch.dot(self.u, self.v) * self.u / torch.dot(self.u, self.u)

    def xy_to_weights(self, x, y):
        return self.w1 + x * self.u + y * self.v

    # If optional parameters are provided output is slightly different structured than the other case.
    def weights_to_xy(self):
        non_orth_v = self.w3 - self.w1
        w3_x = (torch.dot(self.u, non_orth_v) / torch.dot(self.u, self.u)).item()
        model_weights = [[0, 0], [1, 0], [w3_x, 1]]
        return np.array(model_weights)


class Buffer(torch.utils.data.Dataset):

    def __init__(self, size):
        super(Buffer, self).__init__()
        # Dictionary, key is label value is x
        self.data = {}
        self.seen_samples = 0
        self.size = size

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

    # Reservoir sampling
    # TODO: small opt, don't always check the length, once the buffer is full, it'll stay full
    def sample(self, data):
        for idx, (x, y) in enumerate(zip(*data)):
            if len(self) < self.size:
                self.add_item(x, y)
            else:
                r = np.random.randint(0, self.seen_samples + idx)
                if r < self.size:
                    self.pop(r)
                    self.add_item(x, y)
        self.seen_samples += len(data[0])

    # Retrieve random samples from labels not in current batch
    def retrieve(self, labels, size):
        y = set([l.item() for l in labels])
        retr_labels = set(self.data.keys()) - set(y)

        # Buffer contains only samples of current batch
        if len(retr_labels) == 0:
            return None, None

        retr_len = sum([len(self.data[label]) for label in retr_labels])
        try:
            idx = np.array(sorted(random.sample(range(retr_len), size)))  # Np doesn't support sampling wo replacement
        except ValueError:  # retr_len is shorter than size
            idx = np.array(sorted(range(retr_len)))

        data_iter = iter(retr_labels)
        curr_label = next(data_iter)

        retr_x, retr_y = [], []
        for c, i in enumerate(idx):
            while i >= len(self.data[curr_label]):
                i -= len(self.data[curr_label])
                idx[c+1:] -= len(self.data[curr_label])
                curr_label = next(data_iter)
            retr_x.append(self.data[curr_label][i])
            retr_y.append(curr_label)

        return torch.stack(retr_x), torch.tensor(retr_y)
