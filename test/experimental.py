import cole as cl
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm


# data = cl.get_split_mnist(tasks=1)

# img = data.train[0][256][0]
# img = img.unsqueeze(0)
# img = nn.functional.max_pool2d(img, kernel_size=3)
# img = img.squeeze(0)
#
# img = img.numpy()
# img = np.moveaxis(img, 0, 2)
#
#
# plt.imshow(img)
# plt.show()


class tmlp(nn.Module):

    def __init__(self, nb_classes=10, hid_nodes=400, hid_layers=2, down_sample=1, input_size=28):
        """
        Simple 2-layer MLP with RELU activation
        :param nb_classes: nb of outputs nodes, i.e. classes
        :param hid_nodes: nb of hidden nodes per layer
        :param hid_layers: nb of hidden layers in model
        :param down_sample: down sample data before entering model with a factor down_sample
        :param input_size: data width/height before possible down_sampling
        """
        super(tmlp, self).__init__()

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


def hessian(grad, model):
    hess = torch.empty((grad.numel(), grad.numel()))
    for i, g in enumerate(grad):
        hess_row = autograd.grad(g, model.parameters(), retain_graph=True)
        hess_row = torch.cat([hr.view(-1) for hr in hess_row])
        hess[i] = hess_row
    return hess

# x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# loss = torch.sum(torch.pow(x, 2))
# grad = autograd.grad(loss, x, create_graph=True)[0]
#
# for g in grad:
#     print(autograd.grad(g, x, retain_graph=True))
# exit()

model = tmlp(hid_nodes=30, hid_layers=1)
dataset = cl.get_split_mnist(tasks=1)
loader = cl.CLDataLoader(dataset.train, bs=10)[0]
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for data, target in loader:
    cl.step(model, optimizer, data, target)

test_loader = cl.CLDataLoader(dataset.test, bs=256)[0]
data, target = next(iter(test_loader))

output = model(data)
loss = F.cross_entropy(output, target)
grad = autograd.grad(loss, model.parameters(), create_graph=True)
grad = torch.cat([g.view(-1) for g in grad])

hess = hessian(grad, model)
plt.imshow(hess, vmin=0, vmax=1e-2)
plt.show()

hess = hess.numpy()
w, v = np.linalg.eig(hess)
plt.plot(w)
plt.show()
print("Hallo!")
