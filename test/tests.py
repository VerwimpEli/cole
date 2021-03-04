import unittest
import torch
import cole
import os
import time
from copy import deepcopy


class CoreTest(unittest.TestCase):

    def test_split_mnist_task_1(self):
        data = cole.get_split_mnist(tasks=1)
        train_set = data.train

        self.assertEqual(len(train_set), 1)     # Only a single set in the dataset

        train_labels = set(label.item() for label in data.train[0].y)
        expected_labels = {0, 1}

        self.assertEqual(train_labels, expected_labels)

    def test_split_mnist_multiple_tasks(self):
        data = cole.get_split_mnist(tasks=[3, 4])
        train_set = data.train

        self.assertEqual(len(train_set), 2)     # Only a single set in the dataset

        train_labels_1 = set(label.item() for label in data.train[0].y)
        train_labels_2 = set(label.item() for label in data.train[1].y)

        self.assertEqual(train_labels_1, {4, 5})
        self.assertEqual(train_labels_2, {6, 7})

    def test_split_mnist_multiple_non_consecutive_tasks(self):
        data = cole.get_split_mnist(tasks=[2, 4, 5])
        train_set = data.train

        self.assertEqual(len(train_set), 3)     # Only a single set in the dataset

        train_labels_1 = set(label.item() for label in data.train[0].y)
        train_labels_2 = set(label.item() for label in data.train[1].y)
        train_labels_3 = set(label.item() for label in data.train[2].y)

        self.assertEqual(train_labels_1, {2, 3})
        self.assertEqual(train_labels_2, {6, 7})
        self.assertEqual(train_labels_3, {8, 9})

    def test_joint_multiple_tasks(self):
        data = cole.get_split_mnist(tasks=[2, 3, 4], joint=True)
        train_set = data.train

        self.assertEqual(len(train_set), 1)  # Only a single set in the dataset

        train_labels_1 = set(label.item() for label in data.train[0].y)

        self.assertEqual(train_labels_1, {2, 3, 4, 5, 6, 7})

    def test_cl_data_loader(self):
        self.cl_data_loader(task_size=0)
        self.cl_data_loader(task_size=105)

    def cl_data_loader(self, task_size):

        data_set = cole.get_split_mnist(tasks=1)
        loaders = cole.CLDataLoader(data_set.train, bs=25, shuffle=False, task_size=task_size)

        self.assertEqual(len(loaders), 1)

        loader = next(iter(loaders))

        size = 0
        target_order_1 = []
        target_order_2 = []
        for data, target in loader:
            size += len(target)
            target_order_1.append(target)

        loader = cole.CLDataLoader(data_set.train, bs=25, shuffle=False, task_size=task_size)[0]
        for data, target in loader:
            target_order_2.append(target)

        if task_size != 0:
            self.assertEqual(size, task_size)
        else:
            self.assertEqual(size, len(data_set.train[0]))

        # Order should always be the same if shuffle is False
        order = torch.all(torch.cat(target_order_1) == torch.cat(target_order_2))
        self.assertTrue(order)

        shuffled_loader = cole.CLDataLoader(data_set.train, bs=25, shuffle=True, task_size=task_size)
        shuffled_order = []
        for data, target in shuffled_loader[0]:
            shuffled_order.append(target)

        # Shuffled order is different than non-shuffled
        shuf_order = torch.all(torch.cat(target_order_1) == torch.cat(shuffled_order))
        self.assertFalse(shuf_order)

        shuffled_loader = cole.CLDataLoader(data_set.train, bs=25, shuffle=True, task_size=task_size)
        shuffled_order_2 = []
        for data, target in shuffled_loader[0]:
            shuffled_order_2.append(target)

        # Shuffled order is different when dataloader is re-initiated
        shuf_order_2 = torch.all(torch.cat(shuffled_order_2) == torch.cat(shuffled_order))
        self.assertFalse(shuf_order_2)

        shuffled_order_3 = []
        for data, target in shuffled_loader[0]:
            shuffled_order_3.append(target)

        # Shuffled order is different when iterated again over loader
        shuf_order_2 = torch.all(torch.cat(shuffled_order_2) == torch.cat(shuffled_order_3))
        self.assertFalse(shuf_order_2)

    def test_single_label(self):

        label = 1
        ds = cole.get_single_label_mnist(label)
        self.assertEqual(ds.train[0].get_labels(), {label})

    def test_fill_buffer(self):
        buffer_size = 100
        buffer = cole.Buffer(size=buffer_size, sampler="first_in")
        data = cole.get_split_mnist(1)
        loader = cole.CLDataLoader(data.train)[0]

        for data, target in loader:
            buffer.sample((data, target))

        self.assertEqual(len(buffer), buffer_size)

    def test_iterate_buffer(self):
        # TODO TODO TODO
        pass


class UtilTest(unittest.TestCase):

    def test_load_model_MLP(self):
        model = cole.MLP()
        torch.save(model.state_dict(), "./temp.pt")
        loaded_model = cole.build_model(f"./temp.pt", 'mnist')
        os.system("rm ./temp.pt")

        flat_model = cole.flatten_parameters(model)
        flat_loaded_model = cole.flatten_parameters(loaded_model)

        compare = torch.all(flat_loaded_model == flat_model)
        self.assertTrue(compare)

    def test_load_model_resnet(self):
        model = cole.get_resnet18()
        torch.save(model.state_dict(), "./temp.pt")
        loaded_model = cole.build_model(f"./temp.pt", 'cifar')
        os.system("rm ./temp.pt")

        flat_model = cole.flatten_parameters(model)
        flat_loaded_model = cole.flatten_parameters(loaded_model)

        compare = torch.all(flat_loaded_model == flat_model)
        self.assertTrue(compare)

    def test_weight_plane(self):
        model_1, model_2, model_3 = cole.MLP(), cole.MLP(), cole.MLP()
        plane = cole.WeightPlane(*(cole.flatten_parameters(m) for m in [model_1, model_2, model_3]))

        # Assert base vectors are orthonormal for random models, ortho normality is pretty bad for random
        self.assertAlmostEqual(torch.dot(plane.u, plane.v), 0, delta=1e-3)
        self.assertAlmostEqual(torch.norm(plane.u).item(), 1.0, delta=1e-5)
        self.assertAlmostEqual(torch.norm(plane.v).item(), 1.0, delta=1e-5)

        flat_model_1 = torch.zeros_like(cole.flatten_parameters(model_1))

        flat_model_2 = torch.zeros_like(flat_model_1)
        flat_model_2[0] = 2.5
        flat_model_2 += flat_model_1

        flat_model_3 = torch.zeros_like(flat_model_1)
        flat_model_3[1] = 1.5
        flat_model_3 += flat_model_1

        plane = cole.WeightPlane(flat_model_1, flat_model_2, flat_model_3)

        # test plane xy to weights
        x, y = 0.25, 0.79
        new_weights = plane.xy_to_weights(x, y)
        self.assertAlmostEqual(new_weights[0].item(), flat_model_1[0].item() + x * 2.5, delta=1e-5)
        self.assertAlmostEqual(new_weights[1].item(), flat_model_1[1].item() + y * 1.5, delta=1e-5)

        model_xys = plane.weights_to_xy()
        self.assertEqual(model_xys[2][0], 0.0)

        flat_model_4 = flat_model_3.clone()
        flat_model_4[3:100] += 0.05
        proj = plane.project_onto_plane(flat_model_4)
        self.assertAlmostEqual(proj[0], 0)
        self.assertAlmostEqual(proj[1], 1.0)


if __name__ == '__main__':
    unittest.main()
