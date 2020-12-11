import sys
import unittest
import torch
import cole


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

    def test_fill_buffer(self):
        buffer_size = 100
        buffer = cole.Buffer(size=buffer_size, sampler="first_in")
        data = cole.get_split_mnist(1)
        loader = cole.CLDataLoader(data.train)[0]

        for data, target in loader:
            buffer.sample((data, target))

        self.assertEqual(len(buffer), buffer_size)


if __name__ == '__main__':
    unittest.main()
