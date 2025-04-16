import unittest
import glob
import os

import numpy as np

from futur.utils.datasets.mvtech import MVTechDataset_cls

class TestDatasets(unittest.TestCase):

    def setUp(self):
        BASE_PATH = "/Users/theo.moreau/Downloads/mvtec_anomaly_detection/hazelnut/train/good"
        self.train_path = glob.glob(os.path.join(BASE_PATH, '*.png'))
        self.train_labels = [0] * len(self.train_path)

        self.dataset = MVTechDataset_cls(self.train_path, self.train_labels)


    def test_length(self):
        self.assertEqual(len(self.dataset), len(self.train_path))

    def test_item_type(self):
        self.assertIsInstance(self.dataset[0][0], np.ndarray)
        self.assertIsInstance(self.dataset[0][1], int)

if __name__ == "__main__":
    unittest.main()