import unittest

import numpy as np

from futur.utils.datasets.mvtech import MVTechDataset_cls
from futur.utils.splitters.unsupervised_AD import MvtechADUnsupervisedSplit

class TestDatasets(unittest.TestCase):

    def setUp(self):
        self.binary_splitter = MvtechADUnsupervisedSplit(
            dataset_path="/Users/theo.moreau/Downloads/mvtec_anomaly_detection/",
            classname="bottle",
            val_split=0.5
        )

        self.dataset = MVTechDataset_cls(self.binary_splitter.train)


    def test_cls_dataset(self):
        self.assertEqual(len(self.dataset), len(self.binary_splitter.train))
        self.assertIsInstance(self.dataset[0][0], np.ndarray)
        self.assertIsInstance(self.dataset[0][1], int)

if __name__ == "__main__":
    unittest.main()