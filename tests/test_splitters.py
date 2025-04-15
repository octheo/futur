import math
import sys
import os
import unittest

from futur.utils.splitters.unsupervised_AD import MvtechADUnsupervisedSplit


class TestUnsupervisedSplitters(unittest.TestCase):

    def setUp(self):
        self.binary_splitter = MvtechADUnsupervisedSplit(
            dataset_path="/Users/theo.moreau/Downloads/mvtec_anomaly_detection/",
            classname="bottle",
            val_split=0.5
        )
        self.multiclass_splitter = MvtechADUnsupervisedSplit(
            dataset_path="/Users/theo.moreau/Downloads/mvtec_anomaly_detection/",
            classname="zipper",
            val_split=0.5,
            multiclass=True
        )

        self.nb_bottle_train_examples = 209
        self.nb_bottle_test_defect_examples = 63
        self.nb_bottle_test_good_examples = 20
        self.nb_bottle_val_examples = self.nb_bottle_test_defect_examples + self.nb_bottle_test_good_examples

        self.zipper_defect_classes = 7
        self.nb_zipper_train_examples = 240
        self.nb_zipper_test_defect_examples = 119
        self.nb_zipper_test_good_examples = 32
        self.nb_zipper_val_examples = self.nb_zipper_test_defect_examples + self.nb_zipper_test_good_examples

    def test_binary_split(self):

        self.assertEqual(len(self.binary_splitter.class_mapping), 2)
        self.assertEqual(len(self.binary_splitter.train), self.nb_bottle_train_examples)
        self.assertEqual(len(self.binary_splitter.val) + len(self.binary_splitter.test), self.nb_bottle_val_examples)

    def test_multiclass_split(self):

        self.assertEqual(len(self.multiclass_splitter.class_mapping), self.zipper_defect_classes + 1)     
        self.assertEqual(len(self.multiclass_splitter.train), self.nb_zipper_train_examples)
        self.assertEqual(len(self.multiclass_splitter.val) + len(self.multiclass_splitter.test), self.nb_zipper_val_examples)

if __name__ == "__main__":
    unittest.main()

