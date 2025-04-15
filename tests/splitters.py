import math
import sys
import os
import unittest

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.splitters.supervised_AD import MVTech_AD_supervised_cls_split


class TestUnsupervisedSplitters(unittest.TestCase):

    def setUp(self):
        self.binary_splitter = MVTech_AD_supervised_cls_split(
            dataset_path="/Users/theo.moreau/Downloads/mvtec_anomaly_detection/",
            classname="bottle",
            train_split=0.8,
            val_split=0.5
        )
        self.multiclass_splitter = MVTech_AD_supervised_cls_split(
            dataset_path="/Users/theo.moreau/Downloads/mvtec_anomaly_detection/",
            classname="zipper",
            train_split=0.8,
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
        self.binary_splitter.unsupervised_train_test()

        self.assertEqual(len(self.binary_splitter.class_mapping), 2)
        self.assertEqual(self.binary_splitter.nb_defect_samples, self.nb_bottle_val_examples)
        self.assertEqual(self.binary_splitter.nb_no_defect_samples, self.nb_bottle_train_examples)
        self.assertEqual(len(self.binary_splitter.train), self.nb_bottle_train_examples)

        self.assertEqual(len(self.binary_splitter.val), math.floor(self.nb_bottle_val_examples*0.5))
        self.assertEqual(len(self.binary_splitter.test), self.nb_bottle_val_examples - math.floor(self.nb_bottle_val_examples*0.5))
    
    def test_multiclass_split(self):
        self.multiclass_splitter.unsupervised_train_test()

        self.assertEqual(len(self.multiclass_splitter.class_mapping), self.zipper_defect_classes + 1)
        self.assertEqual(self.multiclass_splitter.nb_defect_samples, self.nb_zipper_test_defect_examples + self.nb_zipper_test_good_examples)
        self.assertEqual(self.multiclass_splitter.nb_no_defect_samples, self.nb_zipper_train_examples)
        
        self.assertEqual(len(self.multiclass_splitter.train), self.nb_zipper_train_examples)

if __name__ == "__main__":
    unittest.main()

