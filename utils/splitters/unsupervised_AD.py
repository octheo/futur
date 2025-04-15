import os
import glob
import math

from .unsupervised_base import UnsupervisedSplit

class MvtechADUnsupervisedSplit(UnsupervisedSplit):
    def __init__(self, dataset_path, classname, val_split, multiclass=False):
        super().__init__(dataset_path, classname, val_split, multiclass)
    
    def _get_all_defect_names(self):
        all_subclasses = os.listdir(os.path.join(self.root_dir, 'test'))
        return [item for item in all_subclasses if item != "good"]
    
    def _split_train_test(self):

        train_samples = glob.glob(os.path.join(self.root_dir, 'train/good/*.png'))
        self.train = [(train_sample, 0) for train_sample in train_samples]
        
        good_test_samples = glob.glob(os.path.join(self.root_dir, 'test/good/*.png'))
        val_threshold = math.floor(self.val_split*len(good_test_samples))      
        self.val = [(good_test_sample, 0) for good_test_sample in good_test_samples[:val_threshold]]
        self.test = [(good_test_sample, 0) for good_test_sample in good_test_samples[val_threshold:]]
        
        for i, defect_class in enumerate(self.defect_classes):
            samples = glob.glob(os.path.join(self.root_dir, 'test', defect_class, '*.png'))
            
            val_threshold = math.floor(self.val_split*len(samples))      
            self.val += [(sample, i) for sample in samples[:val_threshold]]
            self.test += [(sample, i) for sample in samples[val_threshold:]]