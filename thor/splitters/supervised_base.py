import os
import glob
import re
import math
from collections import Counter
from abc import ABC, abstractmethod

from PIL import Image
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


class MVTech_SP_split(ABC):
    def __init__(self, dataset_path, classname, train_split, val_split, dist_adjust=False, multiclass=False):
        """
        Args:
            root_dir (string): Path to either 'train' or 'test' directory
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_path = dataset_path
        self.classname = classname
        self.root_dir = os.path.join(dataset_path, classname)
        self.train_split = train_split
        self.val_split = val_split
        self.dist_adjust = dist_adjust
        self.defect_classes = self.defect_classes()
        self.multiclass = multiclass
        self.class_mapping = {}

        self.no_defect_samples = glob.glob(os.path.join(self.root_dir, '*/good/*.png'))
        self.nb_no_defect_samples = len(self.no_defect_samples)
        self.nb_defect_samples = 0
        self.defect_samples = {}

        self.train = []
        self.val = []
        self.test = []

        self.create_samples()
        self.train_test()
    
    @abstractmethod
    def create_samples(self):
      pass

    def defect_classes(self):
      all_files = glob.glob(os.path.join(self.root_dir, '*/*'))
      pattern = re.compile(r'good|ground_truth')
      filtered_files = [file for file in all_files if not pattern.search(file)]
      last_subfolders = [os.path.basename(path) for path in filtered_files]
      return last_subfolders

    def train_test(self):
        if not self.dist_adjust:
            adjust = 1
        else:
            adjust = self.nb_no_defect_samples // self.nb_defect_samples
            
        threshold_train = math.ceil((self.train_split*self.nb_no_defect_samples)/adjust)
        threshold_val = threshold_train + math.ceil((len(self.no_defect_samples[threshold_train:]))/2)
        self.train += (self.no_defect_samples[:threshold_train])
        self.val += (self.no_defect_samples[threshold_train:threshold_val])
        self.test += (self.no_defect_samples[threshold_val:])
        
        for i, defect_class in enumerate(self.defect_classes):
            samples = self.defect_samples[str(i+1)]
            
            train_threshold = math.ceil(self.train_split*len(samples))
            val_threshold = train_threshold + math.ceil((self.val_split/2)*len(samples))
            
            self.train += samples[:train_threshold]
            self.val += samples[train_threshold:val_threshold]
            self.test += samples[val_threshold:]
    
    def plot_dist(self):
        
        fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
        global_dist = []
        for i, (dataset_part, title) in enumerate(zip([self.train, self.val, self.test], ["Train", "Val", "Test"])):
            labels = []
            for item in dataset_part:
                if isinstance(item, tuple):
                    labels.append(item[2])  # Extract the label from the tuple
                else:
                    labels.append(0)  # "good" images are labeled as 0
            
            # Count the occurrences of each label
            label_counts = Counter(labels)

            # Plotting the stacked bar chart
            labels_list = list(label_counts.keys())
            counts_list = list(label_counts.values())
            global_dist.append(counts_list)

            colors = plt.cm.get_cmap('tab20', len(labels_list))
            color_list = [colors(i) for i in range(len(labels_list))]

            ax[i].bar(labels_list, counts_list, color=color_list)
            ax[i].set_xlabel('Labels')
            ax[i].set_ylabel('Frequency')
            ax[i].set_title(title)

            patches = [mpatches.Patch(color=color_list[i], label=f'{self.class_mapping[str(i)]}') for i in range(len(labels_list))]
            plt.legend(handles=patches, title="Classes")

        plt.show()
        return global_dist
    
    def return_splits(self):
        return self.train, self.val, self.test