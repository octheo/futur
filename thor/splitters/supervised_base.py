import os
import glob
import re
import math

from abc import ABC, abstractmethod

from PIL import Image
import numpy as np
import matplotlib.patches as mpatches
from collections import Counter
import matplotlib.pyplot as plt


class MVTech_SP_split(ABC):
    def __init__(self, dataset_path, classname, train_split, dist_adjust=False, multiclass=False):
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
        self.dist_adjust = dist_adjust
        self.defect_classes = self.defect_classes()
        self.multiclass = multiclass
        self.class_mapping = {}

        self.no_defect_samples = glob.glob(os.path.join(self.root_dir, '*/good/*.png'))
        self.nb_no_defect_samples = len(self.no_defect_samples)
        self.nb_defect_samples = 0
        self.defect_samples = {}

        self.train = []
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
        threshold_test = threshold_train + (len(self.no_defect_samples[:threshold_train])//adjust)+1
        self.train += (self.no_defect_samples[:threshold_train])
        self.test += (self.no_defect_samples[threshold_train:threshold_test])
        
        for i, defect_class in enumerate(self.defect_classes):
            samples = self.defect_samples[str(i+1)]
            self.train += (samples[:math.ceil(self.train_split*len(samples))])
            self.test += (samples[math.ceil(self.train_split*len(samples)):])
    
    def plot_dist(self):
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        global_dist = []
        for i, (dataset_part, title) in enumerate(zip([self.train, self.test], ["Train", "Val"])):
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