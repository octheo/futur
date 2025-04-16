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


class UnsupervisedSplit(ABC):
    def __init__(self, dataset_path, classname, val_split, multiclass=False):
        """
        Args:
            root_dir (string): Path to either 'train' or 'test' directory
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_path = dataset_path
        self.classname = classname
        self.root_dir = os.path.join(dataset_path, classname)
        self.val_split = val_split
        self.multiclass = multiclass

        self.defect_classes = self._get_all_defect_names()
        self.class_mapping = self._create_class_mapping()

        self.train = []
        self.train_labels = []
        self.val = []
        self.val_labels = []
        self.test = []
        self.test_labels = []

        self._split_train_test()

    @abstractmethod
    def _get_all_defect_names(self): 
      pass
    
    @abstractmethod
    def _split_train_test(self):
        pass

    def _create_class_mapping(self):
        if self.multiclass:
            return {i: defect_class for i, defect_class in enumerate(self.defect_classes + ["good"])}
        else:
            return {0: "good", 1: "defect"}
    
    def plot_dist(self):
        
        fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
        global_dist = []
        for i, (dataset_part, title) in enumerate(zip([self.train, self.val, self.test], ["Train", "Val", "Test"])):
            labels = []
            for item in dataset_part:
                if isinstance(item, tuple):
                    labels.append(item[1])  # Extract the label from the tuple
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