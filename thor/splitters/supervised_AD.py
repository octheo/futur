import os
import glob

from .supervised_base import MVTech_SP_split


class MVTech_AD_supervised_split(MVTech_SP_split):
    def __init__(self, dataset_path, classname, train_split, dist_adjust=False, multiclass=False):
        super().__init__(dataset_path, classname, train_split, dist_adjust, multiclass)
        
    def create_samples(self):
      self.class_mapping["0"] = "No defect"

      for i, defect_class in enumerate(self.defect_classes):
        image_path = os.path.join(self.root_dir, "test/", defect_class)
        mask_path = os.path.join(self.root_dir, "ground_truth/", defect_class)

        images = glob.glob(os.path.join(image_path, '*.png'))
        images.sort()

        masks = glob.glob(os.path.join(mask_path, '*.png'))
        masks.sort()

        self.defect_samples[str(i+1)] = []
        if self.multiclass:
            for image, mask in zip(images, masks):
              self.defect_samples[str(i+1)].append((image, mask, i+1))

            self.class_mapping[str(i+1)] = defect_class
        else:
          for image, mask in zip(images, masks):
            self.defect_samples[str(i+1)].append((image, mask, 1))
            self.class_mapping["1"] = "Defect"
        self.nb_defect_samples += len(self.defect_samples[str(i+1)])

        
class MVTech_LOCO_supervised_split(MVTech_SP_split):
    def __init__(self, dataset_path, classname, train_split, dist_adjust=False, multiclass=False):
        super().__init__(dataset_path, classname, train_split, dist_adjust, multiclass)
        
    def create_samples(self):
      self.class_mapping["0"] = "No defect"

      for i, defect_class in enumerate(self.defect_classes):
        image_path = os.path.join(self.root_dir, "test/", defect_class)
        mask_path = os.path.join(self.root_dir, "ground_truth/", defect_class)

        images = glob.glob(os.path.join(image_path, '*.png'))
        images.sort()

        masks = glob.glob(os.path.join(mask_path, '*/*.png'))
        masks.sort()

        self.defect_samples[str(i+1)] = []
        if self.multiclass:
            for image, mask in zip(images, masks):
              self.defect_samples[str(i+1)].append((image, mask, i+1))

            self.class_mapping[str(i+1)] = defect_class
        else:
          for image, mask in zip(images, masks):
            self.defect_samples[str(i+1)].append((image, mask, 1))
            self.class_mapping["1"] = "Defect"
        self.nb_defect_samples += len(self.defect_samples[str(i+1)])