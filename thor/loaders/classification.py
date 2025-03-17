from torch.utils.data import Dataset, random_split

class MVTech_classfication_split():
    def __init__(self, dataset_path, classname, train_split, dist_adjust=1, multiclass=False):
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

    def defect_classes(self):
      all_files = glob.glob(os.path.join(self.classname, '*/*'))
      pattern = re.compile(r'good|ground_truth')
      filtered_files = [file for file in all_files if not pattern.search(file)]
      last_subfolders = [os.path.basename(path) for path in filtered_files]
      return last_subfolders

    def train_test(self):
        threshold = math.ceil((self.train_split*self.nb_no_defect_samples)/self.dist_adjust)
        self.train += (self.no_defect_samples[:threshold])
        self.test += (self.no_defect_samples[threshold:threshold+20])
        for i, defect_class in enumerate(self.defect_classes):
            samples = self.defect_samples[str(i+1)]
            self.train += (samples[:math.ceil(self.train_split*len(samples))])
            self.test += (samples[math.ceil(self.train_split*len(samples)):])


class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        Args:
            root_dir (string): Path to either 'train' or 'test' directory
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(self.samples[idx], tuple):
            img_path, mask_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            mask = plt.imread(mask_path)
            mask = np.expand_dims(mask, axis=0)
        else:
            image = Image.open(self.samples[idx]).convert('RGB')
            mask = np.zeros((1, image.size[0], image.size[1]))
            label = 0

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, mask, label