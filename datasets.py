from torch import utils
from PIL import Image
import os


class ImageDataset(utils.data.Dataset):

    def __init__(self, root, meta, transform=None):
        self.root = root
        self.transform = transform
        self.meta = []
        with open(meta) as file:
            for line in file:
                path, label = line.rstrip().split()
                self.meta.append((path, int(label)))

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        path, label = self.meta[idx]
        path = os.path.join(self.root, path)
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label
