from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class CompressedImageDataset(Dataset):
    def __init__(self, compressed_dir, original_dir, transform = None):
        self.compressed_images = sorted(os.listdir(compressed_dir))
        self.original_images = sorted(os.listdir(original_dir))
        self.compressed_dir = compressed_dir
        self.original_dir = original_dir
        self.transfrom = transform

    def __len__(self):
        return len(self.compressed_images)
    
    def __getitem__(self, index):
        compressed_path = os.path.join(self.compressed_dir, self.compressed_images[index])
        original_path = os.path.join(self.original_dir, self.original_images[index])

        compressed = Image.open(compressed_path)
        original = Image.open(original_path)

        if (self.transfrom):
            compressed = self.transfrom(compressed)
            original = self.transfrom(original)

        return compressed, original
    
    #def testni_print(self):
        #print(f"Originals: {self.original_images}")
        #print(f"Originals: {self.compressed_images}")
    

original = "./images/"
compressed = "./compressed/"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = CompressedImageDataset(compressed, original, transform)
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)