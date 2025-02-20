import torch
from data import CompressedImageDataset
import os
#device = "cuda" if torch.cuda.is_available() else "cpu"

#print(device)

data = CompressedImageDataset(compressed_dir="./compressed/", original_dir="./images/")

compressed_dir = "./compressed/"
original_dir = "./images/"

compressed_files = set(os.listdir(compressed_dir))
original_files = set(os.listdir(original_dir))

# Find missing images
missing_in_compressed = original_files - compressed_files
missing_in_original = compressed_files - original_files

print(f"Images missing in compressed folder: {len(missing_in_compressed)}")
print(f"Images missing in original folder: {len(missing_in_original)}")

# Print some missing files
print("Some missing in compressed:", list(missing_in_compressed)[:5])
print("Some missing in original:", list(missing_in_original)[:5])