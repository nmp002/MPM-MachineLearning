import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class MicroscopyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._get_samples()

    def _get_samples(self):
        samples = []
        for _, row in self.data_frame.iterrows():
            sample_id = row['sample_id']
            recurrence_score = row['recurrence_score']
            sample_path = os.path.join(self.root_dir, f"{sample_id}")

            if os.path.exists(sample_path):
                for fov in range(1, 6):
                    fov_dir = os.path.join(sample_path, f"fov{fov}")
                    if os.path.exists(fov_dir):
                        fad_path = os.path.join(fov_dir, "fad.tif")
                        nadh_path = os.path.join(fov_dir, "nadh.tif")
                        shg_path = os.path.join(fov_dir, "shg.tif")

                        if os.path.exists(fad_path) and os.path.exists(nadh_path) and os.path.exists(shg_path):
                            samples.append((fad_path, nadh_path, shg_path, recurrence_score))
                        else:
                            print(f"Missing images in {fov_dir}")
                    else:
                        print(f"Missing FOV directory: {fov_dir}")
            else:
                print(f"Missing sample directory: {sample_path}")

        print(f"Total samples found: {len(samples)}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fad_path, nadh_path, shg_path, label = self.samples[idx]

        # Load images
        fad_image = Image.open(fad_path).convert("L")  # Convert to grayscale
        nadh_image = Image.open(nadh_path).convert("L")
        shg_image = Image.open(shg_path).convert("L")

        # Resize all images to match the size of the FAD image
        size = fad_image.size
        nadh_image = nadh_image.resize(size)
        shg_image = shg_image.resize(size)

        # Combine all images into a single 3-channel image (FAD, NADH, SHG)
        combined_image = Image.merge("RGB", (fad_image, nadh_image, shg_image))

        # Apply transformations if provided
        if self.transform:
            combined_image = self.transform(combined_image)

        # Ensure the label is returned as a tensor
        label_tensor = torch.tensor(label, dtype=torch.float)

        return combined_image, label_tensor


# Define transformations (resize, convert to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to 512x512 or another desired size
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize with mean and std
])

# Create dataset and dataloader
dataset = MicroscopyDataset(csv_file="C:/Users/nmp002/PycharmProjects/HighlandsMachineLearning/data/labels.csv",
                            root_dir="C:/Users/nmp002/PycharmProjects/HighlandsMachineLearning/data",
                            transform=transform)

print(f"Number of samples in dataset: {len(dataset)}")

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Test loading data
if __name__ == "__main__":
    for images, labels in dataloader:
        print("Image batch shape:", images.shape)  # Check the shape of the image tensor
        print("Labels batch:", labels)  # Check the corresponding labels
        break
