import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt


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
                        orr_path = os.path.join(fov_dir, f"fov{fov}colorORRMapUniform.jpg")

                        if all(os.path.exists(p) for p in [fad_path, nadh_path, shg_path, orr_path]):
                            samples.append((fad_path, nadh_path, shg_path, orr_path, recurrence_score))
                        else:
                            print(f"Missing images in {fov_dir}")
                    else:
                        print(f"Missing FOV directory: {fov_dir}")
            else:
                print(f"Missing sample directory: {sample_path}")

        print(f"Total samples found: {len(samples)}")
        return samples

    def crop_orr_map(self, orr_image):
        width, height = orr_image.size
        top_crop = int(height * 0.08)  # Crop top 7%
        bottom_crop = int(height * 0.89)  # Keep 93% height
        left_crop = int(width * 0.12)  # Crop left 20%
        right_crop = int(width * 0.81)  # Keep 80% width
        return orr_image.crop((left_crop, top_crop, right_crop, bottom_crop))

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        fad_path, nadh_path, shg_path, orr_path, label = self.samples[idx]

        # Load images
        fad_image = Image.open(fad_path).convert("L")  # Convert to grayscale
        nadh_image = Image.open(nadh_path).convert("L")
        shg_image = Image.open(shg_path).convert("L")
        orr_image = Image.open(orr_path).convert("L")

        # Crop the ORR map to remove title and color bar
        orr_image = self.crop_orr_map(orr_image)

        # Resize all images to match the size of the FAD image
        size = fad_image.size
        nadh_image = nadh_image.resize(size)
        shg_image = shg_image.resize(size)
        orr_image = orr_image.resize(size)

        # Combine all images into a single 4-channel image (FAD, NADH, SHG, ORR)
        combined_image = Image.merge("RGBA", (fad_image, nadh_image, shg_image, orr_image))

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
    transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])  # Normalize with mean and std
])

# Create dataset and dataloader
dataset = MicroscopyDataset(csv_file="C:/Users/nmp002/PycharmProjects/HighlandsMachineLearning/data/labels.csv",
                            root_dir="C:/Users/nmp002/PycharmProjects/HighlandsMachineLearning/data",
                            transform=transform)

print(f"Number of samples in dataset: {len(dataset)}")

def visualize_cropped_orr(dataset, num_samples=3):
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))

    for i in range(num_samples):
        _, _, _, orr_path, _ = dataset.samples[i]  # Get ORR map path
        orr_image = Image.open(orr_path).convert("L")

        # Original ORR map
        axes[i, 0].imshow(orr_image, cmap="jet")
        axes[i, 0].set_title("Original ORR Map")
        axes[i, 0].axis("off")

        # Cropped ORR map
        cropped_orr = dataset.crop_orr_map(orr_image)
        axes[i, 1].imshow(cropped_orr, cmap="jet")
        axes[i, 1].set_title("Cropped ORR Map")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

# visualize_cropped_orr(dataset)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Test loading data
if __name__ == "__main__":
    for images, labels in dataloader:
        print("Image batch shape:", images.shape)  # Check the shape of the image tensor
        print("Labels batch:", labels)  # Check the corresponding labels
        break
