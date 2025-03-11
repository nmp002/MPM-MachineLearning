import torch
from torch.utils.data import Dataset
import tifffile as tiff
import pandas as pd
import os


class MicroscopyDataset(Dataset):
    def __init__(self, csv_file, root_dir, channels=['fad', 'nadh', 'shg', 'orr'], label='classification',
                 transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._get_samples()
        self.label = label
        self.channels = channels

    def _get_samples(self):
        samples = {}
        for _, row in self.data_frame.iterrows():
            sample_id = row['sample_id']
            recurrence_score = row['recurrence_score']
            # score_range = row['score_range']
            sample_path = os.path.join(self.root_dir, f"{sample_id}")

            if os.path.exists(sample_path):
                fov_list = []
                for fov in range(1, 6):
                    fov_dir = os.path.join(sample_path, f"fov{fov}")
                    if os.path.exists(fov_dir):
                        fad_path = os.path.join(fov_dir, "fad.tiff")
                        nadh_path = os.path.join(fov_dir, "nadh.tiff")
                        shg_path = os.path.join(fov_dir, "shg.tiff")
                        orr_path = os.path.join(fov_dir, "orr.tiff")

                        if all(os.path.exists(p) for p in [fad_path, nadh_path, shg_path, orr_path]):
                            fov_list.append((fad_path, nadh_path, shg_path, orr_path, recurrence_score))

                if fov_list:
                    samples[sample_id] = fov_list  # Store FOVs under the sample ID
            else:
                print(f"Missing sample directory: {sample_path}")

        print(f"Total unique samples found: {len(samples)}")
        return samples  # Now returns a dict of sample_id -> list of FOVs

    def __len__(self):
        return len(self.samples)

    def tiff_to_tensor(self, file_path):
        img = tiff.imread(file_path)
        img_tensor = torch.tensor(img, dtype=torch.float)  # Convert NumPy array to tensor
        if img_tensor.dim() == 2:  # If grayscale, add channel dimension
            img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    def __getitem__(self, idx):
        fad_path, nadh_path, shg_path, orr_path, label = self.samples[idx]

        # Load images as tensors
        image_tensors = {
            'fad': self.tiff_to_tensor(fad_path),
            'nadh': self.tiff_to_tensor(nadh_path),
            'shg': self.tiff_to_tensor(shg_path),
            'orr': self.tiff_to_tensor(orr_path)
        }

        # Combine selected channels into a single image
        selected_tensors = [image_tensors[channel] for channel in self.channels]
        combined_image = torch.cat(selected_tensors, dim=0)

        combined_image[torch.isnan(combined_image)] = 0

        # Check for NaN values inside data
        if torch.isnan(combined_image).any() or torch.isinf(combined_image).any():
            print(f"Warning: NaN or Inf detected in image at index {idx}")

        # Apply transformations if provided
        if self.transform:
            combined_image = self.transform(combined_image)

        # Ensure the label is a tensor
        label_tensor = torch.tensor(label, dtype=torch.float)

        if self.label == 'classification':
            label_tensor = torch.tensor(1 if label_tensor >= 25 else 0, dtype=torch.float32)

        return combined_image, label_tensor
