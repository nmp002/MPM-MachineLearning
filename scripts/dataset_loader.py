import torch
from torch.utils.data import Dataset
import tifffile as tiff
import pandas as pd
import os

def tiff_to_tensor(file_path):
    img = tiff.imread(file_path)
    img_tensor = torch.tensor(img, dtype=torch.float)  # Convert NumPy array to tensor
    if img_tensor.dim() == 2:  # If grayscale, add channel dimension
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

class MicroscopyDataset(Dataset):
    def __init__(self, csv_file, root_dir,  channels=('fad','nadh','shg','orr'), label='classification', transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label = label
        self.channels = channels
        self.sample_wise_paths = []
        self.sample_n = 0

        # For each sample in the data frame
        for _, row in self.data_frame.iterrows():
            sample_id = row['sample_id']
            score = row['recurrence_score']
            sample_path = os.path.join(self.root_dir, f"{sample_id}")
            if os.path.exists(sample_path):
                # For each potential FOV of the sample
                self.sample_wise_paths.append([])
                for fov in range(1, 6):
                    fov_dir = os.path.join(sample_path, f"fov{fov}")
                    if os.path.exists(fov_dir):
                        channel_paths = [os.path.join(fov_dir, f'{channel}.tiff') for channel in self.channels]
                        if all(os.path.exists(p) for p in channel_paths):
                            # Get each channel path
                            self.sample_wise_paths[-1].append((channel_paths, score))
                            self.sample_n += 1
                        else:
                            print(f"Missing images in {fov_dir}")
                    else:
                        print(f"Missing FOV directory: {fov_dir}")

            else:
                print(f"Missing sample directory: {sample_path}")
        print(f"Total samples found: {self.sample_n}")

    def __getitem__(self, index):
        # De-nest fov paths and get the indexed item path
        channel_paths, score = self._denest()[index]
        combined_image = torch.cat([tiff_to_tensor(channel) for channel in channel_paths], dim=0)
        return combined_image, score

    def _denest(self):
        return [p for path in self.sample_wise_paths for p in path]

    def __len__(self):
        return self.sample_n

    def get_sample_images(self, sample_id):
        indices = self.get_sample_indices(sample_id)
        sample_images = []
        score = None
        for index in indices:
            img, score = self[index]
            sample_images.append(img)
        return sample_images, score

    def get_sample_indices(self, sample_id):
        if isinstance(sample_id, int):
            sample_id = self.data_frame['sample_id'].iloc[sample_id]
        indices = []
        for i, fov_path in enumerate(self._denest()):
            if sample_id in fov_path[0][0]:
                indices.append(i)
        return indices
