import os
import shutil
import re

# Define the base and destination directories
base_dir = r"\\deckard\BMEG\Rajaram-Lab\Powell, Nicholas\Highlands Project\Imaging\Highlands FFPE Samples"
dest_dir = r"\\deckard\BMEG\Rajaram-Lab\Powell, Nicholas\Highlands Project\Imaging\Highlands FFPE Samples\data_for_ML_processing"

date_folders = ["05_27_2024", "06_05_2024", "06_18_2024", "08_13_2024", "08_14_2024"]

# Define mapping for renaming
file_mappings = {
    "Green": "fad",
    "UV": "shg",
    "Blue": "nadh"
}

# Process all 30 samples
for sample_num in range(1, 31):
    sample_folder = f"Sample_{str(sample_num).zfill(3)}"
    print(f"Processing {sample_folder}...")
    for date_folder in date_folders:
        date_path = os.path.join(base_dir, date_folder)
        sample_path = os.path.join(date_path, sample_folder)
        if os.path.isdir(sample_path):
            print(f"  Found {sample_folder} in {date_folder}")
            dest_sample_folder = os.path.join(dest_dir, f"Sample{str(sample_num).zfill(3)}")
            for fov_folder in os.listdir(sample_path):
                fov_path = os.path.join(sample_path, fov_folder)
                if os.path.isdir(fov_path) and fov_folder.startswith("fov"):
                    dest_fov_folder = os.path.join(dest_sample_folder, fov_folder)
                    for redox_folder in os.listdir(fov_path):
                        if re.match(r"redox_755(-\d+)?", redox_folder) or re.match(r"redox_855(-\d+)?", redox_folder):
                            redox_path = os.path.join(fov_path, redox_folder, "References")
                            if os.path.exists(redox_path):
                                for file in os.listdir(redox_path):
                                    for keyword, new_name in file_mappings.items():
                                        if keyword in file and file.endswith(".tif"):
                                            old_file_path = os.path.join(redox_path, file)
                                            new_file_name = f"{new_name}.tif"
                                            new_file_path = os.path.join(dest_fov_folder, new_file_name)
                                            os.makedirs(dest_fov_folder, exist_ok=True)
                                            shutil.copy2(old_file_path, new_file_path)
                                            print(f"    Copied {old_file_path} to {new_file_path}")
print("Data extraction and organization complete.")
