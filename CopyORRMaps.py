import os
import shutil
import re
from pathlib import Path


def copy_ORR_maps(source_base, dest_base):
    """
    Copies ORR map JPG files into the correct SampleXXX/fovY folders based on filename.

    :param source_base: Root directory containing date folders with ORR maps.
    :param dest_base: Destination directory where files should be organized.
    """
    dest_base = Path(dest_base)
    dest_base.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"fov(\d+)colorORRMapUniform\.jpg")

    for date_folder in sorted(Path(source_base).iterdir()):
        if date_folder.is_dir() and date_folder.name[:2].isdigit():
            print(f"Processing folder: {date_folder}")

            for file in date_folder.rglob("fov*colorORRMapUniform.jpg"):
                match = pattern.search(file.name)
                if match:
                    fov_number = match.group(1)
                    sample_number = "Sample" + date_folder.name.replace("_", "")  # Extract SampleXXX from folder name

                    destination_dir = dest_base / sample_number / fov_number
                    destination_dir.mkdir(parents=True, exist_ok=True)

                    destination_path = destination_dir / file.name
                    shutil.copy2(file, destination_path)
                    print(f"Copied: {file} -> {destination_path}")


if __name__ == "__main__":
    source_directory = r"\\deckard\BMEG\Rajaram-Lab\Powell, Nicholas\Highlands Project\Imaging\Highlands FFPE Samples"
    destination_directory = r"C:\Users\nmp002\PycharmProjects\HighlandsMachineLearning\data"

    copy_ORR_maps(source_directory, destination_directory)
