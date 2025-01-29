import os
import shutil
import re
from pathlib import Path


def copy_ORR_maps(source_base, dest_base):
    """
    Copies ORR map JPG files into the correct existing SampleXXX/fovY folders based on directory structure.

    :param source_base: Root directory containing date folders with ORR maps.
    :param dest_base: Destination directory where files should be organized.
    """
    dest_base = Path(dest_base)
    dest_base.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"Sample_(\d+)\\fov(\d+)")

    for date_folder in sorted(Path(source_base).iterdir()):
        if date_folder.is_dir() and date_folder.name[:2].isdigit():
            print(f"Processing folder: {date_folder}")

            for file in date_folder.rglob("fov*colorORRMapUniform.jpg"):
                match = pattern.search(str(file.parent))
                if match:
                    sample_number = f"Sample{match.group(1)}"
                    fov_number = f"fov{match.group(2)}"

                    destination_dir = dest_base / sample_number / fov_number
                    if destination_dir.exists():  # Only copy if the destination folder exists
                        destination_path = destination_dir / file.name
                        shutil.copy2(file, destination_path)
                        print(f"Copied: {file} -> {destination_path}")
                    else:
                        print(f"Skipping: {file} (Destination folder {destination_dir} does not exist)")


if __name__ == "__main__":
    source_directory = r"\\deckard\BMEG\Rajaram-Lab\Powell, Nicholas\Highlands Project\Imaging\Highlands FFPE Samples"
    destination_directory = r"C:\\Users\\nmp002\\PycharmProjects\\HighlandsMachineLearning\\data"

    copy_ORR_maps(source_directory, destination_directory)
