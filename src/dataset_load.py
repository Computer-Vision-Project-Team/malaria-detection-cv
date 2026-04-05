"""Dataset loading and splitting utilities for malaria classification.

Mirrors the notebook cells for downloading the NIH Malaria dataset from Kaggle
and performing a stratified 70/15/15 split using splitfolders.
"""

import os
import splitfolders


def download_dataset(dest_dir="data/raw"):
    """Download the NIH Malaria dataset from Kaggle and unzip it."""
    import subprocess

    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "cell-images-for-detecting-malaria.zip")

    print("Downloading NIH Malaria dataset...")
    subprocess.run(
        ["kaggle", "datasets", "download",
         "-d", "iarunava/cell-images-for-detecting-malaria",
         "-p", dest_dir],
        check=True,
    )

    print("Unzipping dataset...")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    print("Cleaning up zip file...")
    os.remove(zip_path)

    raw_data_dir = os.path.join(dest_dir, "cell_images", "cell_images")
    print(f"\nData extraction complete! Folders found: {os.listdir(raw_data_dir)}")
    return raw_data_dir


def split_dataset(input_folder, output_folder="data/processed", seed=42):
    """Perform a stratified 70/15/15 split using splitfolders."""
    print("Starting the stratified split (70% Train, 15% Val, 15% Test)...")
    print("This might take a minute or two as it processes over 27,000 images.\n")

    splitfolders.ratio(
        input_folder,
        output=output_folder,
        seed=seed,
        ratio=(0.70, 0.15, 0.15),
        group_prefix=None,
        move=False,
    )

    train_dir = os.path.join(output_folder, "train")
    val_dir = os.path.join(output_folder, "val")
    test_dir = os.path.join(output_folder, "test")

    print("Data splitting complete! New paths:")
    print(f"  -> Train directory: {train_dir}")
    print(f"  -> Val directory:   {val_dir}")
    print(f"  -> Test directory:  {test_dir}\n")

    # Stratification check
    print("--- Stratification Check ---")
    for dataset in ["train", "val", "test"]:
        for class_name in ["Parasitized", "Uninfected"]:
            path = os.path.join(output_folder, dataset, class_name)
            count = len([f for f in os.listdir(path) if f.endswith(".png")])
            print(f"{dataset.capitalize().ljust(5)} | {class_name.ljust(10)} : {count} images")

    return train_dir, val_dir, test_dir


if __name__ == "__main__":
    raw_dir = download_dataset()
    split_dataset(raw_dir)
