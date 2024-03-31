import os
import torch
from torchvision import transforms
import pandas as pd
import cv2


root_path = os.path.join(os.getcwd())
custom_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
])


def split_data(data_path, train_ratio=0.8):
    data = torch.load(data_path)
    data_size = len(data["targets"])

    # shuffle the data (note: data["images"] and data["targets"] are lists)

    zips = list(zip(data["images"], data["targets"]))
    import random
    random.shuffle(zips)

    data["images"], data["targets"] = zip(*zips)

    train_size = int(data_size * train_ratio)
    test_size = data_size - train_size

    train_data = {
        "images": data["images"][:train_size],
        "targets": data["targets"][:train_size],
        "classes": data["classes"]
    }

    test_data = {
        "images": data["images"][train_size:],
        "targets": data["targets"][train_size:],
        "classes": data["classes"]
    }

    return train_data, test_data

def preprocessing_amd():
    print("Preprocessing AMD dataset...")
    data_path = os.path.join(root_path, "AMD")

    data = {
        "images": [],
        "targets": [],
        "classes": ["Non-AMD", "AMD"]
    }

    for folder in ["AMD", "Non-AMD"]:
        folder_path = os.path.join(data_path, folder)
        label = 1 if folder == "AMD" else 0

        for image_name in os.listdir(folder_path):
            if not image_name.endswith(".jpg"):
                continue
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            image = custom_transform(image)

            data["images"].append(image)
            data["targets"].append(label)
    
    print(f"Preprocessing AMD dataset done! {len(data['images'])} images found.")
    print("Saving preprocessed data...")

    torch.save(data, os.path.join(data_path, "preprocessed.pt"))

    print("Splitting data into train and test set...")

    train_data, test_data = split_data(os.path.join(data_path, "preprocessed.pt"))

    torch.save(train_data, os.path.join(data_path, "train.pt"))
    torch.save(test_data, os.path.join(data_path, "test.pt"))


def preprocessing_palm():
    print("Preprocessing PALM dataset...")

    data_path = os.path.join(root_path, "PALM")

    data = {
        "images": [],
        "targets": [],
        "classes": ["PM", "HiPM", "Normal"]
    }

    folder_path = os.path.join(data_path, "PALM")

    for image_name in os.listdir(folder_path):
        if not image_name.endswith(".jpg"):
            continue
        image_path = os.path.join(folder_path, image_name)
        
        image = cv2.imread(image_path)
        image = custom_transform(image)

        if image_name.startswith("P"):
            label = 0
        elif image_name.startswith("H"):
            label = 1
        else:
            label = 2

        data["images"].append(image)
        data["targets"].append(label)
    
    print(f"Preprocessing PALM dataset done! {len(data['images'])} images found.")
    print("Saving preprocessed data...")

    torch.save(data, os.path.join(data_path, "preprocessed.pt"))

    print("Splitting data into train and test set...")
    train_data, test_data = split_data(os.path.join(data_path, "preprocessed.pt"))

    torch.save(train_data, os.path.join(data_path, "train.pt"))
    torch.save(test_data, os.path.join(data_path, "test.pt"))

def preprocessing_uwf():
    print("Preprocessing UWF dataset...")
    data_path = os.path.join(root_path, "UWF")

    train_data = {
        "images": [],
        "targets": [],
        "classes": ["normal", "mild", "moderate", "severe", "PDR"]
    }

    test_data = {
        "images": [],
        "targets": [],
        "classes": ["normal", "mild", "moderate", "severe", "PDR"]
    }

    for folder in ["ultra-widefield-training", "ultra-widefield-validation"]:
        folder_path = os.path.join(data_path, folder, "Images")

        labels = pd.read_csv(os.path.join(data_path, folder, f"{folder}.csv"))

        for i, row in labels.iterrows():
            image_name = row["image_path"]
            # remove first folder name
            image_name = os.path.join(*image_name.split("\\")[1:])
            image_name = os.path.join(folder_path, image_name)

            if not image_name.endswith(".jpg"):
                continue

            image = cv2.imread(image_name)
            image = custom_transform(image)

            label = row["DR_level"]

            if label >= 5:
                continue

            if folder == "ultra-widefield-training":
                train_data["images"].append(image)
                train_data["targets"].append(label)
            else:
                test_data["images"].append(image)
                test_data["targets"].append(label)
    
    data = {
        "images": train_data["images"] + test_data["images"],
        "targets": train_data["targets"] + test_data["targets"],
        "classes": train_data["classes"]
    }

    print(f"Preprocessing UWF dataset done! {len(data['images'])} images found.")
    print("Saving preprocessed data...")

    torch.save(data, os.path.join(data_path, "preprocessed.pt"))

    torch.save(train_data, os.path.join(data_path, "train.pt"))
    torch.save(test_data, os.path.join(data_path, "test.pt"))


if __name__ == '__main__':
    preprocessing_amd()

    preprocessing_palm()

    preprocessing_uwf()
