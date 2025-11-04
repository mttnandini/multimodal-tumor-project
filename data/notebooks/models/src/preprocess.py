import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# ✅ Define image preprocessing steps
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ✅ Load datasets from folders
def load_datasets(base_dir, batch_size=16):
    radiology_path = os.path.join(base_dir, "radiology")
    pathology_path = os.path.join(base_dir, "pathology")

    radiology_dataset = datasets.ImageFolder(root=radiology_path, transform=transform)
    pathology_dataset = datasets.ImageFolder(root=pathology_path, transform=transform)

    # Split into training (80%) and validation (20%)
    radiology_train_len = int(0.8 * len(radiology_dataset))
    pathology_train_len = int(0.8 * len(pathology_dataset))
    radiology_val_len = len(radiology_dataset) - radiology_train_len
    pathology_val_len = len(pathology_dataset) - pathology_train_len

    radiology_train, radiology_val = random_split(radiology_dataset, [radiology_train_len, radiology_val_len])
    pathology_train, pathology_val = random_split(pathology_dataset, [pathology_train_len, pathology_val_len])

    # Create DataLoaders
    radiology_train_loader = DataLoader(radiology_train, batch_size=batch_size, shuffle=True)
    radiology_val_loader = DataLoader(radiology_val, batch_size=batch_size, shuffle=False)
    pathology_train_loader = DataLoader(pathology_train, batch_size=batch_size, shuffle=True)
    pathology_val_loader = DataLoader(pathology_val, batch_size=batch_size, shuffle=False)

    print("✅ Radiology and Pathology datasets loaded successfully!")
    print(f"Radiology samples: {len(radiology_dataset)} | Pathology samples: {len(pathology_dataset)}")

    return (radiology_train_loader, radiology_val_loader,
            pathology_train_loader, pathology_val_loader)
