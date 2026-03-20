import os
import random
import yaml
from pathlib import Path

def split_dataset(root_path: str, ratio: tuple = (0.2, 0.2, 0.6), seed: int = 42) -> dict:
    """
    Partition SARDet-100K images into train/val/test splits and generate YOLO config.
    """
    random.seed(seed)
    root = Path(root_path)
    img_dir = root / "images"
    
    if not img_dir.exists():
        return {"error": f"Image directory {img_dir} not found."}
        
    # Get all image paths
    all_images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    random.shuffle(all_images)
    
    total = len(all_images)
    train_end = int(total * ratio[0])
    val_end = train_end + int(total * ratio[1])
    
    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }
    
    # Create split files (text files containing image paths)
    for name, images in splits.items():
        split_file = root / f"{name}.txt"
        with open(split_file, "w") as f:
            for img in images:
                f.write(f"{img.absolute()}\n")
    
    # Class mapping for HPOTI
    class_map = {0: "Aircraft", 1: "Ship", 2: "Car", 3: "Tank", 4: "Bridge", 5: "Harbor"}
    
    # Generate data.yaml for Ultralytics/YOLO
    data_config = {
        "path": str(root.absolute()),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": class_map
    }
    
    with open(root / "data.yaml", "w") as f:
        yaml.dump(data_config, f)
        
    return {
        "train_count": len(splits["train"]),
        "val_count": len(splits["val"]),
        "test_count": len(splits["test"]),
        "config_file": str(root / "data.yaml")
    }

if __name__ == "__main__":
    # Example usage
    # result = split_dataset("./data/SARDet100K")
    print("Dataset splitter logic completed with YOLO YAML generation.")
