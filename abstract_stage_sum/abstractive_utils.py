
import os
import json
# =========================
# Utility Functions
# =========================
def save_dataset(dataset, path, save_csv=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset.save_to_disk(path)
    if save_csv:
        dataset.to_csv(path + ".csv")
    print(f"[INFO] ✅ Dataset saved to {path}")


def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] ✅ Metrics saved to {path}")

