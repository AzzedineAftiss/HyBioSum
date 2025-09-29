
import argparse
from datasets import load_from_disk, DatasetDict
from extractive_stage_sum.preprocess_longformer import loadPreprocessUtilities

def main(args):
    # === Load dataset ===
    print("Loading dataset...")
    dataset = load_from_disk(args.dataset_dir)

    # === Load preprocessing utilities ===
    datasetMapFn, datasetFilterFn = loadPreprocessUtilities(args.model_name)

    # === Apply preprocessing ===
    print("Preprocessing...")
    dataset = dataset.map(datasetMapFn, num_proc=args.proc)

    parsed_dataset = {
        "train": datasetFilterFn(dataset["train"]),
        "test": datasetFilterFn(dataset["test"]),
        "validation": datasetFilterFn(dataset["validation"]),
    }
    parsed_dataset = DatasetDict(parsed_dataset)

    # === Save dataset ===
    parsed_dataset.save_to_disk(args.output_dir)
    print(f"Preprocessed dataset saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset for Longformer")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to raw dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save preprocessed dataset")
    parser.add_argument("--model_name", type=str, default="allenai/longformer-base-4096")
    parser.add_argument("--proc", type=int, default=2, help="Number of processes for dataset.map")
    args = parser.parse_args()
    main(args)

