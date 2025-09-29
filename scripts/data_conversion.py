import argparse
from datasets import load_dataset, load_from_disk
from extractive_stage_sum.abst2extra import parseAbs2Ext

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Dataset preprocessing - Abstractive to Extractive"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["pubmed", "cord_19"],
        required=True,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory of dataset, if not specified, will be downloaded",
    )
    parser.add_argument("--output", type=str, help="Export directory")
    parser.add_argument("--proc", type=int, default=1, help="Number of processes")
    parser.add_argument(
        "--selection_size",
        type=int,
        default=3,
        help="Sentences to select (at least len(summary))",
    )
    parser.add_argument("--head", action="store_true", help="Show sample rows")

    args = parser.parse_args()

    # Load dataset (either from disk or huggingface hub)
    if args.dataset_dir:
        dataset = load_from_disk(args.dataset_dir)
    else:
        dataset = load_dataset(args.dataset)

    parsed_dataset = parseAbs2Ext(dataset, args.selection_size, args.proc)

    if args.head:
        print(parsed_dataset["train"][:3])

    if args.output:
        parsed_dataset.save_to_disk(args.output)
        print(f"✅ Parsed dataset saved to {args.output}")