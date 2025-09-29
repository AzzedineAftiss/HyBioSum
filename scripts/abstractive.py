
import argparse
import torch
from datasets import load_from_disk
from tqdm import tqdm

from abstract_stage_sum.abstractive_summarizer import load_model, summarizer, ABSTRACTIVE_PROMPT, run_abstractive

from abstract_stage_sum.abstractive_utils import save_dataset, save_metrics
from metrics.rouge import evalROUGE

import os


# =========================
# Main
# =========================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading model...")
    model, tokenizer = load_model(args.model_name)

    print("[INFO] Loading dataset...")
    dataset = load_from_disk(args.dataset_path)

    print("[INFO] Running summarization...")
    # === Run summarization ===
    model_tag = args.model_name.split("/")[-1].replace("-", "_")

    if "validation" in dataset:
        generated_summaries, reference_summaries = run_abstractive(
            dataset["validation"], ABSTRACTIVE_PROMPT, model, tokenizer, device, model_tag=model_tag
        )
        dataset["validation"] = dataset["validation"].add_column(f"{model_tag}_generated_summaries",
                                                                 generated_summaries)
    else:

        generated_summaries, reference_summaries = run_abstractive(
            dataset, ABSTRACTIVE_PROMPT, model, tokenizer, device, model_tag=model_tag
        )
        dataset = dataset.add_column(f"{model_tag}_generated_summaries", generated_summaries)

    print(f"[INFO] Saving results to {args.output_path}")
    # === Save new dataset with generated column ===

    save_dataset(dataset, args.output_path)

    print(f"[INFO] Calculating evaluation metrics......")
    # === Compute metrics ===
    rouge_scores = evalROUGE(reference_summaries, generated_summaries)
    metrics = {"rouge": rouge_scores}
    save_metrics(metrics, os.path.join(args.metrics_dir, f"{model_tag}_abstractive_results.json"))

    print("[INFO] ✅ Abstractive summarization completed.")
    print("ROUGE:", rouge_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abstractive Summarizer with Mistral/Unsloth")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to extractive dataset (HF format)")
    parser.add_argument("--output_path", type=str, default="./datasets/pubmed_abstractive",
                        help="Path to save new dataset")
    parser.add_argument("--metrics_dir", type=str, default="./metrics", help="Where to save metrics")
    parser.add_argument("--model_name", type=str, default="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
                        help="Model name on HuggingFace/Unsloth")

    args = parser.parse_args()
    main(args)