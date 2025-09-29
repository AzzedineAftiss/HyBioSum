import argparse
import torch


import json
import os
from datasets import Dataset, load_from_disk

from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel, LongformerModel
from extractive_stage_sum.model_longformer import LongformerSummarizer
from extractive_stage_sum.summary_utils import custom_collate_fn
from extractive_stage_sum.data_utils import LongformerDataset
from extractive_stage_sum.model_longformer import LongformerSummarizer
from metrics.recall import recall
from metrics.rouge import evalROUGE




def save_dataset_with_extractives(dataset, extractive_summaries, output_dir):
    """Attach extractive summaries and save as HuggingFace dataset + CSV/JSON."""
    os.makedirs(output_dir, exist_ok=True)

    new_dataset = Dataset.from_dict({
       # "article": dataset["article"],
        "abstract": dataset["ref_summary"],
        "extractive_summaries": extractive_summaries
    })

    # Save HF dataset
    new_dataset.save_to_disk(os.path.join(output_dir, "extractive_dataset"))

    # Save CSV for inspection
    new_dataset.to_csv(os.path.join(output_dir, "extractive_dataset.csv"))

    print(f"[INFO] ✅ Dataset saved at {output_dir}/extractive_dataset (HF + CSV)")
    return new_dataset

def save_metrics(metrics, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_extractive_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] ✅ Metrics saved at {out_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load model + tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    base_model = AutoModel.from_pretrained(args.model_name)
    model = LongformerSummarizer(base_model, tokenizer, input_size=args.input_size).to(device)

    # === Load checkpoint ===
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # === Load dataset ===
    dataset = load_from_disk(args.dataset_dir)
    val_dataset = LongformerDataset(
        ext_dataset=dataset["validation"], tokenizer=tokenizer, input_size=args.input_size
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    total_ref, total_ext = [], []
    extractive_summaries = []

    # === Inference on validation set ===
    with torch.no_grad():
        for batch_documents, batch_labels in val_loader:
            batch_documents = {
                key: val.to(device) if torch.is_tensor(val) else val
                for key, val in batch_documents.items()
            }
            batch_labels = batch_labels.to(device)

            outputs, _ = model(batch_documents)

            for i in range(len(batch_labels)):
                # always select k=20
                # ext_size = min(args.selection_size, len((batch_labels[i] == 1).nonzero(as_tuple=True)[0]))  # top-20 sentences
                print(f"ext_size = {args.selection_size}")
                ext_sentences, _ = model.summarizeFromDataset(
                    outputs[i], batch_documents["ids"][i], args.selection_size
                )
                print(f"length of extracted sentencs : {len(ext_sentences)}")
                extractive_summary = "\nn".join(ext_sentences)

                extractive_summaries.append(extractive_summary)
                total_ref.append(batch_documents["ref_summary"][i])
                total_ext.append(extractive_summary)

    # === Compute metrics ===
    rouge_scores = evalROUGE(total_ref, total_ext)
    recall_score = recall(batch_labels, outputs)

    metrics = {
        "rouge": rouge_scores,
        "recall": recall_score
    }

    print("[INFO] ✅ Evaluation completed.")
    print("ROUGE:", rouge_scores)
    print("Recall:", recall_score)

    # === Save dataset with new column ===
    #dataset["validation"] = dataset["validation"].add_column("extractive_summaries", extractive_summaries)
    #save_dataset(dataset, args.output_dir)
    # Save dataset with extractive summaries
    output_dataset = save_dataset_with_extractives(dataset["validation"], total_ext, args.output_dir)

    # === Save metrics ===
    model_name = args.model_name.replace("/", "_")
    save_metrics(metrics, os.path.join(args.metrics_dir, "extractive_results.json"), model_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="allenai/longformer-base-4096")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--selection_size", type=int, default=20)
    parser.add_argument("--input_size", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, default="./datasets/pubmed_extractive")
    parser.add_argument("--metrics_dir", type=str, default="./metrics")

    args = parser.parse_args()
    main(args)
