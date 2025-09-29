
import argparse
import torch
import json
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from extractive_stage_sum.model_longformer import LongformerSummarizer
from abstract_stage_sum.abstractive_summarizer import load_model, summarizer, ABSTRACTIVE_PROMPT



def save_outputs(results, output_file):
    """Save results to JSON + CSV for easy reuse."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save JSON
    with open(output_file + ".json", "w") as f:
        json.dump(results, f, indent=4)

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file + ".csv", index=False)

    print(f"[INFO] ✅ Saved outputs to {output_file}.json and {output_file}.csv")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load Extractive Model ===
    tokenizer_ext = AutoTokenizer.from_pretrained(args.extractive_model_name, use_fast=False)
    base_model = AutoModel.from_pretrained(args.extractive_model_name).to(device)
    ext_model = LongformerSummarizer(base_model, tokenizer_ext, input_size=args.input_size).to(device)


    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ext_model.load_state_dict(checkpoint["model_state_dict"])
    ext_model.eval()



    # === Load Abstractive Model ===
    abs_model, abs_tokenizer = load_model(args.abstractive_model_name)
    abs_model.eval()
    with open(args.input_file, "r") as f:
        document = f.read()

    # === Extractive Summary ===
    ext_sentences, _ = ext_model.summarize(document, strategy="count", strategy_args=args.num_sentences)
    extractive_summary = "\n".join(ext_sentences)



    # === Abstractive Summary ===
    abstractive_summary = summarizer(extractive_summary, ABSTRACTIVE_PROMPT, abs_model, abs_tokenizer, device)

    # === Package Results ===
    results = {
        "article": document,
        "extractive_summary": extractive_summary,
        "abstractive_summary": abstractive_summary,
    }

    # Print to console
    print("📄 Extractive Summary:\n", extractive_summary)
    print("\n📝 Abstractive Summary:\n", abstractive_summary)

    # Save to disk
    save_outputs([results], args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Inference: Extractive + Abstractive Summaries")
    parser.add_argument("--extractive_model_name", type=str, default="allenai/longformer-base-4096")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained extractive checkpoint")
    parser.add_argument("--input_file", type=str, required=True, help="Path to text document")
    parser.add_argument("--input_size", type=int, default=4096)
    parser.add_argument("--num_sentences", type=int, default=20, help="Number of extractive sentences")
    parser.add_argument("--abstractive_model_name", type=str,
                        default="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
                        help="HuggingFace/Unsloth model for abstractive summarization")
    parser.add_argument("--output_file", type=str, default="./outputs/inference_results",
                        help="Path prefix to save results (.json, .csv)")

    args = parser.parse_args()
    main(args)