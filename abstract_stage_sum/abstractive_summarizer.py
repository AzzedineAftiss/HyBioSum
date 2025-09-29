
import argparse
import torch
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from unsloth import FastLanguageModel
from metrics.rouge import evalROUGE
import json
import os

# =========================
# Load Model and Tokenizer
# =========================
def load_model(model_name, max_seq_length=2048, dtype=None, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


# =========================
# Prompt Template
# =========================
ABSTRACTIVE_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Summarize the following biomedical research article into a short summary.
Your summary should include:

1. The primary research question or objective.
2. A brief description of the study design and key methods.
3. The major findings or results.
4. The conclusions and implications.


### Input:
{}

### Response:
"""

# =========================
# Summarizer Function
# =========================
def summarizer(sample, prompt, model, tokenizer, device="cuda"):
    print(f"---------------------> The type of sample is : {type(sample)}")
    if isinstance(sample, str):
      article = sample
    else:
      article = sample["extractive_summaries"]

    inputs = tokenizer(
        [prompt.format(article)],
        return_tensors="pt"
    ).to(device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        use_cache=True
    )

    clean_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    if "### Response" in clean_text:
        return clean_text.split("### Response")[1].strip()
    return clean_text.strip()


# =========================
# Run Abstractive Summarization
# =========================
def run_abstractive(dataset, prompt, model, tokenizer, device="cuda", model_tag="mistral"):
    generated_summaries, reference_summaries, articles = [], [], []

    for sample in tqdm(dataset, desc=f"Generating {model_tag} summaries"):
        generated = summarizer(sample, prompt, model, tokenizer)
        generated_summaries.append(generated)
        reference_summaries.append(sample.get("abstract", ""))
        articles.append(sample["extractive_summaries"])

    '''data_dict = {
        "article": articles,
        "abstract": reference_summaries,
        "abstractive_summary": generated_summaries,
    }
    return Dataset.from_dict(data_dict)
'''
    return generated_summaries, reference_summaries
