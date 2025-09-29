import torch
from tqdm import tqdm
from datasets import Dataset

from extractive_stage_sum.transformer_utils import padToSize


class LongformerDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for Longformer-based extractive summarization.
    Converts sentences + labels into tensors with masks ready for the model.
    """
    def __init__(self, ext_dataset, tokenizer, input_size=4096):
        self.labels = []
        self.documents = []

        for data in tqdm(ext_dataset, desc="Loading dataset"):
            labels = [1 if label else 0 for label in data["labels"]]
            ids = data["doc_ids"]

            # Masks
            clss_mask = [False for _ in range(input_size)]
            for i in data["cls_idxs"]:
                clss_mask[i] = True

            attn_mask = [1 for _ in range(len(data["doc_ids"]))]
            global_attn_mask = [0 for _ in range(input_size)]
            for i in data["cls_idxs"]:
                global_attn_mask[i] = 1

            # Store tensors
            self.labels.append(torch.tensor(padToSize(labels, input_size, 0)))
            self.documents.append({
                "ids": torch.tensor(padToSize(ids, input_size, tokenizer.pad_token_id)),
                "clss_mask": torch.tensor(clss_mask),
                "attn_mask": torch.tensor(padToSize(attn_mask, input_size, 0)).unsqueeze(0),
                "global_attn_mask": torch.tensor(global_attn_mask).unsqueeze(0),
                "ref_summary": data["ref_summary"],
                "num_sentences": len(data["labels"]),
            })

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.documents[idx], self.labels[idx]


# -------------------------------
# Parsing utilities
# -------------------------------
def _parseForRoBERTa(sentences, labels, tokenizer, max_tokens=512):
    """
    Parse sentences for RoBERTa-style models.
    Returns token IDs, CLS indexes, and aligned labels.
    """
    doc_tokens = []
    cls_idxs = []
    total_tokens = 0

    for i, sent in enumerate(sentences):
        tokens = [tokenizer.cls_token] + tokenizer.tokenize(sent) + [tokenizer.sep_token]
        if total_tokens + len(tokens) > max_tokens:
            break
        cls_idxs.append(len(doc_tokens))  # position of [CLS]
        doc_tokens.extend(tokens)
        total_tokens += len(tokens)

    doc_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
    labels = labels[:len(cls_idxs)]
    return doc_ids, cls_idxs, labels


def _parseForLongformer(sentences, labels, tokenizer, max_tokens=4096):
    """
    Wrapper for Longformer preprocessing.
    """
    return _parseForRoBERTa(sentences, labels, tokenizer, max_tokens)
