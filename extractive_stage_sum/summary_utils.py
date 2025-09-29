
import torch

def padToSize(to_pad_list, pad_size, filler):
    """Pad a list to a given size with filler tokens."""
    return to_pad_list + [filler] * (pad_size - len(to_pad_list))



import torch

def custom_collate_fn(batch, tokenizer):
    max_doc_ids_len = max(len(item['doc_ids']) for item in batch)
    max_labels_len = max(len(item['labels']) for item in batch)
    max_cls_idxs_len = max(len(item['cls_idxs']) for item in batch)

    padded_batch = {
        'id': [item['id'] for item in batch],
        'ref_summary': [item['ref_summary'] for item in batch],
        'labels': torch.stack([
            torch.cat([torch.tensor(item['labels'], dtype=torch.float),
                       torch.zeros(max_labels_len - len(item['labels']), dtype=torch.float)])
            for item in batch
        ]),
        'ids': torch.stack([
            torch.cat([torch.tensor(item['doc_ids']),
                       torch.tensor([tokenizer.pad_token_id] * (max_doc_ids_len - len(item['doc_ids'])))])
            for item in batch
        ]),
        'cls_idxs': torch.stack([
            torch.cat([torch.tensor(item['cls_idxs']),
                       torch.tensor([-1] * (max_cls_idxs_len - len(item['cls_idxs'])))])
            for item in batch
        ])
    }
    return padded_batch, padded_batch['labels']

def _selectStrategyLength(sentences, predictions, max_length):
    selected_sents, summary_len = [], 0
    sents_priority = torch.argsort(predictions, descending=True)
    i = 0
    while (summary_len < max_length) and (i < len(sents_priority)):
        if summary_len + len(sentences[sents_priority[i]]) < max_length:
            selected_sents.append(sents_priority[i])
            summary_len += len(sentences[sents_priority[i]])
        i += 1
    return sorted(selected_sents)

def _selectStrategyCount(sentences, predictions, num_sents):
    return sorted(torch.topk(predictions, min(len(predictions), num_sents)).indices)

def _selectStrategyRatio(sentences, predictions, ratio):
    doc_length = sum(len(sent) for sent in sentences)
    return _selectStrategyLength(sentences, predictions, doc_length*ratio)

def _selectStrategyThreshold(sentences, predictions, threshold):
    return [i for i, score in enumerate(predictions) if score >= threshold]

def select(sentences, predictions, strategy, strategy_args):
    if strategy == "length":
        selected_sents = _selectStrategyLength(sentences, predictions, strategy_args)
    elif strategy == "count":
        selected_sents = _selectStrategyCount(sentences, predictions, strategy_args)
    elif strategy == "ratio":
        selected_sents = _selectStrategyRatio(sentences, predictions, strategy_args)
    elif strategy == "threshold":
        selected_sents = _selectStrategyThreshold(sentences, predictions, strategy_args)
    else:
        raise NotImplementedError(f"Unknown strategy {strategy}")
    return [sentences[i] for i in selected_sents], selected_sents

def splitDocument(doc_tokens, bos_token, eos_token, max_size):
    def _findNextBOSFrom(start_idx):
        for i in range(start_idx, len(doc_tokens)):
            if bos_token and doc_tokens[i] == bos_token:
                return i
            if not bos_token and doc_tokens[i] == eos_token:
                return i+1
        return -1
    def _findPreviousEOSFrom(start_idx):
        for i in range(start_idx, -1, -1):
            if doc_tokens[i] == eos_token:
                return i
        return -1

    chunks = []
    while len(doc_tokens) > max_size:
        eos_idx = _findPreviousEOSFrom(max_size - 1)
        if eos_idx == -1:
            next_bos_idx = _findNextBOSFrom(max_size)
            if next_bos_idx != -1:
                doc_tokens = doc_tokens[:max_size-1] + [eos_token] + doc_tokens[next_bos_idx:]
            else:
                doc_tokens = doc_tokens[:max_size-1] + [eos_token]
            eos_idx = max_size - 1
        chunks.append(doc_tokens[:eos_idx+1])
        doc_tokens = doc_tokens[eos_idx+1:]
    if doc_tokens: chunks.append(doc_tokens)
    return chunks