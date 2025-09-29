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
