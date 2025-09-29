import torch
import numpy as np

def recall(labels, predictions):
    recalls = []
    for i in range(len(labels)):  # batch
        ref_idxs = (labels[i] == 1).nonzero(as_tuple=True)[0]
        if len(ref_idxs) == 0:
            continue
        sel_idxs = sorted(torch.topk(predictions[i], len(ref_idxs)).indices)
        ref_ptr, sel_ptr, correct = 0, 0, 0
        while ref_ptr < len(ref_idxs) and sel_ptr < len(sel_idxs):
            if ref_idxs[ref_ptr] == sel_idxs[sel_ptr]:
                correct += 1
                ref_ptr += 1
                sel_ptr += 1
            elif ref_idxs[ref_ptr] > sel_idxs[sel_ptr]:
                sel_ptr += 1
            else:
                ref_ptr += 1
        recalls.append(correct / len(ref_idxs))
    return np.average(recalls)