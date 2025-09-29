
import numpy as np

class MetricsLogger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = []
        self.total_recall = []
        self.total_rouge1 = {"fmeasure": [], "precision": [], "recall": []}
        self.total_rouge2 = {"fmeasure": [], "precision": [], "recall": []}
        self.total_rougeL = {"fmeasure": [], "precision": [], "recall": []}
        self.total_bertscore = {"fmeasure": [], "precision": [], "recall": []}

    def add(self, metric_type, value):
        if metric_type == "loss":
            self.total_loss.append(value)
        elif metric_type == "recall":
            self.total_recall.append(value)
        elif metric_type == "rouge":
            for key in ["rouge1", "rouge2", "rougeL"]:
                for m in ["fmeasure", "precision", "recall"]:
                    getattr(self, f"total_{key}")[m].append(value[key][m])
        elif metric_type == "bertscore":
            for m in ["fmeasure", "precision", "recall"]:
                self.total_bertscore[m].append(value[m])

    def averages(self):
        return {
            "loss": np.average(self.total_loss),
            "recall": np.average(self.total_recall),
            "rouge1": {m: np.average(self.total_rouge1[m]) for m in self.total_rouge1},
            "rouge2": {m: np.average(self.total_rouge2[m]) for m in self.total_rouge2},
            "rougeL": {m: np.average(self.total_rougeL[m]) for m in self.total_rougeL},
            "bertscore": {m: np.average(self.total_bertscore[m]) for m in self.total_bertscore},
        }

    def format(self, types):
        avgs = self.averages()
        out = "| "
        if "loss" in types:
            out += f"Loss {avgs['loss']:.5f} | "
        if "recall" in types:
            out += f"Recall {avgs['recall']:.5f} | "
        if "rouge" in types:
            out += (f"R-1 r: {avgs['rouge1']['recall']*100:.2f} -- p: {avgs['rouge1']['precision']*100:.2f} -- f1: {avgs['rouge1']['fmeasure']*100:.2f} | "
                    f"R-2 r: {avgs['rouge2']['recall']*100:.2f} -- p: {avgs['rouge2']['precision']*100:.2f} -- f1: {avgs['rouge2']['fmeasure']*100:.2f} | "
                    f"R-L r: {avgs['rougeL']['recall']*100:.2f} -- p: {avgs['rougeL']['precision']*100:.2f} -- f1: {avgs['rougeL']['fmeasure']*100:.2f} | ")
        if "bertscore" in types:
            out += (f"BS r: {avgs['bertscore']['recall']*100:.2f} -- "
                    f"p: {avgs['bertscore']['precision']*100:.2f} -- "
                    f"f1: {avgs['bertscore']['fmeasure']*100:.2f} | ")
        return out
