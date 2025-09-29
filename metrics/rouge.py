from rouge_score.rouge_scorer import RougeScorer

from rouge_score.rouge_scorer import RougeScorer

def evalROUGE(ref_summaries, ext_summaries):
    total = {"rouge1": {"precision": 0, "recall": 0, "fmeasure": 0},
             "rouge2": {"precision": 0, "recall": 0, "fmeasure": 0},
             "rougeL": {"precision": 0, "recall": 0, "fmeasure": 0}}
    scorer = RougeScorer(["rouge1", "rouge2", "rougeL"])
    batch_size = len(ref_summaries)

    for i in range(batch_size):
        scores = scorer.score(ref_summaries[i], ext_summaries[i])
        for key in total:
            total[key]["precision"] += scores[key].precision
            total[key]["recall"] += scores[key].recall
            total[key]["fmeasure"] += scores[key].fmeasure

    return {key: {m: total[key][m] / batch_size for m in total[key]} for key in total}
