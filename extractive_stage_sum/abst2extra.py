import spacy
from datasets import Dataset, DatasetDict
from rouge_score.rouge_scorer import RougeScorer

# Load spacy for sentence splitting (download with: python -m spacy download en_core_web_sm)
doc2sentences = spacy.load("en_core_web_sm")


def _abstractiveToExtractive(document: str, summary: str, extractive_size: int):
    """
    Convert abstractive summary into extractive labels using ROUGE recall.

    Args:
        document (str): Full document text
        summary (str): Gold abstractive summary
        extractive_size (int): Number of sentences to select

    Returns:
        doc_sentences (list[str])
        labels (list[bool])
    """
    doc_sentences = [sent.text for sent in doc2sentences(document).sents]
    summ_sentences = [sent.text for sent in doc2sentences(summary).sents]

    selected_sentences_idx = []
    max_rouge = 0.0
    scorer = RougeScorer(["rouge2"], use_stemmer=True)

    # at least as many as gold sentences
    extractive_size = max(extractive_size, len(summ_sentences))

    for _ in range(extractive_size):
        curr_selected_i = -1
        for i, sentence in enumerate(doc_sentences):
            if i in selected_sentences_idx:
                continue

            # Candidate summary if we add this sentence
            tmp_summary = " ".join([doc_sentences[j] for j in selected_sentences_idx] + [sentence])
            rouge_scores = scorer.score(summary, tmp_summary)

            rouge_with_sentence = rouge_scores["rouge2"].fmeasure

            # Only keep if improves
            if rouge_with_sentence > max_rouge + 1e-3:
                curr_selected_i = i
                max_rouge = rouge_with_sentence

        if curr_selected_i < 0:
            break
        selected_sentences_idx.append(curr_selected_i)

    labels = [(i in selected_sentences_idx) for i in range(len(doc_sentences))]
    return doc_sentences, labels


def parseAbs2Ext(dataset, selection_size: int = 3, num_proc: int = 1) -> DatasetDict:
    """
    Parse dataset from abstractive format to extractive.

    Args:
        dataset (DatasetDict): HuggingFace dataset with "article" and "abstract"
        selection_size (int): number of sentences to select
        num_proc (int): parallel processes for dataset.map()

    Returns:
        DatasetDict: with keys ["train", "validation", "test"] containing:
            - sentences
            - labels
            - ref_summary
            - id
    """

    def _parseDataset(data):
        document, summary = data["article"], data["abstract"]
        sents, labels = _abstractiveToExtractive(document, summary, extractive_size=selection_size)
        return {
            "__sentences": sents,
            "__labels": labels,
            "__summary": summary,
        }

    dataset = dataset.map(_parseDataset, num_proc=num_proc)

    def _filterDataset(dataset_split):
        dataset_content = {
            "id": [i for i in range(len(dataset_split["__sentences"]))],
            "sentences": dataset_split["__sentences"],
            "ref_summary": dataset_split["__summary"],
            "labels": dataset_split["__labels"],
        }
        return Dataset.from_dict(dataset_content)

    parsed_dataset = {
        "train": _filterDataset(dataset["train"]),
        "validation": _filterDataset(dataset["validation"]),
        "test": _filterDataset(dataset["test"]),
    }

    return DatasetDict(parsed_dataset)

