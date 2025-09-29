from transformers import AutoTokenizer, \
    BertTokenizer, BertTokenizerFast, \
    RobertaTokenizer, RobertaTokenizerFast, \
    LongformerTokenizer, LongformerTokenizerFast


from datasets import Dataset

def _parseForRoBERTa(sentences, labels, tokenizer, max_tokens=512):
    doc_tokens, cls_idxs = [], []
    total_tokens = 0
    for i, sent in enumerate(sentences):
        tokens = [tokenizer.cls_token] + tokenizer.tokenize(sent) + [tokenizer.sep_token]
        if total_tokens + len(tokens) > max_tokens:
            break
        cls_idxs.append(len(doc_tokens))  # position of [CLS] token
        doc_tokens.extend(tokens)
        total_tokens += len(tokens)
    doc_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
    labels = labels[:len(cls_idxs)]
    return doc_ids, cls_idxs, labels

def _parseForLongformer(sentences, labels, tokenizer, max_tokens=4096):
    return _parseForRoBERTa(sentences, labels, tokenizer, max_tokens)

def preprocessUtilitiesLongformer(tokenizer):
    def parseDataset(data):
        doc_ids, cls_idxs, labels = _parseForLongformer(
            data["sentences"], data["labels"], tokenizer
        )
        return {"__doc_ids": doc_ids, "__cls_idxs": cls_idxs, "__labels": labels}

    def filterDataset(dataset):
        dataset_content = {
            "id": dataset["id"],
            "ref_summary": dataset["ref_summary"],
            "labels": dataset["__labels"],
            "doc_ids": dataset["__doc_ids"],
            "cls_idxs": dataset["__cls_idxs"],
        }
        return Dataset.from_dict(dataset_content)

    return parseDataset, filterDataset

def loadPreprocessUtilities(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if isinstance(tokenizer, (BertTokenizer, BertTokenizerFast)):
        raise NotImplementedError("BERT preprocess not yet implemented")
    elif isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
        return preprocessUtilitiesLongformer(tokenizer)  # Reuse RoBERTa logic
    elif isinstance(tokenizer, (LongformerTokenizer, LongformerTokenizerFast)):
        return preprocessUtilitiesLongformer(tokenizer)
    raise NotImplementedError(f"Tokenizer type for {model_name} not supported")