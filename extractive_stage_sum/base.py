
import torch
import spacy
import itertools
from extractive_stage_sum.summary_utils import select, splitDocument

class BaseSummarizer(torch.nn.Module):
    def __init__(self, model_name, input_size):
        super().__init__()
        self._doc2sentences = None
        self.model_name = model_name
        self.input_size = input_size
        self.tokenizer = None

    def forward(self, batch):
        raise NotImplementedError

    def predictChunk(self, chunk_tokens):
        raise NotImplementedError

    def summarizeFromDataset(self, predictions, doc_ids, summary_size):
        doc_ids = [id for id in doc_ids if id != self.tokenizer.pad_token_id]
        doc_sentences = [
            self.tokenizer.decode(list(ids)[:-1])
            for x, ids in itertools.groupby(doc_ids, lambda id: id == self.tokenizer.cls_token_id)
            if not x
        ]
        return self.summarizeSentences(doc_sentences, "count", summary_size, predictions=predictions)

    def predict(self, sentences):
        doc_tokens = self.tokenizer.tokenize(
            f"{self.tokenizer.sep_token}{self.tokenizer.cls_token}".join(sentences)
        )
        doc_tokens = [self.tokenizer.cls_token] + doc_tokens + [self.tokenizer.sep_token]
        doc_chunks = splitDocument(
            doc_tokens, self.tokenizer.cls_token, self.tokenizer.sep_token, self.input_size
        )
        predictions = torch.as_tensor([]).to(next(self.parameters()).device)
        for chunk in doc_chunks:
            chunk_preds = self.predictChunk(chunk)
            predictions = torch.cat((predictions, chunk_preds))
        return predictions

    def summarizeSentences(self, sentences, strategy="ratio", strategy_args=0.3, predictions=None):
        if predictions is None:
            predictions = self.predict(sentences)
        return select(sentences, predictions, strategy, strategy_args)

    def summarize(self, document, strategy="ratio", strategy_args=0.3):
        if self._doc2sentences is None:
            self._doc2sentences = spacy.load("en_core_web_sm")
        doc_sentences = [sent.text for sent in self._doc2sentences(document).sents]
        return self.summarizeSentences(doc_sentences, strategy, strategy_args)
