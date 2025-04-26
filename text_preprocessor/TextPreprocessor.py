import numpy as np
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_punctuation,
    strip_numeric,
    remove_stopwords,
    strip_multiple_whitespaces,
    strip_short,
    stem_text
)

class TextPreprocessor:
    def __init__(self, embedding_model=None, unk_token_vector=None):
        """
        embedding_model: Gensim model (e.g., Word2Vec) or KeyedVectors
        unk_token_vector: Optional. If None, will compute mean vector automatically.
        """
        self.filters = [
            lambda x: x.lower(),
            strip_punctuation,
            strip_numeric,
            remove_stopwords,
            strip_multiple_whitespaces,
            strip_short,
            stem_text
        ]
        self.embedding_model = embedding_model
        self.unk_token_vector = unk_token_vector

    def preprocess(self, text):
        return preprocess_string(text, self.filters)

    def preprocess_corpus(self, corpus):
        return [self.preprocess(doc) for doc in corpus]

    def token_known(self, token):
        return token in self.embedding_model

    def count_unknown_tokens(self, tokens):
        return sum(1 for token in tokens if not self.token_known(token))

    def filter_corpus(self, corpus, max_unknowns=0):
        cleaned_corpus = []
        for doc in corpus:
            tokens = self.preprocess(doc)
            unknown_count = self.count_unknown_tokens(tokens)
            if tokens and unknown_count <= max_unknowns:
                cleaned_corpus.append(tokens)
        return cleaned_corpus

    def _initialize_unk_vector(self):
        """
        Lazy initialization of UNK vector based on mean of all word vectors.
        """
        if self.embedding_model is None:
            raise ValueError("Cannot initialize UNK vector without embedding model.")

        if self.unk_token_vector is None:
            print("[Info] Initializing UNK vector as mean of all word vectors.")
            self.unk_token_vector = np.mean(self.embedding_model.vectors, axis=0)

    def get_token_embedding(self, token):
        if self.embedding_model is None:
            raise ValueError("No embedding model provided.")

        if token in self.embedding_model:
            return self.embedding_model[token]
        else:
            if self.unk_token_vector is None:
                self._initialize_unk_vector()
            return self.unk_token_vector

    def get_text_embeddings(self, text):
        tokens = self.preprocess(text)
        embeddings = []
        for token in tokens:
            vector = self.get_token_embedding(token)
            embeddings.append(vector)
        return embeddings

    def get_corpus_embeddings(self, corpus):
        return [self.get_text_embeddings(doc) for doc in corpus]

    def get_text_embedding_pooled(self, text, pooling="mean"):
        embeddings = self.get_text_embeddings(text)
        if not embeddings:
            return np.zeros(self.embedding_model.vector_size)
        
        embeddings = np.array(embeddings)
        if pooling == "mean":
            return embeddings.mean(axis=0)
        elif pooling == "sum":
            return embeddings.sum(axis=0)
        else:
            raise ValueError("Unsupported pooling type. Choose 'mean' or 'sum'.")

    def get_corpus_embedding_pooled(self, corpus, pooling="mean"):
        return [self.get_text_embedding_pooled(doc, pooling) for doc in corpus]

if __name__ == "__main__":
   
import torch

# Initialize preprocessor
embeddings = torch.load("msmarco_wordvectors.kv")
preprocessor = TextPreprocessor(embedding_model=embeddings)

# Example text
text = "Gensim is efficient at text processing!"

# Get token-level embeddings
embeddings = preprocessor.get_text_embeddings(text)
print(f"Found {len(embeddings)} token embeddings.")

# Get pooled vector
pooled_vector = preprocessor.get_text_embedding_pooled(text, pooling="mean")
print(f"Pooled vector shape: {pooled_vector.shape}")

# Corpus example
corpus = [
    "Word2Vec rocks NLP.",
    "Efficient text processing."
]
corpus_vectors = preprocessor.get_corpus_embedding_pooled(corpus, pooling="mean")
print(f"Corpus pooled embeddings: {[vec.shape for vec in corpus_vectors]}")
