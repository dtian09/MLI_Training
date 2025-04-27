import numpy as np
from collections import Counter
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_punctuation,
    strip_numeric,
    remove_stopwords,
    strip_multiple_whitespaces,
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
            stem_text
        ]
        self.embedding_model = embedding_model
        self.unk_token_vector = unk_token_vector

    def preprocess(self, text):
        return preprocess_string(text, self.filters)

    def preprocess_corpus(self, corpus):
        return [self.preprocess(doc) for doc in corpus]

    def _initialize_unk_vector(self):
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

    def filter_corpus_by_word_coverage(self, corpus, p=95):
        """
        Filter the corpus to contain the most frequently occuring tokens 
        (the smallest set of tokens) whose cumulative frequency >= p% (e.g. 95%) of total words.
        """
        tokenized_corpus = self.preprocess_corpus(corpus)

        # Flatten and count frequencies
        all_tokens = [token for doc in tokenized_corpus for token in doc]
        total_word_count = len(all_tokens)
        token_freq = Counter(all_tokens)

        # Sort tokens by frequency descending
        sorted_tokens = token_freq.most_common()

        # Accumulate until reaching p% of total word count
        cumulative = 0
        most_frequent_tokens = set()
        for token, freq in sorted_tokens:
            cumulative += freq
            most_frequent_tokens.add(token)
            if (cumulative / total_word_count) * 100 >= p:
                break

        # Now filter corpus
        filtered_corpus = []
        filtered_word_count = 0
        for doc in tokenized_corpus:
            filtered_doc = [token for token in doc if token in most_frequent_tokens]
            if filtered_doc:
                filtered_corpus.append(filtered_doc)
                filtered_word_count += len(filtered_doc)

        # Report
        word_retention = (filtered_word_count / total_word_count) * 100
        print(f"[Info] Word retention after filtering: {word_retention:.2f}%")
        print(f"[Info] Number of most frequent unique tokens: {len(most_frequent_tokens)}")

        return filtered_corpus
    
    if __name__ == "__main__":
    from gensim.models import KeyedVectors

    # Load word vectors
    embeddings = KeyedVectors.load("msmarco_wordvectors.kv")
    preprocessor = TextPreprocessor(embedding_model=embeddings)

    # Example corpus
    corpus = [
        "Word2Vec rocks NLP.",
        "Efficient text processing.",
        "Natural Language Processing is amazing.",
        "Deep Learning is revolutionary.",
        "I love machine learning.",
        "Quantum AI will change the world."
    ]

    # 1. Filter corpus to achieve 95% word coverage
    filtered_corpus = preprocessor.filter_corpus_by_word_coverage(corpus, p=95)
    
    ''' output
    [Info] Word retention after filtering: 95.13%
    [Info] Number of allowed unique tokens: 18
    '''

    print("\nFiltered and tokenized corpus:")
    for doc in filtered_corpus:
        print(doc)
    Filtered and tokenized corpus:

    '''    
    output:

    Filtered and tokenized corpus:

    ['word2vec', 'rock', 'nlp']
    ['effici', 'text', 'process']
    ['natur', 'languag', 'process']
    ['deep', 'learn', 'revolutionari']
    ['love', 'machin', 'learn']
    ['quantum', 'ai', 'chang', 'world']
    '''
    
    # 2. Get embeddings
    corpus_vectors = preprocessor.get_corpus_embedding_pooled([' '.join(doc) for doc in filtered_corpus], pooling="mean")
    print(f"\nCorpus pooled embeddings: {[vec.shape for vec in corpus_vectors]}")

