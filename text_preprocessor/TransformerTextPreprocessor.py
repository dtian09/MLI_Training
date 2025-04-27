from transformers import AutoTokenizer, AutoModel
import torch
from collections import Counter

class BERTTextPreprocessor:
    def __init__(self, model_name="bert-base-uncased", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def preprocess(self, text):
        return text.lower().strip()

    def get_text_embeddings(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        token_embeddings = outputs.last_hidden_state.squeeze(0)
        return token_embeddings  # (sequence_length, hidden_size)

    def get_text_embedding_pooled(self, text, pooling="mean"):
        token_embeddings = self.get_text_embeddings(text)

        if pooling == "mean":
            return token_embeddings.mean(dim=0)
        elif pooling == "cls":
            return token_embeddings[0]
        else:
            raise ValueError("Unsupported pooling method. Use 'mean' or 'cls'.")

    def tokenize_corpus(self, corpus):
        """
        Tokenizes a list of documents into token strings (not IDs).
        """
        tokenized_corpus = []
        for doc in corpus:
            preprocessed_doc = self.preprocess(doc)
            tokens = self.tokenizer.tokenize(preprocessed_doc)
            tokenized_corpus.append(tokens)
        return tokenized_corpus

    def filter_corpus_by_word_coverage(self, corpus, p=95):
        """
        - Tokenize the corpus
        - Find the smallest set of tokens covering at least p% of total words
        - Filter the corpus to only keep those tokens
        """
        tokenized_corpus = self.tokenize_corpus(corpus)

        # Flatten and count frequencies
        all_tokens = [token for doc in tokenized_corpus for token in doc]
        total_word_count = len(all_tokens)
        token_freq = Counter(all_tokens)

        # Sort tokens by frequency descending
        sorted_tokens = token_freq.most_common()

        # Accumulate until reaching p% of total word count
        cumulative = 0
        allowed_tokens = set()
        for token, freq in sorted_tokens:
            cumulative += freq
            allowed_tokens.add(token)
            if (cumulative / total_word_count) * 100 >= p:
                break

        # Now filter the tokenized corpus
        filtered_corpus = []
        filtered_word_count = 0
        for doc in tokenized_corpus:
            filtered_doc = [token for token in doc if token in allowed_tokens]
            if filtered_doc:
                filtered_corpus.append(filtered_doc)
                filtered_word_count += len(filtered_doc)

        # Report
        word_retention = (filtered_word_count / total_word_count) * 100
        print(f"[Info] Word retention after filtering: {word_retention:.2f}%")
        print(f"[Info] Number of allowed unique tokens: {len(allowed_tokens)}")

        return filtered_corpus

if __name__ == "__main__":
    preprocessor = BERTTextPreprocessor(model_name="bert-base-uncased", device="cuda")

    corpus = [
        "Gensim is great for word embeddings, but BERT is powerful for contextual embeddings.",
        "Natural Language Processing enables machines to understand human language.",
        "Transformers have revolutionized the field of deep learning.",
        "I love training BERT models for various NLP tasks!",
        "Transfer learning is an important technique in modern AI systems."
    ]

    # 1. Filter corpus to achieve 95% word coverage
    filtered_corpus = preprocessor.filter_corpus_by_word_coverage(corpus, p=95)

    print("\nFiltered and tokenized corpus:")
    for doc in filtered_corpus:
        print(doc)

    # 2. Example: get pooled embedding of a new sentence
    text = "BERT is a powerful tool for understanding language."
    pooled_vector = preprocessor.get_text_embedding_pooled(text, pooling="mean")
    print("\nPooled vector shape:", pooled_vector.shape)
