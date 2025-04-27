from transformers import AutoTokenizer, AutoModel
import torch

class BERTTextPreprocessor:
    def __init__(self, model_name="bert-base-uncased", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def preprocess(self, text):
        return text.lower().strip()  # Basic cleaning (could be extended)

    def get_text_embeddings(self, text):
        """
        Get token embeddings from BERT for the input text.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # outputs.last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        token_embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension

        return token_embeddings  # (sequence_length, hidden_size)

    def get_text_embedding_pooled(self, text, pooling="mean"):
        """
        Pool token embeddings into a single vector.
        """
        token_embeddings = self.get_text_embeddings(text)  # (seq_len, hidden_size)

        if pooling == "mean":
            return token_embeddings.mean(dim=0)  # (hidden_size,)
        elif pooling == "cls":
            return token_embeddings[0]  # CLS token embedding
        else:
            raise ValueError("Unsupported pooling method. Use 'mean' or 'cls'.")

if __name__ == "__main__":
    preprocessor = BERTTextPreprocessor(model_name="bert-base-uncased", device="cuda")

    text = "Gensim is great for word embeddings, but BERT is powerful for contextual embeddings."

    pooled_vector = preprocessor.get_text_embedding_pooled(text, pooling="mean")

    print(pooled_vector.shape)  # torch.Size([768]) for BERT-base
