from TextPreprocessor import TextPreprocessor
from gensim.models import Word2Vec

# 1. Load/prepare corpus
corpus = [
    "how to train a neural network",
    "best laptop for machine learning",
    "python code for sentiment analysis",
    "difference between ai and ml",
    "introduction to deep learning",
]

preprocessor = TextPreprocessor()

# 2. Preprocess corpus (tokenize and clean)
tokenized_corpus = preprocessor.preprocess_corpus(corpus)
print("Tokenized Corpus:", tokenized_corpus)

# 3. Train Word2Vec
model = Word2Vec(sentences=tokenized_corpus, 
                 vector_size=100, 
                 window=5, 
                 min_count=1, 
                 workers=4, 
                 sg=1) # Skip-gram model (sg=0 for CBOW)

# 4. Save only the word vectors
model.wv.save("msmarco_wordvectors.kv")
