Text Preprocessing Tools
- TextPreprocessor.py: 
  - Filter the corpus to contain the most frequently occuring words 
        (the smallest set of unique words) whose cumulative frequency >= p% (e.g. 95%) of total words. 
  - Create Word2Vec embeddings of the filtered corpus (firstly, run 0_train_word2vec.py, then TextPreprocessor.py is called to create the embeddings of the filtered corpus (see the example in TextPreprocessor.py).
- TransformerTextPreprocessor.py: 
  - Filter the corpus to contain the most frequently occuring words (the smallest set of unique words) whose cumulative frequency >= p% (e.g. 95%) of total words. 
  - Create text embeddings of the filtered corpus using the BERT transformer (see the example in TransformerTextPreprocessor.py).
