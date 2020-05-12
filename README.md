# Anchored_clinicalBERT_NER
An anchoring approach to Medical Information Extraction using Clinical BERT Embeddings
We use noisy-labeled (with UMLS terms) MIMIC III notes to "warm-start" clinicalBERT (Alsentzer et al., 2019) for NER on this noisy-labeled dataset, and then retrieve this warm-started BERT model and finetune it for NER on the 2010 i2b2 dataset.
We explore two different top layers to perform NER: Linear layer or Bi-LSTM + CRF layer.
We evaluate Micro Averaged F1-Score, Accuracy and per-tag F1-Score on a word-piece-token level (vs. a word-token level as in the competition) and compare to our own baseline using this evaluation methodology instead of the SOTA (Si et al., 2019).
