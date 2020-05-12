import torch
import os
import math
import pandas as pd
import numpy as np
import torch.nn.functional as F
from typing import Union
from seqeval.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoTokenizer, BertPreTrainedModel, BertModel, AdamW
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, random_split, RandomSampler, SequentialSampler, DataLoader
from tqdm import tqdm, trange
from bert_bilstm_crf import *
from bert_linear_classifier import *


class DataProcessorUMLS:
    def __init__(
            self,
            path_umls_mimic_data: str,
            path_clinical_bert_model: str,
            max_len_sent_without_special_tokens: int,
            batch_size: int,
    ):
        self.path_umls_mimic_data = path_umls_mimic_data
        self.path_clinical_bert_model = path_clinical_bert_model
        self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS = max_len_sent_without_special_tokens
        self.tokenizer_clinbert = AutoTokenizer.from_pretrained(path_clinical_bert_model)
        self.df_umls_mimic = pd.read_csv(path_umls_mimic_data)
        self.batch_size = batch_size
        self.labels_to_idx = {
            'B-problem': 0, 'B-test': 1, 'B-treatment': 2,
            'I-problem': 3, 'I-test': 4, 'I-treatment': 5,
            'O': 6, 'X': 7, '[CLS]': 8, '[SEP]': 9,
        }
        self.idx_to_labels = {v: k for k, v in self.labels_to_idx.items()}
        self.input_ids = None
        self.attention_masks = None
        self.labels_tokenized = None

    def extract_sentences_labels_from_df(self):
        sentences = []
        labels = []
        n_sentences_too_long = 0
        avg_sentence_length = []
        for sentence_id in self.df_umls_mimic.overall_sent.unique():
            df_temp = self.df_umls_mimic[self.df_umls_mimic.overall_sent == sentence_id]
            sentence = df_temp.WORD.tolist()
            sentence_clean = [x for x in sentence if x is not np.nan]
            avg_sentence_length.append(len(sentence_clean))
            if len(sentence_clean) > self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS:
                n_sentences_too_long += 1
            # sentence = " ".join(sentence)
            sentences.append(sentence_clean)
            label = df_temp.LABELS.tolist()
            label_clean = [label[i] for i in range(len(label)) if sentence[i] is not np.nan]
            labels.append(label_clean)
        avg_sentence_length = np.mean(avg_sentence_length)
        print(f"Number of sentences longer than 126: {n_sentences_too_long}" +
              f"out of {self.df_umls_mimic.overall_sent.nunique()}")
        print(f"Average sentence length: {round(avg_sentence_length, 2)}")
        print(self.df_umls_mimic.LABELS.value_counts() / len(self.df_umls_mimic) * 100)
        return sentences, labels

    def split_tokenized_sentences_too_long_properly(self, temp_sent, temp_label, sentences_tokenized, labels_tokenized):
        len_current = 0
        while len_current < len(temp_sent):
            try:
                current_sent = temp_sent[
                               len_current: len_current + self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS
                               ]
                current_lab = temp_label[
                              len_current: len_current + self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS
                              ]
                if (current_sent[0] != "[CLS]") and (current_lab[0] != "[CLS]"):
                    current_sent = ["[CLS]"] + current_sent
                    current_lab = ["[CLS]"] + current_lab

                if (current_sent[-1] != "[SEP]") and (current_lab[-1] != "[SEP]"):
                    current_sent = current_sent + ["[SEP]"]
                    current_lab = current_lab + ["[SEP]"]

                assert len(current_sent) == len(current_lab)
                assert len(current_sent) <= self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS + 2
                current_sent_tokenized = torch.tensor(
                    self.tokenizer_clinbert.convert_tokens_to_ids(current_sent)
                )
                current_lab_tokenized = torch.tensor(
                    [self.labels_to_idx[lab] for lab in current_lab]
                )
                sentences_tokenized.append(current_sent_tokenized)
                labels_tokenized.append(current_lab_tokenized)
                len_current += self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS
            except IndexError:
                eos = temp_sent[len_current:]
                eol = temp_label[len_current:]
                if (eos[0] != "[CLS]") and (eol[0] != "[CLS]"):
                    eos = ["[CLS]"] + eos
                    eol = ["[CLS]"] + eol

                if (eos[-1] != "[SEP]") and (eol[-1] != "[SEP]"):
                    eos = eos + ["[SEP]"]
                    eol = eol + ["[SEP]"]
                eos_tokenized = torch.tensor(
                    self.tokenizer_clinbert.convert_tokens_to_ids(eos)
                )
                eol_tokenized = torch.tensor(
                    [self.labels_to_idx[lab] for lab in eol]
                )
                sentences_tokenized.append(eos_tokenized)  # End of sentence
                labels_tokenized.append(eol_tokenized)  # End of label
        return sentences_tokenized, labels_tokenized

    def get_tokenized_sentences_labels_attention_masks(self):
        sentences, labels = self.extract_sentences_labels_from_df()
        sentences_tokenized = []
        labels_tokenized = []
        count = 0
        for sent, label in zip(sentences, labels):
            #try:
            assert len(sent) == len(label), "Sentence and Labels don't have same length"
            if len(sent) > self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS:
                count += 1
                continue
            temp_sent = []
            temp_label = []
            temp_sent.append("[CLS]")
            temp_label.append("[CLS]")
            for word, lab_word in zip(sent, label):
                token_list = self.tokenizer_clinbert.tokenize(word)
                for m, token in enumerate(token_list):
                    temp_sent.append(token)
                    if m == 0:  # Means that the word wasn't split by the tokenizer
                        temp_label.append(lab_word)
                    else:
                        temp_label.append("X")

            temp_sent.append("[SEP]")
            temp_label.append("[SEP]")
            if len(temp_sent) > self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS + 2:
                sentences_tokenized, labels_tokenized = self.split_tokenized_sentences_too_long_properly(
                    temp_sent=temp_sent,
                    temp_label=temp_label,
                    sentences_tokenized=sentences_tokenized,
                    labels_tokenized=labels_tokenized,
                )
            else:
                temp_sent_ids = self.tokenizer_clinbert.convert_tokens_to_ids(temp_sent)
                temp_label_tok = [self.labels_to_idx[lab] for lab in temp_label]
                sentences_tokenized.append(torch.tensor(temp_sent_ids))
                labels_tokenized.append(torch.tensor(temp_label_tok))
            #except IndexError:
            #    count += 1
            #    #print(sent, "\n", label)

        sentences_tokenized = torch.from_numpy(pad_sequences(
            sentences_tokenized,
            maxlen=self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS + 2,
            dtype="long", truncating="post", padding="post"
        ))
        # sentences_tokenized = torch.stack([torch.tensor(sent) for sent in sentences_tokenized])
        labels_tokenized = torch.from_numpy(pad_sequences(
            labels_tokenized,
            maxlen=self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS + 2,
            value=self.labels_to_idx["O"], padding="post",
            dtype="long", truncating="post"
        ))
        # labels_tokenized = torch.stack([torch.tensor(lab) for lab in labels_tokenized])
        attention_masks = torch.stack([
            torch.tensor([1 if x > 0 else 0 for x in sent_tokens])  # since the padding index is 0
            for sent_tokens in sentences_tokenized
        ])
        print(f"Error with {count} sentences because of nan, ignored those sentences")
        input_ids = torch.squeeze(sentences_tokenized, 0)
        attention_masks = torch.squeeze(attention_masks, 0)
        labels_tokenized = torch.squeeze(labels_tokenized, 0)
        print(f"We now have {len(input_ids)} sentences tokenized with their labels and attention masks")
        return input_ids, attention_masks, labels_tokenized

    def save_tokenized_sentences_labels_attention_masks(
            self,
            input_ids: torch.Tensor,
            attention_masks: torch.Tensor,
            labels_tokenized: torch.Tensor,
            path_to_save_tokenized_data_umls: str,
    ):
        assert path_to_save_tokenized_data_umls[-1] == "/", "Need path to finish with '/'"
        if not os.path.exists(path_to_save_tokenized_data_umls):
            os.makedirs(path_to_save_tokenized_data_umls)

        max_len = self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS + 2
        torch.save(input_ids, path_to_save_tokenized_data_umls + f"input_ids_umls_max_len_{max_len}.pt")
        torch.save(attention_masks, path_to_save_tokenized_data_umls + f"attention_masks_umls_max_len_{max_len}.pt")
        torch.save(labels_tokenized, path_to_save_tokenized_data_umls + f"labels_tokenized_umls_max_len_{max_len}.pt")
        print(f"Saved tokenized data for UMLS/MIMIC III in folder: {path_to_save_tokenized_data_umls}")
    
    def load_tokenized_sentences_labels_attention_masks(
            self,
            path_to_save_tokenized_data_umls: str,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        assert path_to_save_tokenized_data_umls[-1] == "/", "Need path to finish with '/'"
        max_len = self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS + 2
        input_ids = torch.load(path_to_save_tokenized_data_umls + f"input_ids_umls_max_len_{max_len}.pt")
        attention_masks = torch.load(path_to_save_tokenized_data_umls + f"attention_masks_umls_max_len_{max_len}.pt")
        labels_tokenized = torch.load(path_to_save_tokenized_data_umls + f"labels_tokenized_umls_max_len_{max_len}.pt")
        print(f"Loaded tokenized data for UMLS/MIMIC III from folder: {path_to_save_tokenized_data_umls}")
        return input_ids, attention_masks, labels_tokenized

    def get_train_and_valid_dataloader_from_loaded_tokenized_data(
            self, train_size: float, input_ids: torch.Tensor,
            attention_masks: torch.Tensor, labels_tokenized: torch.Tensor
    ) -> (DataLoader, DataLoader):
        assert train_size < 1, "train_size needs to be a fraction between 0 and 1 excluded, default set to 0.9"
        dataset = TensorDataset(input_ids, attention_masks, labels_tokenized)
        n_train = int(train_size * len(dataset))
        n_valid = len(dataset) - n_train
        train_dataset, valid_dataset = random_split(dataset, [n_train, n_valid])
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=self.batch_size, drop_last=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, sampler=valid_sampler,
            batch_size=self.batch_size
        )
        return train_dataloader, valid_dataloader

    def get_train_and_valid_dataloader(self, train_size: float) -> (DataLoader, DataLoader):
        assert train_size < 1, "train_size needs to be a fraction between 0 and 1 excluded, default value should be 0.9"
        input_ids, attention_masks, labels_tokenized = self.get_tokenized_sentences_labels_attention_masks()
        dataset = TensorDataset(input_ids, attention_masks, labels_tokenized)
        n_train = int(train_size * len(dataset))
        n_valid = len(dataset) - n_train
        train_dataset, valid_dataset = random_split(dataset, [n_train, n_valid])
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=self.batch_size, drop_last=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, sampler=valid_sampler,
            batch_size=self.batch_size
        )
        return train_dataloader, valid_dataloader


class DataProcessorI2B2:
    def __init__(
            self,
            path_i2b2_folder: str,
            path_clinical_bert_model: str,
            max_len_sent_without_special_tokens: int,
            batch_size: int,
    ):
        self.path_i2b2_folder = path_i2b2_folder
        self.path_clinical_bert_model = path_clinical_bert_model
        self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS = max_len_sent_without_special_tokens
        self.tokenizer_clinbert = AutoTokenizer.from_pretrained(path_clinical_bert_model)
        self.df_i2b2_train = pd.read_csv(path_i2b2_folder + "train.csv")
        self.df_i2b2_dev = pd.read_csv(path_i2b2_folder + "dev.csv")
        self.df_i2b2_test = pd.read_csv(path_i2b2_folder + "test.csv")
        self.batch_size = batch_size
        self.labels_to_idx = {
            'B-problem': 0, 'B-test': 1, 'B-treatment': 2,
            'I-problem': 3, 'I-test': 4, 'I-treatment': 5,
            'O': 6, 'X': 7, '[CLS]': 8, '[SEP]': 9,
        }
        self.idx_to_labels = {v: k for k, v in self.labels_to_idx.items()}
        self.input_ids_train = None
        self.input_ids_dev = None
        self.input_ids_test = None
        self.attention_masks_train = None
        self.attention_masks_dev = None
        self.attention_masks_test = None
        self.labels_tokenized_train = None
        self.labels_tokenized_dev = None
        self.labels_tokenized_test = None

    def extract_sentences_labels_from_df(self, df_i2b2: pd.DataFrame):
        sentences = []
        labels = []
        n_sentences_too_long = 0
        avg_sentence_length = []
        for sentence_id in df_i2b2.UNIQUE_ID.unique():
            df_temp = df_i2b2[df_i2b2.UNIQUE_ID == sentence_id]
            sentence = df_temp.SENTENCE.tolist()
            sentence_clean = [x for x in sentence if x is not np.nan]
            avg_sentence_length.append(len(sentence_clean))
            if len(sentence_clean) > self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS:
                n_sentences_too_long += 1
            # sentence = " ".join(sentence)
            sentences.append(sentence_clean)
            label = df_temp.LABELS.tolist()
            label_clean = [label[i] for i in range(len(label)) if sentence[i] is not np.nan]
            labels.append(label_clean)
        avg_sentence_length = np.mean(avg_sentence_length)
        print(f"Number of sentences longer than 126: {n_sentences_too_long}" +
              f"out of {df_i2b2.UNIQUE_ID.nunique()}")
        print(f"Average sentence length: {round(avg_sentence_length, 2)}")
        print(df_i2b2.LABELS.value_counts() / len(df_i2b2) * 100)
        return sentences, labels

    def split_tokenized_sentences_too_long_properly(self, temp_sent, temp_label, sentences_tokenized, labels_tokenized):
        len_current = 0
        while len_current < len(temp_sent):
            try:
                current_sent = temp_sent[
                               len_current: len_current + self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS
                               ]
                current_lab = temp_label[
                              len_current: len_current + self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS
                              ]
                if (current_sent[0] != "[CLS]") and (current_lab[0] != "[CLS]"):
                    current_sent = ["[CLS]"] + current_sent
                    current_lab = ["[CLS]"] + current_lab

                if (current_sent[-1] != "[SEP]") and (current_lab[-1] != "[SEP]"):
                    current_sent = current_sent + ["[SEP]"]
                    current_lab = current_lab + ["[SEP]"]

                assert len(current_sent) == len(current_lab)
                assert len(current_sent) <= self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS + 2
                current_sent_tokenized = torch.tensor(
                    self.tokenizer_clinbert.convert_tokens_to_ids(current_sent)
                )
                current_lab_tokenized = torch.tensor(
                    [self.labels_to_idx[lab] for lab in current_lab]
                )
                assert current_lab_tokenized[0] == torch.tensor(8)  # Check that they all start with CLS
                sentences_tokenized.append(current_sent_tokenized)
                labels_tokenized.append(current_lab_tokenized)
                len_current += self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS
            except IndexError:
                eos = temp_sent[len_current:]
                eol = temp_label[len_current:]
                if (eos[0] != "[CLS]") and (eol[0] != "[CLS]"):
                    eos = ["[CLS]"] + eos
                    eol = ["[CLS]"] + eol

                if (eos[-1] != "[SEP]") and (eol[-1] != "[SEP]"):
                    eos = eos + ["[SEP]"]
                    eol = eol + ["[SEP]"]
                eos_tokenized = torch.tensor(
                    self.tokenizer_clinbert.convert_tokens_to_ids(eos)
                )
                eol_tokenized = torch.tensor(
                    [self.labels_to_idx[lab] for lab in eol]
                )
                assert eol_tokenized[0] == torch.tensor(8)  # Check that they all start with CLS
                sentences_tokenized.append(eos_tokenized)  # End of sentence
                labels_tokenized.append(eol_tokenized)  # End of label
        return sentences_tokenized, labels_tokenized

    def get_tokenized_sentences_labels_attention_masks(self, data_part: str):
        if data_part == "train":
            sentences, labels = self.extract_sentences_labels_from_df(self.df_i2b2_train)
        elif data_part == "dev":
            sentences, labels = self.extract_sentences_labels_from_df(self.df_i2b2_dev)
        elif data_part == "test":
            sentences, labels = self.extract_sentences_labels_from_df(self.df_i2b2_test)
        else:
            raise ValueError(f"Expected value in ['train', 'dev', 'test'], got {data_part}")

        sentences_tokenized = []
        labels_tokenized = []
        attention_masks = []
        count = 0
        for sent, label in zip(sentences, labels):
            try:
                assert len(sent) == len(label), "Sentence and Labels don't have same length"
                if len(sent) > self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS:
                    count += 1
                    continue
                temp_sent = []
                temp_label = []
                temp_sent.append("[CLS]")
                temp_label.append("[CLS]")
                for word, lab_word in zip(sent, label):
                    token_list = self.tokenizer_clinbert.tokenize(word)
                    for m, token in enumerate(token_list):
                        temp_sent.append(token)
                        if m == 0:  # Means that the word wasn't split by the tokenizer
                            temp_label.append(lab_word)
                        else:
                            temp_label.append("X")

                temp_sent.append("[SEP]")
                temp_label.append("[SEP]")
                if len(temp_sent) > self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS + 2:
                    sentences_tokenized, labels_tokenized = self.split_tokenized_sentences_too_long_properly(
                        temp_sent=temp_sent,
                        temp_label=temp_label,
                        sentences_tokenized=sentences_tokenized,
                        labels_tokenized=labels_tokenized,
                    )
                else:
                    temp_sent_ids = self.tokenizer_clinbert.convert_tokens_to_ids(temp_sent)
                    temp_label_tok = [self.labels_to_idx[lab] for lab in temp_label]
                    sentences_tokenized.append(torch.tensor(temp_sent_ids))
                    labels_tokenized.append(torch.tensor(temp_label_tok))
            except:
                count += 1
                print(sent, "\n", label)

        sentences_tokenized = torch.from_numpy(pad_sequences(
            sentences_tokenized,
            maxlen=self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS + 2,
            dtype="long", truncating="post", padding="post"
        ))
        # sentences_tokenized = torch.stack([torch.tensor(sent) for sent in sentences_tokenized])
        labels_tokenized = torch.from_numpy(pad_sequences(
            labels_tokenized,
            maxlen=self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS + 2,
            value=self.labels_to_idx["O"], padding="post",
            dtype="long", truncating="post"
        ))
        # labels_tokenized = torch.stack([torch.tensor(lab) for lab in labels_tokenized])
        attention_masks = torch.stack([
            torch.tensor([1 if x > 0 else 0 for x in sent_tokens])  # since the padding index is 0
            for sent_tokens in sentences_tokenized
        ])
        print(f"Error with {count} sentences because of nan, ignored those sentences")
        input_ids = torch.squeeze(sentences_tokenized, 0)
        attention_masks = torch.squeeze(attention_masks, 0)
        labels_tokenized = torch.squeeze(labels_tokenized, 0)
        print(f"We now have {len(input_ids)} sentences tokenized with their labels and attention masks")
        return input_ids, attention_masks, labels_tokenized

    def save_tokenized_sentences_labels_attention_masks(
            self,
            input_ids: torch.Tensor,
            attention_masks: torch.Tensor,
            labels_tokenized: torch.Tensor,
            path_to_save_tokenized_data_i2b2: str,
            data_part: str,
    ):
        assert data_part in ["train", "dev", "test"],\
            f"Expected value in ['train', 'dev', 'test'], got {data_part}"
        assert path_to_save_tokenized_data_i2b2[-1] == "/", "Need path to finish with '/'"
        if not os.path.exists(path_to_save_tokenized_data_i2b2):
            os.makedirs(path_to_save_tokenized_data_i2b2)

        max_len = self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS + 2
        torch.save(
            input_ids, path_to_save_tokenized_data_i2b2 +
                       f"input_ids_i2b2_{data_part}_max_len_{max_len}.pt"
        )
        torch.save(
            attention_masks, path_to_save_tokenized_data_i2b2 +
                             f"attention_masks_i2b2_{data_part}_max_len_{max_len}.pt"
        )
        torch.save(
            labels_tokenized, path_to_save_tokenized_data_i2b2 +
                              f"labels_tokenized_i2b2_{data_part}_max_len_{max_len}.pt"
        )
        print(f"Saved {data_part} tokenized data for i2b2 in folder: {path_to_save_tokenized_data_i2b2}")

    def load_tokenized_sentences_labels_attention_masks(
            self,
            path_to_save_tokenized_data_i2b2: str,
            data_part: str,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        assert data_part in ["train", "dev", "test"], \
            f"Expected value in ['train', 'dev', 'test'], got {data_part}"
        assert path_to_save_tokenized_data_i2b2[-1] == "/", "Need path to finish with '/'"
        max_len = self.MAX_LEN_SENT_WITHOUT_SPECIAL_TOKENS + 2
        input_ids = torch.load(path_to_save_tokenized_data_i2b2 + f"input_ids_i2b2_{data_part}_max_len_{max_len}.pt")
        attention_masks = torch.load(path_to_save_tokenized_data_i2b2 + f"attention_masks_i2b2_{data_part}_max_len_{max_len}.pt")
        labels_tokenized = torch.load(path_to_save_tokenized_data_i2b2 + f"labels_tokenized_i2b2_{data_part}_max_len_{max_len}.pt")
        print(f"Loaded {data_part} tokenized data for i2b2 from folder: {path_to_save_tokenized_data_i2b2}")
        return input_ids, attention_masks, labels_tokenized
    
    def create_dataset_and_dataloader_i2b2(self, input_ids, attention_masks, labels_tokenized):
        dataset = TensorDataset(input_ids, attention_masks, labels_tokenized)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(
            dataset, sampler=sampler, batch_size=self.batch_size, drop_last=True
        )
        return dataloader

    def get_train_dev_and_test_dataloader_i2b2_from_loaded_tokenized_data(
            self, input_ids: torch.Tensor,
            attention_masks: torch.Tensor,
            labels_tokenized: torch.Tensor,
            data_part: str,
    ) -> DataLoader:
        assert data_part in ["train", "dev", "test"], f"Expected value in ['train', 'dev', 'test'], got {data_part}"
        dataloader_i2b2_data_part = self.create_dataset_and_dataloader_i2b2(
            input_ids, attention_masks, labels_tokenized
        )
        return dataloader_i2b2_data_part

    def get_train_dev_and_test_dataloader_i2b2(self) -> (DataLoader, DataLoader, DataLoader):
        input_ids_train, attention_masks_train, labels_tokenized_train = self.get_tokenized_sentences_labels_attention_masks(
            data_part="train"
        )
        input_ids_dev, attention_masks_dev, labels_tokenized_dev = self.get_tokenized_sentences_labels_attention_masks(
            data_part="dev"
        )
        input_ids_test, attention_masks_test, labels_tokenized_test = self.get_tokenized_sentences_labels_attention_masks(
            data_part="test"
        )
        # Saving objects as attributes of the class
        self.input_ids_train = input_ids_train
        self.input_ids_dev = input_ids_dev
        self.input_ids_test = input_ids_test
        self.attention_masks_train = attention_masks_train
        self.attention_masks_dev = attention_masks_dev
        self.attention_masks_test = attention_masks_test
        self.labels_tokenized_train = labels_tokenized_train
        self.labels_tokenized_dev = labels_tokenized_dev
        self.labels_tokenized_test = labels_tokenized_test
        train_dataloader_i2b2 = self.create_dataset_and_dataloader_i2b2(
            input_ids_train, attention_masks_train, labels_tokenized_train
        )
        dev_dataloader_i2b2 = self.create_dataset_and_dataloader_i2b2(
            input_ids_dev, attention_masks_dev, labels_tokenized_dev
        )
        test_dataloader_i2b2 = self.create_dataset_and_dataloader_i2b2(
            input_ids_test, attention_masks_test, labels_tokenized_test
        )
        return train_dataloader_i2b2, dev_dataloader_i2b2, test_dataloader_i2b2


class ModelingUMLS:
    def __init__(
            self,
            train_dataloader: DataLoader,
            valid_dataloader: DataLoader,
            batch_size: int,
            max_len_sent_without_special_tokens: int,
            learning_rate: float,
            epochs: int,
            max_grad_norm: float,
            use_grad_clipping: bool,
            full_finetuning: bool,
            use_bi_lstm_crf: bool,
            hidden_size_lstm: int,
            path_clinical_bert_model: str,
            path_save_clinical_bert_umls: str,
    ):
        self.labels_to_idx = {
            'B-problem': 0, 'B-test': 1, 'B-treatment': 2,
            'I-problem': 3, 'I-test': 4, 'I-treatment': 5,
            'O': 6, 'X': 7, '[CLS]': 8, '[SEP]': 9,
        }
        self.idx_to_labels = {v: k for k, v in self.labels_to_idx.items()}
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.batch_size = batch_size
        self.max_len_sent_without_special_tokens = max_len_sent_without_special_tokens
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.use_grad_clipping = use_grad_clipping
        self.full_finetuning = full_finetuning
        self.use_bi_lstm_crf = use_bi_lstm_crf
        self.hidden_size_lstm = hidden_size_lstm
        self.path_clinical_bert_model = path_clinical_bert_model
        self.path_save_clinical_bert_umls = path_save_clinical_bert_umls


    @property
    def device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            n_gpu = torch.cuda.device_count()
        else:
            device = torch.device("cpu")

        return device

    @property
    def n_gpu(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            n_gpu = torch.cuda.device_count()
        # If there's no GPU available...
        else:
            device = torch.device("cpu")
            n_gpu = 0

        return n_gpu

    def get_optimizer(self, model_ner_clinbert: BertForTokenClassification):
        if self.full_finetuning:
            # Fine tune model all layer parameters
            param_optimizer = list(model_ner_clinbert.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            # Only fine tune classifier parameters
            param_optimizer = list(model_ner_clinbert.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        return optimizer

    def train_model_umls(self):
        f1_score_valid = 0
        num_train_optimization_steps = int(
            math.ceil(len(self.train_dataloader.dataset) / self.batch_size)
        ) * self.epochs
        if self.use_bi_lstm_crf:
            model_ner_clinbert = BertBiLSTMCRF.from_pretrained(
                self.path_clinical_bert_model,
                num_labels=len(self.labels_to_idx),
                hidden_size_lstm=self.hidden_size_lstm,
            )
        else:
            model_ner_clinbert = BertForTokenClassification.from_pretrained(
                self.path_clinical_bert_model,
                num_labels=len(self.labels_to_idx)
            )
        model_ner_clinbert.to(self.device)
        #if self.n_gpu > 1:
        #    model_ner_clinbert = torch.nn.DataParallel(model_ner_clinbert)

        optimizer = self.get_optimizer(model_ner_clinbert)
        print("\n***** Running Training on UMLS *****")
        print(f"Num examples ={len(self.train_dataloader.dataset)}")
        print(
            f"Batch size = {self.batch_size}, Learning Rate={self.learning_rate}, " +
            f"Using Gradient Clipping={self.use_grad_clipping}, Epochs={self.epochs}, " +
            f"Max Sentence Length={self.max_len_sent_without_special_tokens + 2}"
        )
        print(f"Num steps = {num_train_optimization_steps}")
        for epoch_ in trange(self.epochs, desc="Epoch"):
            model_ner_clinbert.train()  # Inside the loop because we'll evaluate on validation at each epoch now
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(self.train_dataloader):
                # add batch to gpu
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                # forward pass
                outputs = model_ner_clinbert(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )
                if self.use_bi_lstm_crf:
                    loss = outputs
                else:
                    loss, scores = outputs[:2]
                #if self.n_gpu > 1:
                    # When multi gpu, average it
                #    loss = loss.mean()

                # backward pass
                loss.backward()

                # track train loss
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

                # gradient clipping
                if self.use_grad_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=model_ner_clinbert.parameters(),
                        max_norm=self.max_grad_norm
                    )

                # update parameters
                optimizer.step()
                optimizer.zero_grad()

            # print train loss per epoch
            print("Train loss: {}".format(tr_loss / nb_tr_steps))
            if epoch_ + 1 != self.epochs:
                f1_score_valid = self.validate_model_umls(model_ner_clinbert, epoch_, print_report=False)
            else:
                f1_score_valid = self.validate_model_umls(model_ner_clinbert, epoch_, print_report=True)

        print(f"Finished training & evaluating ClinicalBERT on UMLS data")
        return model_ner_clinbert, f1_score_valid

    def save_model_ner_clinbert_umls_warm_start(
            self,
            model_ner_clinbert_umls_warm_start: BertForTokenClassification,
            tokenizer_clinbert_umls_warm_start,
    ):
        if not os.path.exists(self.path_save_clinical_bert_umls):
            os.makedirs(self.path_save_clinical_bert_umls)

        model_to_save_umls = (
            model_ner_clinbert_umls_warm_start.module
            if hasattr(model_ner_clinbert_umls_warm_start, "module")
            else model_ner_clinbert_umls_warm_start
        )
        output_model_file_umls = os.path.join(self.path_save_clinical_bert_umls, "pytorch_model.bin")
        output_config_file_umls = os.path.join(self.path_save_clinical_bert_umls, "config.json")
        torch.save(model_to_save_umls.state_dict(), output_model_file_umls)
        model_to_save_umls.config.to_json_file(output_config_file_umls)
        tokenizer_clinbert_umls_warm_start.save_vocabulary(self.path_save_clinical_bert_umls)
        print(f"Saved ClinicalBert with warm start UMLS NER in folder: {self.path_save_clinical_bert_umls}")

    def validate_model_umls(
            self,
            model_ner_clinbert: BertForTokenClassification,
            epoch: int,
            print_report: bool = False,
    ):
        model_ner_clinbert.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        print("***** Running Evaluation (Validation set) *****")
        print(f"Num examples ={len(self.valid_dataloader.dataset)}")
        print(f"Batch size = {self.batch_size}")
        for step, batch in enumerate(self.valid_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, label_ids = batch
            with torch.no_grad():
                outputs = model_ner_clinbert(
                    input_ids, token_type_ids=None, attention_mask=input_mask,
                )

            # Get NER predict result
            if self.use_bi_lstm_crf:
                logits = outputs
            else:
                logits = outputs[0]  # For eval mode, the first result of outputs is logits
                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                logits = logits.detach().cpu().numpy()

            # Get NER true result
            label_ids = label_ids.to('cpu').numpy()
            # Only predict the real word, mark=0, will not calculate
            input_mask = input_mask.to('cpu').numpy()
            # Compare the valuable predict result
            for i, mask in enumerate(input_mask):
                # Real one
                temp_1 = []
                # Predict one
                temp_2 = []

                for j, m in enumerate(mask):
                    # Mark=0, meaning its a pad word, dont compare
                    if m:
                        if (
                                (self.idx_to_labels[label_ids[i][j]] != "[CLS]")
                                and (self.idx_to_labels[label_ids[i][j]] != "[SEP]")
                                and (self.idx_to_labels[label_ids[i][j]] != "X")
                        ):  # Exclude the X, CLS and SEP labels
                            temp_1.append(self.idx_to_labels[label_ids[i][j]])
                            temp_2.append(self.idx_to_labels[logits[i][j]])
                    else:
                        break

                y_true.append(temp_1)
                y_pred.append(temp_2)

        f1_score_valid = round(f1_score(y_true, y_pred), 4)
        accuracy_valid = round(accuracy_score(y_true, y_pred), 4)
        print(f"Validation F1 Score epoch {epoch+1}/{self.epochs}: {f1_score_valid}")
        print(f"Validation Accuracy Score epoch {epoch+1}/{self.epochs}: {accuracy_valid}")
        if print_report:
            report = classification_report(y_true, y_pred, digits=4)
            print(report)
        print("--------------------------------------------------------------")
        return round(accuracy_score(y_true, y_pred), 4)


class ModelingI2B2:
    def __init__(
            self,
            train_dataloader: DataLoader,
            dev_dataloader: Union[None, DataLoader],
            test_dataloader: DataLoader,
            batch_size: int,
            max_len_sent_without_special_tokens: int,
            learning_rate: float,
            epochs: int,
            max_grad_norm: float,
            use_grad_clipping: bool,
            full_finetuning: bool,
            use_bi_lstm_crf: bool,
            hidden_size_lstm: int,
            path_clinical_bert_umls_warm_start_model: str,
            path_save_clinical_bert_i2b2: str,
    ):
        self.labels_to_idx = {
            'B-problem': 0, 'B-test': 1, 'B-treatment': 2,
            'I-problem': 3, 'I-test': 4, 'I-treatment': 5,
            'O': 6, 'X': 7, '[CLS]': 8, '[SEP]': 9,
        }
        self.idx_to_labels = {v: k for k, v in self.labels_to_idx.items()}
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader
        self.batch_size = batch_size
        self.max_len_sent_without_special_tokens = max_len_sent_without_special_tokens
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.use_grad_clipping = use_grad_clipping
        self.full_finetuning = full_finetuning
        self.use_bi_lstm_crf = use_bi_lstm_crf
        self.hidden_size_lstm = hidden_size_lstm
        self.path_clinical_bert_umls_warm_start_model = path_clinical_bert_umls_warm_start_model
        self.path_save_clinical_bert_i2b2 = path_save_clinical_bert_i2b2

    @property
    def device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        return device

    @property
    def n_gpu(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            n_gpu = torch.cuda.device_count()
        # If there's no GPU available...
        else:
            device = torch.device("cpu")
            n_gpu = 0

        return n_gpu

    def get_optimizer(self, model_ner_clinbert: BertForTokenClassification):
        if self.full_finetuning:
            # Fine tune model all layer parameters
            param_optimizer = list(model_ner_clinbert.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            # Only fine tune classifier parameters
            param_optimizer = list(model_ner_clinbert.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        return optimizer

    def train_model_i2b2_full_data(self):
        f1_score_data_test = 0
        num_train_optimization_steps = int(
            math.ceil(len(self.train_dataloader.dataset) / self.batch_size)
        ) * self.epochs
        if self.use_bi_lstm_crf:
            model_ner_clinbert_i2b2 = BertBiLSTMCRF.from_pretrained(
                self.path_clinical_bert_umls_warm_start_model,
                num_labels=len(self.labels_to_idx),
                hidden_size_lstm=self.hidden_size_lstm,
            )
        else:
            model_ner_clinbert_i2b2 = BertForTokenClassification.from_pretrained(
                self.path_clinical_bert_umls_warm_start_model,
                num_labels=len(self.labels_to_idx)
            )
        model_ner_clinbert_i2b2.to(self.device)
        # if self.n_gpu > 1:
        #    model_ner_clinbert_i2b2 = torch.nn.DataParallel(model_ner_clinbert_i2b2)

        optimizer = self.get_optimizer(model_ner_clinbert_i2b2)
        print("\n***** Running Training on i2b2 *****")
        print(f"Num examples = {len(self.train_dataloader.dataset)}")
        print(
            f"Batch size = {self.batch_size}, Learning Rate = {self.learning_rate}," +
            f"Using Gradient Clipping = {self.use_grad_clipping}, Epochs = {self.epochs}" +
            f"Max Sentence Length = {self.max_len_sent_without_special_tokens + 2}" +
            f"Use Bi_LSTM_CRF = {self.use_bi_lstm_crf}, LSTM Hidden Size = {self.hidden_size_lstm}"
        )
        print(f"Num steps = {num_train_optimization_steps}")
        for epoch_ in trange(self.epochs, desc="Epoch"):
            model_ner_clinbert_i2b2.train()  # Inside the loop because we'll evaluate on validation at each epoch now
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(self.train_dataloader):
                # add batch to gpu
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                # forward pass
                outputs = model_ner_clinbert_i2b2(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )
                if self.use_bi_lstm_crf:
                    loss = outputs
                else:
                    loss, scores = outputs[:2]
                # if self.n_gpu > 1:
                #    # When multi gpu, average it
                #    loss = loss.mean()
                # backward pass
                loss.backward()
                # track train loss
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

                # gradient clipping
                if self.use_grad_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=model_ner_clinbert_i2b2.parameters(),
                        max_norm=self.max_grad_norm
                    )

                # update parameters
                optimizer.step()
                optimizer.zero_grad()

            # print train loss per epoch
            print("Train loss: {}".format(tr_loss / nb_tr_steps))
            if epoch_ + 1 == self.epochs:
                f1_score_data_test = self.validate_model_i2b2(
                    model_ner_clinbert_i2b2, epoch_,
                    dataloader=self.test_dataloader,
                    print_report=True, data_part="test"
                )
            else:
                continue

        print(f"Finished training & evaluating warm-started ClinicalBERT on full i2b2 data")
        return model_ner_clinbert_i2b2, f1_score_data_test

    def train_model_i2b2(self, train_or_not: bool):
        f1_score_data_test = 0
        num_train_optimization_steps = int(
            math.ceil(len(self.train_dataloader.dataset) / self.batch_size)
        ) * self.epochs
        if self.use_bi_lstm_crf:
            model_ner_clinbert_i2b2 = BertBiLSTMCRF.from_pretrained(
                self.path_clinical_bert_umls_warm_start_model,
                num_labels=len(self.labels_to_idx),
                hidden_size_lstm=self.hidden_size_lstm,
            )
        else:
            model_ner_clinbert_i2b2 = BertForTokenClassification.from_pretrained(
                self.path_clinical_bert_umls_warm_start_model,
                num_labels=len(self.labels_to_idx)
            )
        model_ner_clinbert_i2b2.to(self.device)
        #if self.n_gpu > 1:
        #    model_ner_clinbert_i2b2 = torch.nn.DataParallel(model_ner_clinbert_i2b2)

        optimizer = self.get_optimizer(model_ner_clinbert_i2b2)
        print("\n***** Running Training on i2b2 *****")
        print(f"Num examples = {len(self.train_dataloader.dataset)}")
        print(
            f"Batch size = {self.batch_size}, Learning Rate = {self.learning_rate}," +
            f"Using Gradient Clipping = {self.use_grad_clipping}, Epochs = {self.epochs}" +
            f"Max Sentence Length = {self.max_len_sent_without_special_tokens + 2}" +
            f"Use Bi_LSTM_CRF = {self.use_bi_lstm_crf}, LSTM Hidden Size = {self.hidden_size_lstm}"
        )
        print(f"Num steps = {num_train_optimization_steps}")
        if train_or_not:
            for epoch_ in trange(self.epochs, desc="Epoch"):
                model_ner_clinbert_i2b2.train()  # Inside the loop because we'll evaluate on validation at each epoch now
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                for step, batch in enumerate(self.train_dataloader):
                    # add batch to gpu
                    batch = tuple(t.to(self.device) for t in batch)
                    b_input_ids, b_input_mask, b_labels = batch

                    # forward pass
                    outputs = model_ner_clinbert_i2b2(
                        b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels
                    )
                    if self.use_bi_lstm_crf:
                        loss = outputs
                    else:
                        loss, scores = outputs[:2]
                    #if self.n_gpu > 1:
                    #    # When multi gpu, average it
                    #    loss = loss.mean()
                    # backward pass
                    loss.backward()
                    # track train loss
                    tr_loss += loss.item()
                    nb_tr_examples += b_input_ids.size(0)
                    nb_tr_steps += 1

                    # gradient clipping
                    if self.use_grad_clipping:
                        torch.nn.utils.clip_grad_norm_(
                            parameters=model_ner_clinbert_i2b2.parameters(),
                            max_norm=self.max_grad_norm
                        )

                    # update parameters
                    optimizer.step()
                    optimizer.zero_grad()

                # print train loss per epoch
                print("Train loss: {}".format(tr_loss / nb_tr_steps))
                if epoch_ + 1 != self.epochs:
                    f1_score_data_dev = self.validate_model_i2b2(
                        model_ner_clinbert_i2b2, epoch_,
                        dataloader=self.dev_dataloader,
                        print_report=False, data_part="dev"
                    )
                else:
                    f1_score_data_dev = self.validate_model_i2b2(
                        model_ner_clinbert_i2b2, epoch_,
                        dataloader=self.dev_dataloader,
                        print_report=True, data_part="dev"
                    )
                    print("#################### EVALUATING ON TEST DATA ####################")
                    f1_score_data_test = self.validate_model_i2b2(
                        model_ner_clinbert_i2b2, epoch_,
                        dataloader=self.test_dataloader,
                        print_report=True, data_part="test"
                    )
        else:
            print("#################### EVALUATING ON TEST DATA ####################")
            f1_score_data_test = self.validate_model_i2b2(
                model_ner_clinbert_i2b2, 0,
                dataloader=self.test_dataloader,
                print_report=True, data_part="test"
            )

        print(f"Finished training & evaluating warm-started ClinicalBERT on i2b2 data")
        return model_ner_clinbert_i2b2, f1_score_data_test

    def validate_model_i2b2(
            self,
            model_ner_clinbert_i2b2: BertForTokenClassification,
            epoch: int,
            dataloader: DataLoader,
            data_part: str,
            print_report: bool = False,
    ):
        assert data_part in ["dev", "test"], f"Expected value in ['dev', 'test'], fot {data_part}"
        DATA_PART = data_part[0].upper() + data_part[1:]
        model_ner_clinbert_i2b2.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        print(f"***** Running Evaluation on {data_part} set *****")
        print(f"Num examples = {len(dataloader.dataset)}")
        print(f"Batch size = {self.batch_size}")
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, label_ids = batch
            with torch.no_grad():
                outputs = model_ner_clinbert_i2b2(
                    input_ids, token_type_ids=None, attention_mask=input_mask,
                )

            # Get NER predict result
            if self.use_bi_lstm_crf:
                logits = outputs
            else:
                logits = outputs[0]  # For eval mode, the first result of outputs is logits
                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                logits = logits.detach().cpu().numpy()

            # Get NER true result
            label_ids = label_ids.to('cpu').numpy()
            # Only predict the real word, mark=0, will not calculate
            input_mask = input_mask.to('cpu').numpy()
            # Compare the valuable predict result
            for i, mask in enumerate(input_mask):
                # Real one
                temp_1 = []
                # Predict one
                temp_2 = []

                for j, m in enumerate(mask):
                    # Mark=0, meaning its a pad word, dont compare
                    if m:
                        if (
                                (self.idx_to_labels[label_ids[i][j]] != "[CLS]")
                                and (self.idx_to_labels[label_ids[i][j]] != "[SEP]")
                                and (self.idx_to_labels[label_ids[i][j]] != "X")
                        ):  # Exclude the X, CLS and SEP labels
                            temp_1.append(self.idx_to_labels[label_ids[i][j]])
                            temp_2.append(self.idx_to_labels[logits[i][j]])
                    else:
                        break

                y_true.append(temp_1)
                y_pred.append(temp_2)

        f1_score_data_part = round(f1_score(y_true, y_pred), 4)
        accuracy_valid_data_part = round(accuracy_score(y_true, y_pred), 4)
        print(f"Evaluation on {DATA_PART} Data")
        print(f"{DATA_PART} F1 Score epoch {epoch + 1}/{self.epochs}: {f1_score_data_part}")
        print(f"{DATA_PART} Accuracy Score epoch {epoch + 1}/{self.epochs}: {accuracy_valid_data_part}")
        if print_report:
            report = classification_report(y_true, y_pred, digits=4)
            print(report)
        print("--------------------------------------------------------------")
        return f1_score_data_part
    
    def save_model_ner_clinbert_i2b2_with_warm_start(
            self,
            model_ner_clinbert_i2b2_with_warm_start: BertForTokenClassification,
            tokenizer_clinbert_i2b2_with_warm_start,
    ):
        if not os.path.exists(self.path_save_clinical_bert_i2b2):
            os.makedirs(self.path_save_clinical_bert_i2b2)

        model_to_save_i2b2 = (
            model_ner_clinbert_i2b2_with_warm_start.module
            if hasattr(model_ner_clinbert_i2b2_with_warm_start, "module")
            else model_ner_clinbert_i2b2_with_warm_start
        )
        output_model_file_i2b2 = os.path.join(self.path_save_clinical_bert_i2b2, "pytorch_model.bin")
        output_config_file_i2b2 = os.path.join(self.path_save_clinical_bert_i2b2, "config.json")
        torch.save(model_to_save_i2b2.state_dict(), output_model_file_i2b2)
        model_to_save_i2b2.config.to_json_file(output_config_file_i2b2)
        tokenizer_clinbert_i2b2_with_warm_start.save_vocabulary(self.path_save_clinical_bert_i2b2)
        print(f"Saved ClinicalBert with warm start i2b2 NER in folder: {self.path_save_clinical_bert_i2b2}")
