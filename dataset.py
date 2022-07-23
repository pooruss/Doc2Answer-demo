""" data reader for CoKE
"""
import os
import csv
import sys
import tqdm
import torch
import logging
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, RobertaTokenizer, BertTokenizer
# logging.basicConfig(
#     format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#     datefmt='%m/%d/%Y %H:%M:%S')
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.info(logger.getEffectiveLevel())

torch.manual_seed(1)

class BaseDataset(Dataset):
    """ DataReader
    """

    def __init__(self, data_dir, dataset, do_train, do_eval, do_test, max_seq_len, pretrained_path):
        # D:/Study/BMKG/git_clone/BMKG/bmkg/base_model/
        if "roberta" in pretrained_path:
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_path)
            self.cls_id = self.tokenizer.convert_tokens_to_ids("<s>")
            self.sep_id = self.tokenizer.convert_tokens_to_ids("</s>")
            self.pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
            self.mask_id = self.tokenizer.convert_tokens_to_ids("<mask>")
            self.unk_id = self.tokenizer.convert_tokens_to_ids("<unk>")
        else:
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)
            self.cls_id = self.tokenizer.convert_tokens_to_ids("[CLS]")
            self.sep_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
            self.pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
            self.mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        self.max_seq_len = max_seq_len
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_test = do_test
        self.features = self.read_features(data_dir, dataset)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        return torch.tensor(feature.src_ids, dtype=torch.long), \
               torch.tensor(feature.label_ids, dtype=torch.long), \
               torch.tensor(feature.mask_ids, dtype=torch.long)

    def get_npy(self, data_root, dataset_name):
        if self.do_train:
            features_file = os.path.join(data_root, dataset_name, "train_features.npy")
            features = np.load(features_file, allow_pickle=True)
        elif self.do_eval:
            features_file = os.path.join(data_root, dataset_name, "dev_features.npy")
            features = np.load(features_file, allow_pickle=True)
        else:
            features_file = os.path.join(data_root, dataset_name, "test_features.npy")
            features = np.load(features_file, allow_pickle=True)
        return features

    def read_features(self, data_root, dataset):
        features = np.array([])
        if isinstance(dataset, list):
            for dataset_name in dataset:
                feature = self.get_npy(os.path.join(data_root, dataset_name))
                features = np.concatate(features, feature, dim=0)
        else:
            features = self.get_npy(os.path.join(data_root, dataset_name))
        return features





