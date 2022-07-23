import os
import sys
import json
import tqdm
import numpy as np
from transformers import AutoTokenizer, RobertaTokenizer, BertTokenizer


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, src_ids, mask_ids, label_ids):
        self.src_ids = src_ids
        self.mask_ids = mask_ids
        self.label_ids = label_ids

class PAProcessor(object):
    """Processor for Doc Answer data set."""
    def __init__(self, pretrained_path, max_seq_len, do_train=False, do_dev=False, do_test=False):
        self.labels = set()
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
        self.max_len = max_seq_len       
        self.id2passage = dict()
        if do_train:
            self.task = "train"
        elif do_dev:
            self.task = "dev"
        else:
            self.task = "test"

    def get_passage_answer(self, data_root, json_file, task):
        self.task = task
        def convert_string_to_ids(string):
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(string))
            
        def get_answer_position(src, tgt):
            tgt_pos = 0
            for pos, token in enumerate(src):
                if tgt_pos >= len(tgt):
                    return pos - len(tgt) + 1
                if token == tgt[tgt_pos]:
                    if tgt_pos == len(tgt) - 1:
                        return pos - len(tgt) + 1
                    else:
                        tgt_pos += 1
                elif token == tgt[0]:
                    tgt_pos = 1
                else:
                    tgt_pos = 0
                    continue
            return -1
        
        features = []
        json_file = open(data_root + json_file, "r", encoding='utf-8')
        data_dict = json.load(json_file)
        data_dict = data_dict["data"]
        for data in tqdm.tqdm(data_dict):
            paragraphs = data["paragraphs"]
            for para in paragraphs:
                passage = para["context"]
                qass = para["qas"]
                answer_list = []
                for qas in qass:
                    answers = qas["answers"]
                    for answer in answers:
                        answer = answer["text"]
                        answer_list.append(answer)

                pas_ids = self.tokenizer(text=passage, max_length=self.max_len, padding='max_length', truncation='longest_first')
                input_mask = pas_ids["attention_mask"]
                pas_ids = pas_ids["input_ids"]

                label_ids = [0] * self.max_len
                for idx, answer in enumerate(answer_list):
                    if idx > 2:
                        break
                    ans_ids = convert_string_to_ids(answer)
                    
                    ans_pos = get_answer_position(pas_ids, ans_ids)
                    if ans_pos == -1 or ans_pos + len(ans_ids) > self.max_len:
                        # print(passageid)
                        # print(passage)
                        # print(pas_tokens)
                        # print(ans_tokens)
                        continue
                    # print(pas_ids[ans_pos : ans_pos + len(ans_ids)])
                    # print(ans_ids)
                    for i in range(len(ans_ids)):
                        label_ids[ans_pos + i] = 1
                        assert pas_ids[ans_pos + i] == ans_ids[i]

                assert len(label_ids) == self.max_len
                assert len(input_mask) == self.max_len
                assert len(pas_ids) == self.max_len
                pas_ids = np.array(pas_ids)
                ans_ids = np.array(ans_ids)
                features.append(InputFeatures(src_ids=pas_ids, mask_ids=input_mask, label_ids=label_ids))
                # print(passage)
                # print(answer)
                # print(ans_ids)
                # print(label_ids)
                # if len(features) > 500:
                #     break
        print(len(features))
        np.save((data_root + self.task + "_features.npy"), features)

if __name__ == '__main__':

    pretrained_path = sys.argv[1]
    data_root = sys.argv[2]  # data_root
    max_seq_len = int(sys.argv[3]) 
    
    processor = PAProcessor(pretrained_path, max_seq_len)

    processor.get_passage_answer(data_root, 'train.json', 'train')
    processor.get_passage_answer(data_root, 'dev.json', 'dev')




