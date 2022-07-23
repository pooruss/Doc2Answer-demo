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

    def get_pid_answer_offset(self, data_root, jsonl_name, output_name):
        # get passageid, answer, offset ------> file1
        exapmle_dict = {}
        filter_set = set()
        output_file = open(os.path.join(data_root, output_name), "w", encoding='utf-8')
        # PAQ.metadata.jsonl 
        input_file = open(os.path.join(data_root, jsonl_name), 'r', encoding='utf-8')
        for line in input_file:
            # print(line.strip())
            exapmle_dict = json.loads(line.strip())
            exapmle_dict = exapmle_dict["answers"]
            for answer in exapmle_dict:
                passage_id = answer["passage_id"]
                offset = answer["offset"]
                ans = answer["text"]
                if passage_id + '\t' + ans + '\t' + str(offset) in filter_set:
                    continue
                # print(passage_id + '\t' + ans + '\t' + str(offset))
                output_file.write(passage_id + '\t' + ans + '\t' + str(offset) + '\n')
                filter_set.add(passage_id + '\t' + ans + '\t' + str(offset))
        exapmle_dict = None
        filter_set = None

    def get_id2passage(self, data_root):
        id2passage = {}
        max_text_len = 0
        # psgs_w100.tsv
        for line in tqdm.tqdm(open(data_root, 'r', encoding='utf-8')):
            u = line.strip()
            id, passage, title = u.split('\t')
            id2passage[id] = passage[1:-1]
            # pas_ids = self.tokenizer(text=passage, max_length=self.max_len, padding='max_length', truncation='longest_first')
            # attn_mask = pas_ids["attention_mask"]
            # print(attn_mask)
            # pas_ids = pas_ids["input_ids"]
            # assert len(pas_ids) == self.max_len
            # text_len = len(pas_ids)
            # if text_len > max_text_len:
            #     max_text_len = text_len
        # print(max_text_len)
        self.id2passage = id2passage
        id2passage = None

    def get_passage_answer(self, data_root, pad_file_name):
        def convert_string_to_ids(string):
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(string))
        def get_answer_position(src, tgt):
            tgt_pos = 0
            for pos, token in enumerate(src):
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
        
        pao_file = open(data_root + pad_file_name, "r", encoding='utf-8')
        features = []
        for line in tqdm.tqdm(pao_file):
            passageid, answer, offset = line.strip().split("\t")
            passage = self.id2passage[passageid].replace('\"\"', '')
            answer = answer.replace('\"', '')
            pas_ids = self.tokenizer(text=passage, max_length=self.max_len, padding='max_length', truncation='longest_first')
            input_mask = pas_ids["attention_mask"]
            pas_ids = pas_ids["input_ids"]
            ans_ids = convert_string_to_ids(answer)
            # assert ans_ids in pas_ids
            pas_ids = np.array(pas_ids)
            ans_ids = np.array(ans_ids)
            ans_pos = get_answer_position(pas_ids, ans_ids)
            if ans_pos == -1:
                # print(passageid)
                # print(passage)
                # print(pas_tokens)
                # print(ans_tokens)
                continue
            label_ids = [0] * self.max_len
            for i in range(len(ans_ids)):
                label_ids[ans_pos + i] = 1
                assert pas_ids[ans_pos + i] == ans_ids[i]
            assert len(label_ids) == self.max_len
            assert len(input_mask) == self.max_len
            assert len(pas_ids) == self.max_len
            features.append(InputFeatures(src_ids=pas_ids, mask_ids=input_mask, label_ids=label_ids))
            if len(features) > 200000:
                break
        np.save((data_root + self.task + "_features.npy"), features)

if __name__ == '__main__':

    pretrained_path = sys.argv[1]  # FB15K237 18 ; WN18RR 17
    data_root = sys.argv[2]  # data_root
    jsonl_file = sys.argv[3]  # PAQ.metadata.jsonl 
    pad_file_name = sys.argv[4] # PAQ.pid.ans.off
    id2passage_file = sys.argv[5]  # psgs_w100.tsv
    max_seq_len = int(sys.argv[6]) 

    jsonl_file = data_root + jsonl_file
    id2passage_file = data_root + id2passage_file
    
    processor = PAProcessor(pretrained_path, max_seq_len, do_train=True)
    # processor.get_pid_answer_offset(data_root, jsonl_file, pad_file_name)

    processor.get_id2passage(id2passage_file)
    processor.get_passage_answer(data_root, pad_file_name)



