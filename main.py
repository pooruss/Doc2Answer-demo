import argparse
from tqdm import tqdm
from model import Bert
from trainer import Trainer
from dataset import BaseDataset
from data.preprocess import InputFeatures
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from utils import boolean_string


torch.manual_seed(1)


parser = argparse.ArgumentParser()

parser.add_argument("--max_seq_len", default=512, type=int, required=True, help="sequence length.")
parser.add_argument("--epochs", default=5, type=int, required=True, help="epoch.")
parser.add_argument("--use_cuda", default=True, type=boolean_string, required=True, help="gpu or cpu.")
parser.add_argument("--batch_size", default=16, type=int, required=True, help="batch size.")
parser.add_argument("--learning_rate", default=1e-5, type=float, required=True, help="lr.")
parser.add_argument("--weight_decay", default=0.0001, type=float, required=True, help="lr.")
parser.add_argument("--save_interval", default=10, type=int, required=True, help="ckpt nums")
parser.add_argument("--print_interval", default=50, type=int, required=True, help="ckpt nums")
parser.add_argument("--save_path", default='/data/private/wanghuadong/liangshihao/QA/output/', type=str, required=True, help="save directory.")
parser.add_argument("--data_dir", default='/data/private/wanghuadong/liangshihao/QA/data/', type=str, required=True, help="save directory.")
parser.add_argument("--pretrained_path", default='/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base-multilingual-cased/', type=str, required=True, help="pretrained path.")
parser.add_argument("--gpu_ids", default='6', type=str, required=True, help="gpu ids.")

args = parser.parse_args()


# build dataloader
def build_dataloader(dataset):
    full_dataset = BaseDataset(data_dir=args.data_dir, dataset=dataset, do_train=True, do_eval=True, do_test=False, max_seq_len=args.max_seq_len, pretrained_path=args.pretrained_path)
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    return train_dataloader, test_dataloader

def validation(model, test_dataloader, device):
    model.eval()
    total_loss = 0.0
    total_span_loss, total_span_acc, total_zero_loss, total_zero_acc = 0.0, 0.0, 0.0, 0.0 
    for batch_data in tqdm(test_dataloader):
        with torch.no_grad():
            src_ids, label_ids, input_mask = batch_data
            src_ids, label_ids, input_mask = src_ids.to(device), label_ids.to(device), input_mask.to(device)
            span_loss, zero_loss, span_logits, zero_logits = model(src_ids, label_ids, input_mask)
            # print(logits)
            # print(label_ids[span_pos].view(-1))
            span_pos = (label_ids == 1)
            zero_pos = (label_ids == 0)
            span_acc = span_logits.view(-1, 2).max(dim=1)[1].eq(label_ids[span_pos].view(-1)).sum()
            span_acc = (span_acc * 100 / label_ids[span_pos].view(-1).size(0))
            zero_acc = zero_logits.view(-1, 2).max(dim=1)[1].eq(label_ids[zero_pos].view(-1)).sum()
            zero_acc = (zero_acc * 100 / label_ids[zero_pos].view(-1).size(0))
            total_loss += span_loss
            total_span_loss += span_loss
            total_span_acc += span_acc
            total_zero_acc += zero_acc
            total_zero_loss += zero_loss
    total_test_data = len(test_dataloader)
    span_loss = total_span_loss / total_test_data
    zero_loss = total_zero_loss / total_test_data
    span_acc = total_span_acc / total_test_data
    zero_acc = total_span_acc / total_test_data
    print("Validation. span_loss:{}, span_acc:{}, zero_loss:{}, zero_acc:{}".format
        (str(span_loss.cpu().detach().numpy()), str(span_acc.cpu().detach().numpy()), str(zero_loss.cpu().detach().numpy()), str(zero_acc.cpu().detach().numpy()))
    )
    model.train()
    return total_loss / total_test_data

def train(model, train_dataloader, test_dataloader, device):
    # build optim
    model.to(device)
    optimizer = optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    step_per_epoch = len(train_dataloader)
    # train progress
    total_train_step = step_per_epoch * args.epochs
    save_interval = total_train_step / args.save_interval
    print_interval = total_train_step / args.print_interval
    last_save_val_loss = 100.0
    for epoch in range(args.epochs):
        for iter, batch_data in enumerate(train_dataloader):
            src_ids, label_ids, input_mask = batch_data
            src_ids, label_ids, input_mask = src_ids.to(device), label_ids.to(device), input_mask.to(device)
            with torch.no_grad():
                span_pos = (label_ids == 1)
                zero_pos = (label_ids == 0)
            optimizer.zero_grad()
            span_loss, zero_loss, span_logits, zero_logits = model(src_ids, label_ids, input_mask)
            # print(logits)
            # print(label_ids[span_pos].view(-1))
            span_acc = span_logits.view(-1, 2).max(dim=1)[1].eq(label_ids[span_pos].view(-1)).sum()
            span_acc = (span_acc * 100 / label_ids[span_pos].view(-1).size(0))
            zero_acc = zero_logits.view(-1, 2).max(dim=1)[1].eq(label_ids[zero_pos].view(-1)).sum()
            zero_acc = (zero_acc * 100 / label_ids[zero_pos].view(-1).size(0))
            
            loss = zero_loss + span_loss
            loss.backward()
            optimizer.step()
            
            if iter % print_interval == 0:
                print("Step:{}/{}, span_loss:{}, span_acc:{}, zero_loss:{}, zero_acc:{}".format
                    (str(iter), str(total_train_step), str(loss.cpu().detach().numpy()), str(span_acc.cpu().detach().numpy()), str(zero_loss.cpu().detach().numpy()), str(zero_acc.cpu().detach().numpy()))
                )
            if iter % save_interval == 0:
                val_loss = validation(model, test_dataloader, device)
                if last_save_val_loss > val_loss:
                    last_save_val_loss = val_loss
                    torch.save(model.state_dict(), args.save_path + 'epoch_{}.pt'.format(str(epoch)))
        scheduler.step()


def main():
    # build dataloader
    train_dataloader, test_dataloader = build_dataloader("paq")
    # build model
    model_config = {"pretrained_path":args.pretrained_path}
    model = Bert(model_config=model_config)
    device_ids = list()
    for gpu_id in args.gpu_ids:
        device_ids.append(int(gpu_id))
    device = torch.device('cuda:{}'.format(device_ids[0]) if args.use_cuda else 'cpu')

    train(model, train_dataloader, test_dataloader, device)


if __name__ == '__main__':
    main()




