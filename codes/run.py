import os
import time
import json
import argparse
import torch
import torch.optim as optim

from models import Model
from models import save_model
from dataset import get_dataset, get_dataloader
from metrics import recall,mean_rank, mean_reciprocal_rank

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "gpu")

def set_parser():
    parser = argparse.ArgumentParser(description='train.py')

    parser.add_argument('--n_emb', type=int, default=512, help="Embedding size")
    parser.add_argument('--n_hidden', type=int, default=512, help="Hidden size")
    parser.add_argument('--d_ff', type=int, default=2048, help="Hidden size of Feedforward")
    parser.add_argument('--n_head', type=int, default=8, help="Number of head")
    parser.add_argument('--n_block', type=int, default=6, help="Number of block")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--vocab_size', type=int, default=30000, help="Vocabulary size")
    parser.add_argument('--epoch', type=int, default=50, help="Number of epoch")
    parser.add_argument('--report', type=int, default=500, help="Number of report interval")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate")
    parser.add_argument('--restore', type=str, default='', help="Restoring model path")
    parser.add_argument('--mode', type=str, default='train', help="Train or test")
    parser.add_argument('--dir', type=str, default='ckpt', help="Checkpoint directory")
    parser.add_argument('--max_len', type=int, default=20, help="Limited length for text")
    parser.add_argument('--n_com', type=int, default=5, help="Number of input comments")
    return parser.parse_args()


class Config(object):

    def __init__(self, args, data_path="data"):
        self.args = args
        self.data_path = data_path
        self.train_path = os.path.join(data_path, "train-context.json")
        self.dev_path = os.path.join(data_path, "dev-candidate.json")
        self.test_path = os.path.join(data_path, "test-candidate.json")
        self.vocab_path = os.path.join(data_path, "dicts-30000.json")
        self.w2i_vocabs = json.load(open(self.vocab_path, 
                                        'r', encoding='utf8'))['word2id']
        self.i2v_vocabs = json.load(open(self.vocab_path, 
                                        'r', encoding='utf8'))['id2word']
        self.args.vocab_size = len(self.w2i_vocabs)
        if not os.path.exists(self.args.dir):
            os.mkdir(self.args.dir)

def train(config):
    # train_path:train-context.json
    args = config.args
    train_set = get_dataset(config.train_path, config.w2i_vocabs, config, is_train=True)
    dev_set = get_dataset(config.dev_path, config.w2i_vocabs, config, is_train=False)
    # X:img,torch.stack;
    train_batch = get_dataloader(train_set, args.batch_size, is_train=True)
    model = Model(n_emb=args.n_emb, n_hidden=args.n_hidden, vocab_size=args.vocab_size,
                  dropout=args.dropout, d_ff=args.d_ff, n_head=args.n_head, n_block=args.n_block)
    if args.restore != '':
        model_dict = torch.load(args.restore)
        model.load_state_dict(model_dict)
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
    best_score = -1000000

    for i in range(args.epoch):
        model.train()
        report_loss, start_time, n_samples = 0, time.time(), 0
        count, total = 0, len(train_set) // args.batch_size + 1
        for batch in train_batch:
            Y, T = batch
            Y = Y.to(device)
            T = T.to(device)
            optimizer.zero_grad()
            loss = model(Y, T)
            loss.backward()
            optimizer.step()
            report_loss += loss.item()
            #break
            n_samples += len(Y.data)
            count += 1
            if count % args.report == 0 or count == total:
                print('%d/%d, epoch: %d, report_loss: %.3f, time: %.2f'
                      % (count, total, i+1, report_loss / n_samples, time.time() - start_time))
                score = eval(model, dev_set, args.batch_size)
                model.train()
                if score > best_score:
                    best_score = score
                    save_model(os.path.join(args.dir, 'best_checkpoint.pt'), model)
                else:
                    save_model(os.path.join(args.dir, 'checkpoint.pt'), model)
                report_loss, start_time, n_samples = 0, time.time(), 0

    return model


def eval(model, dev_set, batch_size):
    print("starting evaluating...")
    start_time = time.time()
    model.eval()
    # predictions, references = [], []
    dev_batch = get_dataloader(dev_set, batch_size, is_train=False)

    loss = 0
    with torch.no_grad():
        for batch in dev_batch:
            Y, T = batch
            Y = Y.to(device)
            T = T.to(device)
            loss += model(Y, T).item()
    print(loss)
    print("evaluting time:", time.time() - start_time)

    return -loss


def test(test_set, model):
    print("starting testing...")
    start_time = time.time()
    model.eval()
    predictions, references = [], []
    with torch.no_grad():
        for i in range(len(test_set)):
            Y, T, data = test_set.get_and_candidate(i)
            Y = Y.to(device)
            T = T.to(device)
            ids = model.ranking(Y, T).data

            candidate = []
            comments = list(data['candidate'].keys())
            for id in ids:
                candidate.append(comments[id])
            predictions.append(candidate)
            references.append(data['candidate'])
            if i % 100 == 0:
                print(i)

    recall_1 = recall(predictions, references, 1)
    recall_5 = recall(predictions, references, 5)
    recall_10 = recall(predictions, references, 10)
    mr = mean_rank(predictions, references)
    mrr = mean_reciprocal_rank(predictions, references)
    s = "r1={}, r5={}, r10={}, mr={}, mrr={}"
    print(s.format(recall_1, recall_5, recall_10, mr, mrr))

    print("testing time:", time.time() - start_time)
    # for ref, pre in zip(references, predictions):
    #     print(ref)
    #     print("-"*20)  
    #     print(pre)
    #     print("*"*100)
        

if __name__ == '__main__':
    args = set_parser()
    config = Config(args)
    if args.mode == 'train':
        train(config)
    else:
        test_set = get_dataset(config.test_path, config.w2i_vocabs, config, is_train=False)
        model = Model(n_emb=args.n_emb, n_hidden=args.n_hidden, vocab_size=args.vocab_size,
                  dropout=args.dropout, d_ff=args.d_ff, n_head=args.n_head, n_block=args.n_block)
        model_dict = torch.load(args.restore)
        model.load_state_dict(model_dict)
        model.to(device)
        test(test_set, model)
