import json
import time
import torch
import torch.nn as nn
import torch.utils.data


def load_from_json(fin):
    datas = []
    for line in fin:
        data = json.loads(line)
        datas.append(data)
    return datas


def dump_to_json(datas, fout):
    for data in datas:
        fout.write(json.dumps(data, sort_keys=True, 
            separators=(',', ': '), ensure_ascii=False))
        fout.write('\n')
    fout.close()

class DataSet(torch.utils.data.Dataset):

    def __init__(self, data_path, vocabs, config, is_train=True):
        print("starting load...")
        self.opt = config.args
        self.w2id = config.w2i_vocabs
        self.id2w = config.i2v_vocabs
        start_time = time.time()
        self.datas = load_from_json(open(data_path, 'r', encoding='utf8'))
        print("loading time:", time.time() - start_time)

        self.vocabs = vocabs
        self.vocab_size = len(self.vocabs)
        self.is_train = is_train

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        T = self.load_comments(data['context'])

        if not self.is_train:
            comment = data['comment'][0]
        else:
            comment = data['comment']
        Y = DataSet.padding(comment, self.opt.max_len)

        return Y, T

    def get_candidate(self, index):
        data = self.datas[index]
        T = self.load_comments(data['context'])

        Y = [DataSet.padding(c, self.opt.max_len) for c in data['candidate']]
        return torch.stack(Y), T, data


    def load_comments(self, context):
        if opt.n_com == 0:
            return torch.LongTensor([1]+[0]*self.opt.max_len*5+[2])
        return DataSet.padding(context, self.opt.max_len*self.opt.n_com)

    @staticmethod
    def padding(data, max_len):
        data = data.split()
        if len(data) > max_len-2:
            data = data[:max_len-2]
        Y = list(map(lambda t: self.w2id.get(t, 3), data))
        Y = [1] + Y + [2]
        length = len(Y)
        Y = torch.cat([torch.LongTensor(Y), torch.zeros(max_len - length).long()])
        return Y

    @staticmethod
    def transform_to_words(ids):
        words = []
        for id in ids:
            if id == 2:
                break
            words.append(self.id2w[str(id.item())])
        return "".join(words)


def get_dataset(data_path, vocabs, config, is_train=True,):
    return DataSet(data_path, vocabs, config, is_train=is_train)

def get_dataloader(dataset, batch_size, is_train=True):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)
