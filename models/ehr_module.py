import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import numpy as np


def value_embedding_data(d = 512, split = 200):
    vec = np.array([np.arange(split) * i for i in range(d//2)], dtype=np.float32).transpose()
    vec = vec / vec.max() 
    embedding = np.concatenate((np.sin(vec), np.cos(vec)), 1)
    embedding[0, :d] = 0
    embedding = torch.from_numpy(embedding)
    return embedding

class EHR_Embedding(nn.Module):
    def __init__(self, embed_type, varible_num=1, split_num=200, embed_size=256, device='cpu'):
        super(EHR_Embedding, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.embed_type = embed_type
        if embed_type == 'demo':
            self.anchor_age = nn.Embedding(9, embed_size)
            self.insurance = nn.Embedding(3, embed_size)
            self.language = nn.Embedding(2, embed_size)
            self.marital_status = nn.Embedding(5, embed_size)
            self.ethnicity = nn.Embedding(8, embed_size)
        elif embed_type == 'chart':
            self.var = nn.Embedding(varible_num, embed_size)
            self.value = nn.Embedding.from_pretrained(value_embedding_data(embed_size, split_num))
            self.eye_opening = nn.Embedding(4, embed_size)
            self.motor_response = nn.Embedding(6, embed_size)
            self.verbal_response = nn.Embedding(6, embed_size)
            self.map = nn.Linear(2 * embed_size, embed_size)
        elif embed_type == 'lab':
            self.var = nn.Embedding(varible_num, embed_size)
            self.value = nn.Embedding.from_pretrained(value_embedding_data(embed_size, split_num))
            self.map = nn.Linear(2 * embed_size, embed_size)
        elif embed_type == 'procedure':
            self.operation = nn.Embedding(varible_num, embed_size)
        elif embed_type == 'time':
            self.value = nn.Embedding.from_pretrained(value_embedding_data(embed_size, split_num))
    
    def forward(self, x):
        if x == []:
            return torch.tensor([]).to(self.device)
        if self.embed_type == 'demo':
            '''
            batch solution
            '''
            demo = x.T
            anchor_age, insurance, language, marital_status, ethnicity = [x for x in demo]
            anchor_age -= 1
            marital_status += 1
            anchor_age = self.anchor_age(anchor_age)
            insurance = self.insurance(insurance)
            language = self.language(language)
            marital_status = self.marital_status(marital_status)
            ethnicity = self.ethnicity(ethnicity)
            embed = torch.stack([anchor_age, insurance, language, marital_status, ethnicity])
            embed = embed.transpose(0, 1)
            # embed = [x for x in embed]
        elif self.embed_type == 'chart':
            '''
            single patient, single time
            '''
            var, value = list(zip(*x))
            var_embed = self.var(torch.tensor(var).to(self.device))
            value_embed = []
            for idx, value_ in enumerate(value):
                value_ = torch.tensor(value_).to(self.device)
                if var[idx] == 0:
                    value_embed_ = self.eye_opening(value_)
                elif var[idx] == 1:
                    value_embed_ = self.motor_response(value_)
                elif var[idx] == 2:
                    value_embed_ = self.verbal_response(value_)
                else:
                    value_embed_ = self.value(value_)
                value_embed.append(value_embed_)
            value_embed = torch.stack(value_embed)
            embed = torch.cat([var_embed, value_embed], dim=1)
            embed = self.map(embed)
        elif self.embed_type == 'lab':
            '''
            single patient, single time
            '''
            embed = []
            var, value = list(zip(*x))
            var, value = torch.tensor(var).to(self.device), torch.tensor(value).to(self.device)
            var_embed = self.var(var)
            value_embed = self.value(value)
            embed = torch.cat([var_embed, value_embed], dim=1)
            embed = self.map(embed)
        elif self.embed_type == 'procedure':
            '''
            single patient, single time
            '''
            embed = torch.tensor(x).to(self.device)
            embed = self.operation(embed)
        elif self.embed_type == 'time':
            if (len(x) == 0) and (x == np.array([])).all():
                value = torch.LongTensor([0]).to(self.device)
            else:
                value = torch.tensor(x).to(self.device)
            value = value / 48
            value = torch.div(value, (1/200), rounding_mode='trunc').long()
            value[value >= 200] = 199
            embed = self.value(value)
        return embed

class EHR_module(nn.Module):
    def __init__(self, embed_size=256, hidden_size=256, device='cpu'):
        super(EHR_module, self).__init__()
        self.device = device
        self.demo_embed = EHR_Embedding(embed_type='demo', device=device)
        self.chart_embed = EHR_Embedding(embed_type='chart', varible_num=9, embed_size=embed_size, device=device)
        self.lab_embed = EHR_Embedding(embed_type='lab', varible_num=22, embed_size=embed_size, device=device)
        self.procedure_embed = EHR_Embedding(embed_type='procedure', varible_num=10, embed_size=embed_size, device=device)
        self.time_embed = EHR_Embedding(embed_type='time', embed_size=embed_size, device=device)

        self.lstm = nn.LSTM(input_size=2 * embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.pooling = nn.AdaptiveMaxPool1d(1)

        self.cls_head = nn.Linear(2 * hidden_size, 2)
    
    def embedding(self, demo, chart, lab, procedure, time, pooling=False):
        demo_embeds = self.demo_embed(demo)
        embeds = []
        for idx, (chart_ts, lab_ts, procedure_ts) in enumerate(zip(chart, lab, procedure)):
            demo_embed = demo_embeds[idx]
            # embed_ts = list(map(lambda e: 
            #                            self.pooling(torch.cat([demo_embed,
            #                                                    self.chart_embed(e[0]),
            #                                                    self.lab_embed(e[1]), 
            #                                                    self.procedure_embed(e[2])]).T.unsqueeze(0)).reshape(-1),
            #                 zip(chart_ts, lab_ts, procedure_ts)))
            embed_ts = []
            for ce, le, pe in zip(chart_ts, lab_ts, procedure_ts):
                ce_embed = self.chart_embed(ce)
                le_embed = self.lab_embed(le)
                pe_embed = self.procedure_embed(pe)
                embed_thistime = torch.cat([demo_embed, ce_embed, le_embed, pe_embed])
                embed_thistime = self.pooling(embed_thistime.T.unsqueeze(0)).reshape(-1)
                embed_ts.append(embed_thistime)
            if embed_ts == []:
                # with open('suspicious_sample.txt', 'a') as f:
                #     f.write("{}\n".format(demo[idx]))
                embed_ts = self.pooling(demo_embed.T.unsqueeze(0)).reshape(1, -1)
            else:
                embed_ts = torch.stack(embed_ts)

            time_embed = self.time_embed(time[idx])
            embed_ts = torch.cat([embed_ts, time_embed], dim=1)

            # time_weight = torch.tensor(time[idx]).to(self.device)
            # time_weight = torch.softmax(time_weight, dim=0)

            # embed_ts = torch.sum(embed_ts * time_weight.reshape(-1,1), dim=0)
            embeds.append(embed_ts)
        # embeds = torch.stack(embeds)
        embeds = pack_sequence(embeds, enforce_sorted=False)
        embeds, (_, _) = self.lstm(embeds)
        embeds, lengths = pad_packed_sequence(embeds, batch_first=True)
        # batch_embeds = []
        # for idx, length in enumerate(lengths):
        #     single_embeds = embeds[idx][:length,:]
        #     batch_embeds.append(single_embeds)
        embeds = list(map(lambda x: x[0][:x[1],:], zip(embeds, lengths)))
        if pooling:
            embeds = list(map(lambda x: self.pooling(x.T.unsqueeze(0)).reshape(-1), embeds))
            embeds = torch.stack(embeds)
        return embeds

    def forward(self, demo, chart, lab, procedure, time):
        embed = self.embedding(demo, chart, lab, procedure, time, pooling=True)
        pred = self.cls_head(embed.float())
        return pred

    