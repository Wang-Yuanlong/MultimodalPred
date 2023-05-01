import torch
import gensim
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


class Note_module(nn.Module):
    def __init__(self, embed_size=512, hidden_size=512, device='cpu', pretrained_path='saved_model/doc2vec_model'):
        super(Note_module, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.device = device
        self.pretrained_path = pretrained_path
        
        self.doc2vec = gensim.models.Doc2Vec.load(pretrained_path)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True)
        self.pooling = nn.AdaptiveMaxPool1d(1)

        self.cls_head = nn.Linear(hidden_size, 2)

    def embedding(self, notes):
        vecs = [[self.doc2vec.infer_vector(y) for y in x] for x in notes]
        vecs = [np.array(vec) for vec in vecs]
        embeds = [torch.tensor(vec, requires_grad=False).to(self.device) for vec in vecs]

        embeds = pack_sequence(embeds, enforce_sorted=False)
        embeds, (_, _) = self.lstm(embeds)
        embeds, lengths = pad_packed_sequence(embeds, batch_first=True)
        embeds = list(map(lambda x: x[0][:x[1],:], zip(embeds, lengths)))

        embeds = list(map(lambda x: self.pooling(x.T.unsqueeze(0)).reshape(-1), embeds))
        embeds = torch.stack(embeds)
        return embeds
    
    def forward(self, notes):
        embeds = self.embedding(notes)
        preds = self.cls_head(embeds.float())
        return preds

if __name__ == "__main__":
    x = [[["the", "woman", "chest"]]]
    o = Note_module(pretrained_path="saved_model/doc2vec_model")
    o(x)

