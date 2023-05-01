import torch
from torch.utils.data import Dataset

class EMBED_dataset(Dataset):
    def __init__(self, split='train', task="mortality"):
        super(EMBED_dataset, self).__init__()
        self.split = split

        embeds = torch.load('./embeds/{}_partial_{}.pth'.format(split, task))

        self.embeds, self.targets = embeds['embeds'], embeds['targets']
        self.label = self.targets
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        target = self.targets[index]
        item = self.embeds[index]
        ehr, cxr, notes = torch.split(item, len(item) // 3)
        return ehr, cxr, notes, target

if __name__ == '__main__':
    x = EMBED_dataset()
    x[0]
    pass