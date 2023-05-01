import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset.mul_dataset import MUL_dataset
from models.mul_module import MUL_module
from torch.cuda.amp import autocast


device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_amp = True
task = 'longstay'
longstay_mintime = 3*24
print(f'on the {device} device')
print('run embedding generation')
print('task: {}'.format(task))

img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(norm_mean, norm_std),
])

train_dataset = MUL_dataset(split='train',img_transform=img_transform, task=task, longstay_mintime=longstay_mintime)
test_dataset = MUL_dataset(split='test', img_transform=img_transform, task=task, longstay_mintime=longstay_mintime)
val_dataset = MUL_dataset(split='val', img_transform=img_transform, task=task, longstay_mintime=longstay_mintime)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.get_collate())
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=test_dataset.get_collate())
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=val_dataset.get_collate())

model = MUL_module(device=device)
model = model.to(device)

@torch.no_grad()
def embed_generation(model, train_loader, test_loader, val_loader):
    model.ehr.load_state_dict(torch.load('./saved_model/best_ehr_partial_model_{}.pth'.format(task)))
    model.cxr.load_state_dict(torch.load('./saved_model/best_cxr_partial_model_{}.pth'.format(task)))
    model.note.load_state_dict(torch.load('./saved_model/best_note_partial_model_{}.pth'.format(task)))

    print('on train set')
    embeds, targets = [], []
    for batch_idx, ((demo, ce_ts, le_ts, pe_ts, timestamps), (img_list, img_positions, img_times), notes, target) in enumerate(train_loader):
        torch.cuda.empty_cache()
        with autocast(enabled=use_amp):
            img_list = [[y.to(device) for y in x] for x in img_list]
            demo = demo.to(device)
            embed = model.embedding(demo, ce_ts, le_ts, pe_ts, timestamps, img_list, img_positions, img_times, notes)
            
            embeds.append(embed.to('cpu'))
            targets.append(target.to('cpu'))
        
        if batch_idx % 50 == 0:
            print('batch [{}/{}]'.format(batch_idx + 1, len(train_loader)))
    
    embeds = torch.cat(embeds)
    targets = torch.cat(targets)

    torch.save({'embeds': embeds, 'targets': targets}, './embeds/train_partial_{}.pth'.format(task))

    print('on test set')
    embeds, targets = [], []
    for batch_idx, ((demo, ce_ts, le_ts, pe_ts, timestamps), (img_list, img_positions, img_times), notes, target) in enumerate(test_loader):
        torch.cuda.empty_cache()
        with autocast(enabled=use_amp):
            img_list = [[y.to(device) for y in x] for x in img_list]
            demo = demo.to(device)
            embed = model.embedding(demo, ce_ts, le_ts, pe_ts, timestamps, img_list, img_positions, img_times, notes)
            
            embeds.append(embed.to('cpu'))
            targets.append(target.to('cpu'))
        
        if batch_idx % 10 == 0:
            print('batch [{}/{}]'.format(batch_idx + 1, len(test_loader)))
    
    embeds = torch.cat(embeds)
    targets = torch.cat(targets)

    torch.save({'embeds': embeds, 'targets': targets}, './embeds/test_partial_{}.pth'.format(task))

    print('on val set')
    embeds, targets = [], []
    for batch_idx, ((demo, ce_ts, le_ts, pe_ts, timestamps), (img_list, img_positions, img_times), notes, target) in enumerate(val_loader):
        torch.cuda.empty_cache()
        with autocast(enabled=use_amp):
            img_list = [[y.to(device) for y in x] for x in img_list]
            demo = demo.to(device)
            embed = model.embedding(demo, ce_ts, le_ts, pe_ts, timestamps, img_list, img_positions, img_times, notes)
            
            embeds.append(embed.to('cpu'))
            targets.append(target.to('cpu'))
        
        if batch_idx % 10 == 0:
            print('batch [{}/{}]'.format(batch_idx + 1, len(val_loader)))
    
    embeds = torch.cat(embeds)
    targets = torch.cat(targets)

    torch.save({'embeds': embeds, 'targets': targets}, './embeds/val_partial_{}.pth'.format(task))

    return

if __name__ == '__main__':
    embed_generation(model, train_loader, test_loader, val_loader)