import torch
import argparse
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset.mul_dataset import MUL_dataset
from models.mul_module import Mul_module
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.metrics import classification_report
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--modality', help="modalities to work on", default='ecn')
parser.add_argument('--mode', help='fusion mode', default='joint')
parser.add_argument('-e', '--epoches', help='number of epoches', type=int, default=30)
parser.add_argument('-t', '--task', help='task to work on', default='mortality')
parser.add_argument('-r', '--use_ratio', help='whether to use ratio threshold', action='store_true')
parser.add_argument('-b', '--best_test_only', help='run best test only', action='store_true')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_amp = True
# globals().update(dict(args._get_kwargs()))
modality = args.modality
mode = args.mode
epoches = args.epoches
task = args.task
use_ratio = args.use_ratio
best_test_only = args.best_test_only
longstay_mintime = 3*24
print(f'on the {device} device')
print('run {} experiment on {}'.format(mode, modality))
print('task: {}'.format(task))
if use_ratio:
    print('use ratio based threshold')
if task == 'longstay':
    print('longstay_mintime: {}'.format(longstay_mintime))


img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(norm_mean, norm_std),
])

if mode == 'early':
    from dataset.embed_dataset import EMBED_dataset
    train_dataset = EMBED_dataset(split='train', task=task)
    test_dataset = EMBED_dataset(split='test', task=task)
    val_dataset = EMBED_dataset(split='val', task=task)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

else:
    train_dataset = MUL_dataset(split='train',img_transform=img_transform, task=task, longstay_mintime=longstay_mintime)
    test_dataset = MUL_dataset(split='test', img_transform=img_transform, task=task, longstay_mintime=longstay_mintime)
    val_dataset = MUL_dataset(split='val', img_transform=img_transform, task=task, longstay_mintime=longstay_mintime)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.get_collate())
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=test_dataset.get_collate())
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=val_dataset.get_collate())


model = Mul_module(device=device, modality=modality, mode=mode)
model = model.to(device)
if task == 'mortality':
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 10], dtype=torch.float)).to(device)
elif task == 'longstay':
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.5, 1], dtype=torch.float)).to(device)
elif task == 'readmission':
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 25], dtype=torch.float)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-3)
scaler = GradScaler()


def train_epoch(model, device, train_loader, optimizer):
    model.train()
    total_loss = []
    if mode != 'early':
        for batch_idx, ((demo, ce_ts, le_ts, pe_ts, timestamps), (img_list, img_positions, img_times), notes, target) in enumerate(train_loader):
            torch.cuda.empty_cache()
            with autocast(enabled=use_amp):
                img_list = [[y.to(device) for y in x] for x in img_list]
                demo = demo.to(device)
                target = target.to(device)
                pred = model(demo, ce_ts, le_ts, pe_ts, timestamps, img_list, img_positions, img_times, notes)
                loss = criterion(pred, target)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            loss_num = loss.data.item()
            total_loss.append(loss.data * len(target))
            if batch_idx % 50 == 0:
                print('batch [{}/{}] loss: {:.3f}'.format(batch_idx + 1, len(train_loader), loss_num))
    else:
        for batch_idx, (ehr, cxr, notes, target) in enumerate(train_loader):
            torch.cuda.empty_cache()
            with autocast(enabled=use_amp):
                target = target.to(device)
                kwargs = {}
                if 'e' in modality:
                    ehr = ehr.to(device)
                    kwargs.update({'ehr_embed':ehr})
                if 'c' in modality:
                    cxr = cxr.to(device)
                    kwargs.update({'cxr_embed':cxr})
                if 'n' in modality:
                    notes = notes.to(device)
                    kwargs.update({'note_embed':notes})
                pred = model(*([None]*9), **kwargs)
                loss = criterion(pred, target)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            loss_num = loss.data.item()
            total_loss.append(loss.data * len(target))
            if batch_idx % 50 == 0:
                print('batch [{}/{}] loss: {:.3f}'.format(batch_idx + 1, len(train_loader), loss_num))
    
    avg_loss = torch.sum(torch.stack(total_loss)) / len(train_dataset)
    return avg_loss
    
@torch.no_grad()
def val_epoch(model, device, val_loader):
    model.eval()
    all_targets = []
    all_preds = []
    if mode != 'early':
        for batch_idx, ((demo, ce_ts, le_ts, pe_ts, timestamps), (img_list, img_positions, img_times), notes, target) in enumerate(val_loader):
            torch.cuda.empty_cache()
            with autocast(enabled=use_amp):
                img_list = [[y.to(device) for y in x] for x in img_list]
                demo = demo.to(device)
                target = target.to(device)
                pred = model(demo, ce_ts, le_ts, pe_ts, timestamps, img_list, img_positions, img_times, notes)
            all_targets.append(target)
            all_preds.append(pred.to('cpu'))
    else:
        for batch_idx, (ehr, cxr, notes, target) in enumerate(val_loader):
            torch.cuda.empty_cache()
            with autocast(enabled=use_amp):
                target = target.to(device)
                kwargs = {}
                if 'e' in modality:
                    ehr = ehr.to(device)
                    kwargs.update({'ehr_embed':ehr})
                if 'c' in modality:
                    cxr = cxr.to(device)
                    kwargs.update({'cxr_embed':cxr})
                if 'n' in modality:
                    notes = notes.to(device)
                    kwargs.update({'note_embed':notes})
                pred = model(*([None]*9), **kwargs)
            all_targets.append(target)
            all_preds.append(pred.to('cpu'))
    all_targets = torch.cat(all_targets).to('cpu').float().numpy()
    all_preds = torch.cat(all_preds).float()
    all_preds = torch.softmax(all_preds, dim=1)[:, 1].to('cpu').numpy()
    auroc = roc_auc_score(all_targets, all_preds)
    return auroc

@torch.no_grad()
def cal_threshold(model, device, val_loader, ratio=None):
    model.eval()
    if ratio == None:
        return None
    all_preds = []
    if mode != 'early':
        for batch_idx, ((demo, ce_ts, le_ts, pe_ts, timestamps), (img_list, img_positions, img_times), notes, target) in enumerate(val_loader):
            torch.cuda.empty_cache()
            with autocast(enabled=use_amp):
                img_list = [[y.to(device) for y in x] for x in img_list]
                demo = demo.to(device)
                pred = model(demo, ce_ts, le_ts, pe_ts, timestamps, img_list, img_positions, img_times, notes)
            all_preds.append(pred.to('cpu'))
    else:
        for batch_idx, (ehr, cxr, notes, target) in enumerate(val_loader):
            torch.cuda.empty_cache()
            with autocast(enabled=use_amp):
                kwargs = {}
                if 'e' in modality:
                    ehr = ehr.to(device)
                    kwargs.update({'ehr_embed':ehr})
                if 'c' in modality:
                    cxr = cxr.to(device)
                    kwargs.update({'cxr_embed':cxr})
                if 'n' in modality:
                    notes = notes.to(device)
                    kwargs.update({'note_embed':notes})
                pred = model(*([None]*9), **kwargs)
            all_preds.append(pred.to('cpu'))
    all_preds = torch.cat(all_preds).float()
    all_probs = torch.softmax(all_preds, dim=1).to('cpu').numpy()
    pos_prob = all_probs[:, 1]
    neg_num = int(len(pos_prob) * (1 - ratio))
    partition = np.partition(pos_prob, neg_num)
    x1, x2 = np.max(partition[:neg_num]), partition[neg_num]
    return (x1 + x2)/2

@torch.no_grad()
def test(model, device, test_loader, threshold = None):
    model.eval()
    all_targets = []
    all_preds = []
    if mode != 'early':
        for batch_idx, ((demo, ce_ts, le_ts, pe_ts, timestamps), (img_list, img_positions, img_times), notes, target) in enumerate(test_loader):
            torch.cuda.empty_cache()
            with autocast(enabled=use_amp):
                img_list = [[y.to(device) for y in x] for x in img_list]
                demo = demo.to(device)
                target = target.to(device)
                pred = model(demo, ce_ts, le_ts, pe_ts, timestamps, img_list, img_positions, img_times, notes)
            all_targets.append(target)
            all_preds.append(pred.to('cpu'))
    else:
        for batch_idx, (ehr, cxr, notes, target) in enumerate(test_loader):
            torch.cuda.empty_cache()
            with autocast(enabled=use_amp):
                target = target.to(device)
                kwargs = {}
                if 'e' in modality:
                    ehr = ehr.to(device)
                    kwargs.update({'ehr_embed':ehr})
                if 'c' in modality:
                    cxr = cxr.to(device)
                    kwargs.update({'cxr_embed':cxr})
                if 'n' in modality:
                    notes = notes.to(device)
                    kwargs.update({'note_embed':notes})
                pred = model(*([None]*9), **kwargs)
            all_targets.append(target)
            all_preds.append(pred.to('cpu'))
    all_targets = torch.cat(all_targets).to('cpu').float().numpy()
    all_preds = torch.cat(all_preds).float()
    all_probs = torch.softmax(all_preds, dim=1).to('cpu').numpy()
    if threshold == None:
        all_preds = np.argmax(all_probs, axis=1)
    else:
        all_preds = (all_probs[:, 1] >= threshold).astype('int')
    all_probs = all_probs[:, 1]
    auroc = roc_auc_score(all_targets, all_probs)
    precision, recall, t = precision_recall_curve(all_targets, all_probs)
    auprc = auc(recall, precision)
    ap = average_precision_score(all_targets, all_probs)
    report = classification_report(all_targets, all_preds, target_names=['negative', 'positive'])
    positive_num = all_preds.sum()
    return auroc, ap, auprc, report, positive_num

@torch.no_grad()
def best_test(model, device, test_loader, val_loader = None, ratio = None):
    model.load_state_dict(torch.load('./saved_model/best_{}_{}_model_{}.pth'.format(mode, modality, task)))
    if val_loader != None:
        threshold = cal_threshold(model, device, val_loader, ratio)
    else:
        threshold = None
    auroc, ap, auprc, report, positive_num = test(model, device, test_loader, threshold)
    print('test metric -- auroc:{:.3f}'.format(auroc))
    print('test metric -- ap:{:.3f}'.format(ap))
    print('test metric -- auprc:{:.3f}'.format(auprc))
    print('test metric -- predicted positive:{}'.format(positive_num))
    print('test metric -- report:\n{}'.format(report))
    return auroc, ap, auprc, report, positive_num


def train(model, device, train_loader, val_loader, test_loader, optimizer, epoch, ratio = None):
    best_roc = 0
    for epoch_idx in tqdm(range(epoch)):
        print('Epoch [{}/{}] '.format(epoch_idx + 1, epoch))
        epoch_loss = train_epoch(model, device, train_loader, optimizer)
        torch.cuda.empty_cache()
        print('Epoch [{}/{}] loss:{:.3f}'.format(epoch_idx + 1, epoch, epoch_loss))

        auroc = val_epoch(model, device, val_loader)
        torch.cuda.empty_cache()

        if auroc > best_roc:
            print('new best auroc: {} -> {}'.format(best_roc, auroc))
            best_roc = auroc
            print('model saved.')
            torch.save(model.state_dict(), './saved_model/best_{}_{}_model_{}.pth'.format(mode, modality, task))

    best_test(model, device, test_loader, val_loader, ratio)

if __name__ == "__main__":
    if use_ratio:
        ratio = ((train_dataset.label.sum() + val_dataset.label.sum())/(len(train_dataset) + len(val_dataset))).item()
    else:
        ratio = None
    
    if best_test_only:
        best_test(model, device, test_loader, val_loader, ratio)
    else:
        train(model, device, train_loader, val_loader, test_loader, optimizer, epoches, ratio)
    # best_test(model, device, test_loader, val_loader, ratio)
