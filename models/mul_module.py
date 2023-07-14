import torch
import torch.nn as nn
from models.ehr_module import EHR_module
from models.cxr_module import CXR_module
from models.note_module import Note_module

class Mul_module(nn.Module):
    def __init__(self, device='cpu', embed_size=512, mode='joint', modality='ecn'):
        super(Mul_module, self).__init__()
        self.device=device
        self.mode = mode
        self.embed_size = embed_size
        self.modality = modality
        if 'e' in modality:
            self.ehr = EHR_module(device=device, embed_size=embed_size // 2)
        if 'c' in modality:
            self.cxr = CXR_module(device=device, embed_size=embed_size)
        if 'n' in modality:
            self.note = Note_module(device=device, embed_size=embed_size)
        
        if mode == 'joint':
            self.cls_head = nn.Linear(len(modality) * embed_size, 2)
        elif mode == 'late':
            if 'e' in modality:
                self.cls_head_ehr = nn.Linear(embed_size, 2)
            if 'c' in modality:
                self.cls_head_cxr = nn.Linear(embed_size, 2)
            if 'n' in modality:
                self.cls_head_note = nn.Linear(embed_size, 2)
        else:
            # 2-layer mlp classifier
            self.cls_head = nn.Sequential(nn.Linear(len(modality) * embed_size, embed_size),
                                          nn.Linear(embed_size, 2))
            # linear classifier
            # self.cls_head = nn.Linear(len(modality) * embed_size, 2)
        self.pooling = nn.AdaptiveMaxPool1d(1)
    
    def embedding(self, demo, chart, lab, procedure, time, imgs, view_points, timestamps, notes):
        embeds = []
        if 'e' in self.modality:
            ehr_embeds = self.ehr.embedding(demo, chart, lab, procedure, time, pooling=True)
            embeds.append(ehr_embeds)
        if 'c' in self.modality:
            cxr_embeds = self.cxr.embedding(imgs, view_points, timestamps, pooling=True)
            embeds.append(cxr_embeds)
        if 'n' in self.modality:
            note_embeds = self.note.embedding(notes)
            embeds.append(note_embeds)
        embeds = torch.cat(embeds, dim=1)
        return embeds

    def forward(self, demo, chart, lab, procedure, time, imgs, view_points, timestamps, notes, ehr_embed=None, cxr_embed=None, note_embed=None):
        if self.mode == 'joint':
            embeds = self.embedding(demo, chart, lab, procedure, time, imgs, view_points, timestamps, notes)
            pred = self.cls_head(embeds)
        if self.mode == 'early':
            embeds = []
            if 'e' in self.modality:
                embeds.append(ehr_embed)
            if 'c' in self.modality:
                embeds.append(cxr_embed)
            if 'n' in self.modality:
                embeds.append(note_embed)
            pred = self.cls_head(torch.cat(embeds, dim=1))
        if self.mode == 'late':
            embeds = self.embedding(demo, chart, lab, procedure, time, imgs, view_points, timestamps, notes)
            embeds = embeds.split(self.embed_size, dim=-1)
            preds = []
            for i, m in enumerate(self.modality):
                if m == 'e':
                    pred = self.cls_head_ehr(embeds[i])
                    preds.append(pred)
                    continue
                if m == 'c':
                    pred = self.cls_head_cxr(embeds[i])
                    preds.append(pred)
                    continue
                if m == 'n':
                    pred = self.cls_head_note(embeds[i])
                    preds.append(pred)
                    continue
            pred = torch.mean(torch.stack(preds), dim=0)
        
        return pred