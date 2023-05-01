import torch
import torch.nn as nn

class CXR_module(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, embed_size=512, device='cpu'):
        super(CXR_module, self).__init__()
        view_points = ['antero-posterior', 'left lateral', 'postero-anterior', 'lateral']
        self.view_points = view_points
        if backbone == 'resnet18':
            from torchvision.models import resnet18
            self.res_antero_posterior = resnet18(pretrained=pretrained)
            self.res_left_lateral = resnet18(pretrained=pretrained)
            self.res_postero_anterior = resnet18(pretrained=pretrained)
            self.res_lateral = resnet18(pretrained=pretrained)
            self.backbone = {'antero-posterior': self.res_antero_posterior,
                             'left lateral': self.res_left_lateral,
                             'postero-anterior': self.res_postero_anterior,
                             'lateral': self.res_lateral}
            for k, v in self.backbone.items():
                v.fc = nn.Linear(v.fc.in_features, embed_size)
        else:
            raise NotImplementedError('unsupported backbone type: {}'.format(backbone))
        
        self.cls_head = nn.Linear(embed_size, 2)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.device = device

    def embedding(self, imgs, view_points, timestamps, pooling=True):
        out_batch = []
        for img_set, view_point_set, timestamp_set in zip(imgs, view_points, timestamps):
            viewpoint_out = dict()
            for img, view_point, timestamp in zip(img_set, view_point_set, timestamp_set):
                if view_point not in self.view_points:
                    continue
                img = torch.cat([img] * 3)
                embed = self.backbone[view_point](torch.unsqueeze(img, dim=0))
                if view_point in viewpoint_out.keys():
                    viewpoint_out[view_point].append((embed, timestamp))
                else:
                    viewpoint_out[view_point] = [(embed, timestamp)]
            
            viewpoint_out_list = []
            timestamp_list = []
            for view_point, embeds in viewpoint_out.items():
                embed, timestamp = zip(*embeds)
                embed = torch.cat(list(embed), dim=0)
                timestamp = torch.tensor(list(timestamp)).to(self.device)

                timestamp = torch.softmax(timestamp, dim=0)
                if pooling:
                    embed = torch.sum(embed * timestamp.reshape(-1,1), dim=0)
                else:
                    #TODO: fix multi viewpoint version
                    timestamp_list.append(timestamp)
                viewpoint_out_list.append(embed)
            
            if pooling:
                out = torch.stack(viewpoint_out_list)
                out = out.T.unsqueeze(0)
                out = self.pooling(out)
                out = out.reshape(-1)
            else:
                #TODO: fix multi viewpoint version
                out = (viewpoint_out_list[0], timestamp_list[0])
            out_batch.append(out)
        if pooling:
            out_batch = torch.stack(out_batch)
            return out_batch
        else:
            return tuple(map(list, zip(*out_batch)))

    def forward(self, imgs, view_points, timestamps):
        embed = self.embedding(imgs, view_points, timestamps)
        out = self.cls_head(embed.float())
        return out


