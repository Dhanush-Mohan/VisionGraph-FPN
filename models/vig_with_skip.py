import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib import Grapher, act_layer
from torch.utils.tensorboard import SummaryWriter

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torchviz import make_dot
from torchinfo import summary
from torchview import draw_graph

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'gnn_patch16_224': _cfg(
        crop_pct=0.9, input_size=(3, 224, 224),
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # Expands the channel dimension to hidden_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        # Projects the channels back to out_features
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    # Stores the input as a shortcut (residual connection).
    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class Upsample(nn.Module):
    """ Transpose Convolution-based downsample
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Stem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        # print("Shapeeee:", x.size())
        # Shapeeee: torch.Size([1, 192, 14, 14])
        return x


class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        
        self.stem = Stem(out_dim=channels, act=act)
        # print("Stem:",self.stem)

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)
        max_dilation = 196 // max(num_knn)
        self.final_one_conv = nn.Conv2d(192,2048,1,bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))
        print("Number of blocks:",self.n_blocks)
        self.reverse_backbone = nn.ModuleList([])
        self.skip_conn = []
        self.one_conv = nn.Conv2d(channels*2,channels,1,bias=True)


        if opt.use_dilation:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                    FFN(channels, channels * 4, act=act, drop_path=dpr[i]),
                                    nn.Conv2d(channels,channels,1,bias=True)
                                    ) for i in range(self.n_blocks)])
            
            for i in range(self.n_blocks):
                self.reverse_backbone += [Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                    FFN(channels, channels * 4, act=act, drop_path=dpr[i]),
                                    nn.Conv2d(channels,channels,1,bias=True)
                                    )]
                if i%4==3:
                    self.reverse_backbone.append(Upsample(channels,channels))
                    skip_connList = nn.ModuleList([])
                    for l in range(i//4+1):
                        skip_connList.append(Upsample(channels,channels))
                    self.skip_conn.append(skip_connList)
        else:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                    FFN(channels, channels * 4, act=act, drop_path=dpr[i]),
                                    nn.Conv2d(channels,channels,1,bias=True)
                                    ) for i in range(self.n_blocks)])
            self.reverse_backbone = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                    FFN(channels, channels * 4, act=act, drop_path=dpr[i]),
                                    nn.Conv2d(channels,channels,1,bias=True)
                                    ) for i in range(self.n_blocks)])

        # self.prediction = Seq(nn.Conv2d(channels, 1024, 1, bias=True),
        #                     nn.BatchNorm2d(1024),
        #                     act_layer(act),
        #                     nn.Dropout(opt.dropout),
        #                     nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.reverse_backbone = Seq(*self.reverse_backbone)
        self.skip_conn = Seq(*self.skip_conn)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        # print(f"Stem Output Shape: {x.shape}")  
        B, C, H, W = x.shape
        skip_ops = []

        # Forward pass through backbone
        for i in range(self.n_blocks):
            x = self.backbone[i](x)
            # print(f"After Backbone Block {i}: {x.shape}")  
            if i % 4 == 3 and i > 1:
                skip_ops.append(x)

        skip_count = 0
        # print(f"Total Reverse Backbone Blocks: {len(self.reverse_backbone)}")

        # Reverse pass through reverse_backbone
        for i in range(len(self.reverse_backbone)):
            x = self.reverse_backbone[i](x)
            # print(f"After Reverse Backbone Block {i}: {x.shape}")  

            if i % 5 == 4 and i > 1:
                y = skip_ops[skip_count]
                skip_count += 1
                sc_list = self.skip_conn[i // 5]
                for ups in sc_list:
                    y = ups(y)

                # print(f"Skip Connection Applied at Block {i}: {y.shape}")
                x = torch.cat((x, y), 1)
                # print(f"After Concatenation at Block {i}: {x.shape}")

                x = self.one_conv(x)
                # print(f"After one_conv at Block {i}: {x.shape}")

        # **Fix Output Size with Adaptive Pooling**
        x = torch.nn.functional.adaptive_avg_pool2d(x, (7, 7))
        # print(f"After Adaptive Pooling: {x.shape}")

        final_output = self.final_one_conv(x)
        # print(f"Final Output Shape: {final_output.shape}")  
        return final_output




@register_model
def vig_ti_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 12 # number of basic blocks in the backbone
            self.n_filters = 192 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


@register_model
def vig_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 16 # number of basic blocks in the backbone
            self.n_filters = 320 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


@register_model
def vig_b_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 16 # number of basic blocks in the backbone
            self.n_filters = 640 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = vig_ti_224_gelu().to(device)

    model.eval()
    ip = torch.rand(1, 3, 224, 224)
    op = model(ip)
    print(op.shape)
    # model.train()
    # model_summary = model
    # summary(model=model_summary,
    #         input_size=(32, 3, 224, 224), # (batch_size, num_patches, embedding_dimension)
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"])
    # make_dot(op.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render("attached", format="png")
    # model_graph = draw_graph(model, input_size=(1,3,224,224), expand_nested=True)
    # model_graph.resize_graph(scale=5.0)
    # model_graph.visual_graph.render(format='svg')
    # writer = SummaryWriter('runs/fpn_model')
    # writer.add_graph(model,ip)
    # writer.close()
    
    # model2 = vig_ti_224_gelu().to(device)
    # model2 = torch.load('vig.pt')
    # model2.load_state_dict(torch.load('vig_ti_74.5.pth',map_location=torch.device('cpu')))

