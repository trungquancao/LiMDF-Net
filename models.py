import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_BACKBONES = [
    "mobilenetv4_conv_small",
    "mobilenetv4_conv_medium",
    "mobilenetv4_conv_large"
]
SUPPORTED_FUSIONS = ["simple_concat", "hadamard", "cross_attention", "self_attention"]

BASE_IMAGE_DIM: int = 512     
BASE_CLINICAL_DIM: int = 256   

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.pow(1. / self.p)
        return x.view(x.size(0), -1) 


class ClinicalMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleConcatFusion(nn.Module):
    def __init__(self, img_dim: int, clin_dim: int):
        super().__init__()
        self.output_dim = img_dim + clin_dim

    def forward(self, img_feat: torch.Tensor, clin_feat: torch.Tensor) -> torch.Tensor:
        return torch.cat([img_feat, clin_feat], dim=1) 

class HadamardFusion(nn.Module):
    def __init__(self, img_dim: int, clin_dim: int):
        super().__init__()
        self.output_dim = clin_dim                     
        self.proj_img = nn.Linear(img_dim, clin_dim)

    def forward(self, img_feat: torch.Tensor, clin_feat: torch.Tensor) -> torch.Tensor:
        i = self.proj_img(img_feat)     
        return i * clin_feat                

class CrossAttentionFusion(nn.Module):
    def __init__(self, img_dim: int, clin_dim: int):
        super().__init__()
        self.d_model = clin_dim
        
        self.W_Q = nn.Linear(img_dim, self.d_model)   # Query from Image
        self.W_K = nn.Linear(clin_dim, self.d_model)  # Key from Clinical
        self.W_V = nn.Linear(clin_dim, self.d_model)  # Value from Clinical
        
        self.output_dim = img_dim + self.d_model

    def forward(self, img_feat: torch.Tensor, clin_feat: torch.Tensor) -> torch.Tensor:
        X = img_feat.unsqueeze(1)    # [B, 1, img_dim]
        Y = clin_feat.unsqueeze(1)   # [B, 1, clin_dim]

        Qx = self.W_Q(X)             # [B, 1, d_model]
        Ky = self.W_K(Y)             # [B, 1, d_model]
        Vy = self.W_V(Y)             # [B, 1, d_model]

        scale = math.sqrt(self.d_model)
        attn_scores = torch.bmm(Qx, Ky.transpose(1, 2)) / scale  
        
        attn_weights = torch.sigmoid(attn_scores)                
        
        Z_seq = torch.bmm(attn_weights, Vy)                      
        Z = Z_seq.squeeze(1)                                     
        
        return torch.cat([img_feat, Z], dim=1)                 

class SelfAttentionFusion(nn.Module):
    def __init__(self, img_dim: int, clin_dim: int, n_heads: int = 8):
        super().__init__()
        self.output_dim = img_dim + clin_dim

        img_heads = n_heads if img_dim % n_heads == 0 else 4
        clin_heads = 4 if clin_dim % 4 == 0 else 2

        def _build_sa(dim, heads):
            adj_dim = (dim // heads) * heads 
            return nn.Linear(dim, adj_dim), nn.Linear(dim, adj_dim), nn.Linear(dim, adj_dim), adj_dim // heads

        self.img_Wq, self.img_Wk, self.img_Wv, self.img_hd = _build_sa(img_dim, img_heads)
        self.clin_Wq, self.clin_Wk, self.clin_Wv, self.clin_hd = _build_sa(clin_dim, clin_heads)
        
        self.img_heads = img_heads
        self.clin_heads = clin_heads

    @staticmethod
    def _self_attn(x: torch.Tensor, Wq, Wk, Wv, n_heads: int, head_dim: int) -> torch.Tensor:
        B = x.size(0)
        Q = Wq(x).view(B, n_heads, head_dim)
        K = Wk(x).view(B, n_heads, head_dim)
        V = Wv(x).view(B, n_heads, head_dim)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(head_dim)  
        attended = torch.bmm(F.softmax(scores, dim=-1), V)                     
        return attended.reshape(B, n_heads * head_dim)           

    def forward(self, img_feat: torch.Tensor, clin_feat: torch.Tensor) -> torch.Tensor:
        img_sa = self._self_attn(img_feat, self.img_Wq, self.img_Wk, self.img_Wv, self.img_heads, self.img_hd)   
        clin_sa = self._self_attn(clin_feat, self.clin_Wq, self.clin_Wk, self.clin_Wv, self.clin_heads, self.clin_hd)  
        
        img_sa = F.pad(img_sa, (0, img_feat.shape[1] - img_sa.shape[1])) + img_feat
        clin_sa = F.pad(clin_sa, (0, clin_feat.shape[1] - clin_sa.shape[1])) + clin_feat

        return torch.cat([img_sa, clin_sa], dim=1)


def _build_fusion_block(fusion_name: str, img_dim: int, clin_dim: int) -> nn.Module:
    if fusion_name == "simple_concat":
        return SimpleConcatFusion(img_dim, clin_dim)
    elif fusion_name == "hadamard":
        return HadamardFusion(img_dim, clin_dim)
    elif fusion_name == "cross_attention":
        return CrossAttentionFusion(img_dim, clin_dim)
    elif fusion_name == "self_attention":
        return SelfAttentionFusion(img_dim, clin_dim)
    else:
        raise ValueError(f"Unknown fusion '{fusion_name}'. Choose from {SUPPORTED_FUSIONS}")

def _detect_backbone_output_dim(backbone: nn.Module, img_size: int = 256) -> int:
    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size)
        out = backbone(dummy)
        dim = out.shape[1] if out.dim() == 4 else out.shape[-1]
    backbone.train(was_training)
    return int(dim)


class MultimodalSkinLesionNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        clinical_dim: int = 81, 
        alpha: float = 1.0, 
        fusion_name: str = "simple_concat",
        use_gem: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.use_gem = use_gem
        
        if alpha <= 1.0:
            self.backbone_name = "mobilenetv4_conv_small"
        elif alpha <= 1.5:
            self.backbone_name = "mobilenetv4_conv_medium"
        else:
            self.backbone_name = "mobilenetv4_conv_large"

        self.img_dim = int((BASE_IMAGE_DIM * alpha) // 8 * 8)
        self.clin_dim = int((BASE_CLINICAL_DIM * alpha) // 8 * 8)
        
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',      
            drop_path_rate=0.1,
        )

        if self.use_gem:
            self.global_pool = GeM()
            pool_name = "GeM"
        else:
            self.global_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            pool_name = "GAP"

        raw_feat_dim = _detect_backbone_output_dim(self.backbone)
        
        self.img_projector = nn.Sequential(
            nn.Linear(raw_feat_dim, self.img_dim),
            nn.BatchNorm1d(self.img_dim),
            nn.ReLU(inplace=True),
        )

        self.clinical_mlp = ClinicalMLP(
            in_features=clinical_dim,
            out_features=self.clin_dim,
        )

        self.fusion = _build_fusion_block(fusion_name, self.img_dim, self.clin_dim)
        fusion_out_dim = self.fusion.output_dim

        clf_hidden_1 = int((256 * alpha) // 8 * 8)
        clf_hidden_2 = int((128 * alpha) // 8 * 8)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_out_dim, clf_hidden_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(clf_hidden_1, clf_hidden_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(clf_hidden_2, num_classes),
        )

    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        img_feat = self.backbone(image)               
        img_feat = self.global_pool(img_feat)         
        img_feat = self.img_projector(img_feat)       

        clin_feat = self.clinical_mlp(clinical)       
        
        fused = self.fusion(img_feat, clin_feat)
        return self.classifier(fused)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())