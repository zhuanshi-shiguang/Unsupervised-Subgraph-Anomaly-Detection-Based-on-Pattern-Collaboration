from operator import attrgetter
from numba.core.cgutils import if_zero
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GINConv, global_add_pool
from GAE.encoder import *
from GAE.decoder import *
from GAE.model import *
from math import *
from GAE.smgnn import *
from GAE.dominant import *
import GCL.losses as L
import GCL.augmentors as A
from GCL.augmentors.augmentor import Graph
from torch_geometric.nn import GATConv, GINConv,global_add_pool, global_mean_pool, global_max_pool
from GCL.models import DualBranchContrast
from torch_scatter import scatter_add
from torch_geometric.utils import degree
def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))
class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=32):
        super(GConv, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # 1. 自适应特征预处理
        self.feature_adapter = FeatureAdapter(input_dim, hidden_dim)
        self.initial_proj = nn.Linear(input_dim, hidden_dim)
        # 2. 基础图卷积层 - 保留原始结构
        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()

        for i in range(num_layers):
            conv = GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
            self.conv_layers.append(conv)
            self.attention_layers.append(AdaptiveAttention(hidden_dim))

            # 自适应归一化
            if hidden_dim > 128:
                self.norms.append(nn.InstanceNorm1d(hidden_dim))
            else:
                self.norms.append(nn.BatchNorm1d(hidden_dim))

        # 4. 输出投影层 - 与原始保持一致
        project_dim = hidden_dim * num_layers
        self.project = nn.Sequential(
            nn.Linear(project_dim, project_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(project_dim // 2, project_dim // 4)
        )
        self.fusion = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, batch):
        # 自适应特征预处理
        z = self.feature_adapter(x)
        #z = self.initial_proj(x)
        deg = degree(edge_index[0], num_nodes=x.size(0), dtype=torch.float)
        deg = deg.unsqueeze(-1) / (deg.max() + 1e-8)
        zs = []
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norms)):
            # 图卷积操作
            conv_out = conv(z, edge_index)
            attn_out = self.attention_layers[i](z, edge_index, deg)
            combined = torch.cat([conv_out, attn_out], dim=-1)
            z = self.fusion[i](combined)
            ##z = F.relu(z)
            z = norm(z)
            zs.append(z)

        # 多尺度特征聚合
        z_out = torch.cat(zs, dim=-1)

        # 图级表示
        if batch is not None:
            gs = [global_add_pool(z, batch) for z in zs]
            g_out = torch.cat(gs, dim=-1)
        return z_out, g_out

class AdaptiveAttention(nn.Module):
    """自适应注意力机制（支持高维特征）"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 简化边信息编码
        self.edge_encoder = nn.Linear(1, 1) if hidden_dim > 128 else nn.Identity()

        # 高效注意力机制
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)

        # 特征压缩层（针对高维）
        if hidden_dim > 256:
            self.value = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.GELU(),
                nn.Linear(256, hidden_dim)
            )
        else:
            self.value = nn.Linear(hidden_dim, hidden_dim)

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, deg):
        # 节点特征投影
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)  # 可能经过压缩

        # 边特征编码
        edge_bias = self.edge_encoder(deg) if isinstance(self.edge_encoder, nn.Linear) else deg

        # 注意力分数计算
        row, col = edge_index
        attn_scores = (q[row] * k[col]).sum(dim=-1) / (self.hidden_dim ** 0.5)

        # 添加边特征偏置
        edge_bias_scalar = edge_bias[row].squeeze()
        attn_scores = attn_scores + edge_bias_scalar

        # 注意力权重归一化
        attn_weights = torch.sigmoid(attn_scores).unsqueeze(1)

        # 聚合邻居信息
        neighbor_agg = torch.zeros_like(x)
        neighbor_agg = neighbor_agg.index_add_(0, row, v[col] * attn_weights)

        # 门控融合
        gate = self.gate(torch.cat([x, neighbor_agg], dim=-1))
        out = gate * x + (1 - gate) * neighbor_agg

        return out

class FeatureAdapter(nn.Module):
    """自适应特征适配器 - 高低维通用"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 低维路径：特征增强
        self.low_dim_path = nn.Sequential(
            nn.Linear(input_dim, max(128, input_dim * 4)),
            nn.GELU(),
            nn.Linear(max(128, input_dim * 4), output_dim)
        )if input_dim<100 else nn.Identity()

        # 高维路径：特征压缩
        self.high_dim_path = nn.Sequential(
            nn.Linear(input_dim, min(256, input_dim // 2)),
            nn.GELU(),
            nn.Linear(min(256, input_dim // 2), output_dim)
        )if input_dim>=100 else nn.Identity()


        # 残差连接
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

        # 特征门控
        self.gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 低维处理路径
        n_samples, n_features = x.shape

        # 计算稀疏率
        non_zero_count = torch.count_nonzero(x).item()  # 获取 Python 标量值

        # 计算总元素数
        total_elements = n_samples * n_features

        # 计算稀疏率
        sparsity = 1.0 - (non_zero_count / total_elements)

        residual = self.residual(x)
        low_out = self.low_dim_path(x) if isinstance(self.low_dim_path, nn.Sequential) else 0
        high_out = self.high_dim_path(x) if isinstance(self.high_dim_path, nn.Sequential) else 0
        if sparsity>0.05:
            return low_out + residual
        else :
            return high_out + residual

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.encoder.to("cuda:0")

    def forward(self, x, edge_index, batch, cycle_edges_list=None,
                tree_root_list=None, path_middle_list=None, one_degree_list=None):
        x.to("cuda:0")
        edge_index.to("cuda:0")
        aug1, aug2 = self.augmentor
        if cycle_edges_list != None or tree_root_list != None or \
                tree_root_list != None or path_middle_list != None:
            if isinstance(aug1, A.SubIncreasing) or isinstance(aug1, A.SubDecreasing):
                x1, edge_index1, edge_weight1, batch1 = \
                    aug1.augment(Graph(x, edge_index, None), batch, cycle_edges_list,
                                 tree_root_list, path_middle_list, one_degree_list)

                z1, g1 = self.encoder(x1, edge_index1, batch1)
            else:
                x1, edge_index1, edge_weight1 = aug1(x, edge_index)
                z1, g1 = self.encoder(x1, edge_index1, batch)

            if isinstance(aug2, A.SubIncreasing) or isinstance(aug2, A.SubDecreasing):
                x2, edge_index2, edge_weight2, batch2 = \
                    aug2.augment(Graph(x, edge_index, None), batch, cycle_edges_list,
                                 tree_root_list, path_middle_list, one_degree_list)

                z2, g2 = self.encoder(x2, edge_index2, batch2)
            else:
                x2, edge_index2, edge_weight2 = aug2(x, edge_index)
                z2, g2 = self.encoder(x2, edge_index2, batch)

        else:
            z1, z2, g1, g2 = None, None, None, None

        z, g = self.encoder(x, edge_index, batch)


        return z, g, z1, z2, g1, g2

class ContrastiveModel(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim):
        super(ContrastiveModel, self).__init__()
        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(input_dim2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, y):
        x, y = F.normalize(x, p=2, dim=0), F.normalize(y, p=2, dim=0)
        h1 = F.relu(self.fc1(x)+self.fc2(y))
        h2 = self.fc3(h1)
        return h2

def initialize_model(args):

    # GAE parameters
    gae_out_channels, gae_num_features, gae_num_layer = \
        args.gae_out_channels, args.gae_num_features, args.gae_num_layer
    gae_embedding_channels, gae_hidden_channels = \
    args.gae_embedding_channels, args.gae_hidden_channels

    # GAE model
    GAEmodel = DOMINANT_MODEL(gae_num_features, gae_hidden_channels, gae_num_layer, nn.ReLU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    GAEmodel = GAEmodel.to(device)

    # GCL parameters
    aug1, aug2 = args.aug1, args.aug2
    gcl_input_dim, gcl_hidden_dim, gcl_num_layer, gcl_output_dim = \
        args.gcl_input_dim, args.gcl_hidden_dim, args.gcl_num_layer, args.gcl_output_dim

    # GCL
    gconv = GConv(input_dim=gcl_input_dim, hidden_dim=gcl_hidden_dim, num_layers=gcl_num_layer,
                  output_dim=gcl_output_dim).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    #contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
    contrast_model = ContrastiveModel(gcl_output_dim, gcl_output_dim, hidden_dim=32).to(device)
    GCLmodel = [encoder_model, contrast_model]

    # inizialize optimizers
    opt_gae = torch.optim.Adam(GAEmodel.parameters(), lr=args.gae_lr)
    opt_gcl = Adam(encoder_model.parameters(), lr=args.gcl_lr)

    return GAEmodel, GCLmodel, opt_gae, opt_gcl

class ImprovedGAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4):
        super(ImprovedGAE, self).__init__()
        # 多尺度特征提取
        self.conv1_gcn = GCNConv(in_channels, hidden_channels)
        self.conv1_gat = GATConv(in_channels, hidden_channels // num_heads, heads=num_heads)

        self.conv2_gcn = GCNConv(hidden_channels, out_channels)
        self.conv2_gat = GATConv(hidden_channels, out_channels // num_heads, heads=num_heads)

        # 特征融合层
        self.fusion = torch.nn.Linear(out_channels * 2, out_channels)

        # 图注意力机制
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(out_channels, out_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels // 2, 1),
            torch.nn.Sigmoid()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(out_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, in_channels)
        )
        self.multihead_attention = MultiHeadAttention(out_channels, num_heads=4)
        # 边预测层
        self.edge_predictor = torch.nn.Sequential(
            torch.nn.Linear(out_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Sigmoid()
        )
        self.mask_predictor = torch.nn.Sequential(
            torch.nn.Linear(out_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, in_channels)
        )

    def encode(self, x, edge_index):
        # 多尺度特征提取
        temp = self.conv1_gcn(x, edge_index)
        h_gcn1 = F.relu(temp)
        h_gat1 = F.relu(self.conv1_gat(x, edge_index))

        h_gcn2 = self.conv2_gcn(h_gcn1, edge_index)
        h_gat2 = self.conv2_gat(h_gat1, edge_index)

        # 特征融合
        h = torch.cat([h_gcn2, h_gat2], dim=1)
        h = self.fusion(h)

        # 应用图注意力机制
        # attn_weights = self.attention(h)
        # h = h * attn_weights
        h = self.multihead_attention(h.unsqueeze(0)).squeeze(0)
        return h


    def recon_loss(self, z, edge_index, neg_edge_index=None):
        # 重构损失计算
        #pos_pred = self.decode(z, pos_edge_index)
        x_reconstructed = self.decoder(z)
        # 邻接矩阵重构 (如果需要)
        if edge_index is not None:
            adj_reconstructed = torch.sigmoid(z @ z.t())
        return x_reconstructed,adj_reconstructed

class MultiHeadAttention(torch.nn.Module):
    """多头注意力机制，并行学习不同的关系模式"""

    def __init__(self, input_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # 为每个注意力头创建独立的变换矩阵
        self.attn_heads = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, self.head_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.head_dim, 1),
                torch.nn.Sigmoid()
            )
            for _ in range(num_heads)
        ])

        # 用于融合多头注意力结果的投影层
        self.projection = torch.nn.Linear(num_heads * input_dim, input_dim)

    def forward(self, x):
        """
        计算多头注意力权重并应用到输入特征
        """
        batch_size, seq_len, _ = x.size() if x.dim() == 3 else (1, x.size(0), x.size(1))

        # 对每个注意力头计算注意力权重
        head_outputs = []
        for head in self.attn_heads:
            # 计算注意力权重
            attn_weights = head(x)
            # 应用注意力权重
            weighted_features = x * attn_weights
            head_outputs.append(weighted_features)

        # 拼接多头注意力结果
        combined = torch.cat(head_outputs, dim=-1)

        # 投影回原始维度
        if batch_size == 1:
            output = self.projection(combined)
        else:
            output = self.projection(combined.view(batch_size, seq_len, -1))

        return output

