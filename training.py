import os
from utils import *
from tqdm import tqdm
from initializer import *
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import f1_score, roc_auc_score
from pyod.models.copod import COPOD
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Times New Roman", "serif"] # 调整各类文本的字体大小
plt.rcParams["font.size"] = 18  # 全局默认字体大小
plt.rcParams["axes.labelsize"] = 18  # 坐标轴标签字体大小
plt.rcParams["axes.titlesize"] = 18  # 标题字体大小
plt.rcParams["legend.fontsize"] = 18  # 图例字体大小
import networkx as nx

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def filter_subgraph(args, data, anomaly_score):

    # obtain the candidate subgraph
    threshold = np.percentile(anomaly_score, q=args.q)
    candi_groups, residal_data, sub_size = GraphProcessor().sample_sub(data, anomaly_score, threshold)
    return residal_data, candi_groups, sub_size

def train_GAE(args, data, model, optimizer=None, is_training=True):
    x, edge_index, A = data.x, data.edge_index, data.A
    if is_training:
        model.train()
        with tqdm(total=args.gcl_epochs, desc='(GAE)') as pbar:
            for epoch in range(1, args.gae_epochs):

                stru_recon, attr_recon = model(x, edge_index)
                stru_score = torch.square(stru_recon**5   - A).sum(1)
                # stru_score = (torch.square(stru_recon - A).sum(1) + torch.square(stru_reconp - Ap).sum(1))/2
                attr_score = torch.square(attr_recon - x).sum(1)
                score = args.alpha * stru_score + (1 - args.alpha) * attr_score
                loss = score.mean()
                total_error = score.clone().detach().cpu().numpy()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()

    else:
        with torch.no_grad():
            A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
            stru_recon, attr_recon = model(data.x, data.edge_index)
            stru_score = torch.square(stru_recon**5   - A).sum(1).sqrt()
            attr_score = torch.square(attr_recon - data.x).sum(1).sqrt()
            score = args.alpha * stru_score + (1 - args.alpha) * attr_score
            total_error = score.detach().cpu().numpy()
    return total_error

def train_imGAE(args,model, data, optimizer, is_training):
    if is_training:
        model.train()
        with tqdm(total=args.gcl_epochs, desc='(GAE)') as pbar:
            for epoch in range(1, args.gae_epochs):
                data=data.to("cuda:0")
                z = model.encode(data.x, data.edge_index)
                attr_recon,stru_recon = model.recon_loss(z, data.edge_index)
                stru_score = torch.square(stru_recon**5  - data.A).sum(1)
                attr_score = torch.square(attr_recon - data.x).sum(1)
                score = args.alpha * stru_score + (1 - args.alpha) * attr_score
                loss = score.mean()
                total_error = score.clone().detach().cpu().numpy()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()
    else:
        with torch.no_grad():
            A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
            z = model.encode(data.x, data.edge_index)
            attr_recon,stru_recon  = model.recon_loss(z, data.edge_index)
            stru_score = torch.square(stru_recon**5  - A).sum(1).sqrt()
            attr_score = torch.square(attr_recon - data.x).sum(1).sqrt()
            score = args.alpha * stru_score + (1 - args.alpha) * attr_score
            total_error = score.detach().cpu().numpy()
    return total_error

def train_GCL(args, encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    optimizer.zero_grad()

    batch_list, cycle_edges_list, tree_root_list, path_middle_list, one_degree_list =\
    dataloader[0], dataloader[1], dataloader[2], dataloader[3], dataloader[4]
    kl_weight = args.kl_weight if hasattr(args, 'kl_weight') else 0.5  # 默认权重
    for idx in range(len(batch_list)):
        data = batch_list[idx]
        data = data.to("cuda:0")
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        z0, g0, z1, z2, g1, g2 = encoder_model(data.x, data.edge_index, data.batch, cycle_edges_list[idx],
                                           tree_root_list[idx], path_middle_list[idx], one_degree_list[idx])
        g0, g1, g2 = [encoder_model.encoder.project(g) for g in [g0, g1, g2]]


        def compute_kl_loss(p, q, eps=1e-12):
            p_dist = F.softmax(p, dim=-1) + eps
            q_dist = F.softmax(q, dim=-1) + eps
            kl_loss = F.kl_div(p_dist.log(), q_dist, reduction='batchmean', log_target=False) + \
                      F.kl_div(q_dist.log(), p_dist, reduction='batchmean', log_target=False)
            return kl_loss / 2.0

        # 计算视图间的KL散度
        kl_loss = compute_kl_loss(g1, g2) + compute_kl_loss(g0, g1)+compute_kl_loss(g0, g2)

        # approximate mutual information via a model
        inner_epochs = args.inner_epochs
        optimizer_local = torch.optim.Adam(contrast_model.parameters(), lr=args.inner_lr)
        for j in range(0, inner_epochs):
            optimizer_local.zero_grad()

            shuffle_g0, shuffle_g1, shuffle_g2 = g0[torch.randperm(g0.shape[0])], g1[torch.randperm(g1.shape[0])], g2[torch.randperm(g2.shape[0])]
            joint1, joint2 = contrast_model(g1, g2), contrast_model(g0, g1)
            margin1, margin2= contrast_model(g1, shuffle_g2), contrast_model(g0, shuffle_g1)
            mi = - (torch.mean(joint1) - torch.log(torch.mean(torch.exp(margin1)))) + \
                 (torch.mean(joint2) - torch.log(torch.mean(torch.exp(margin2))))

            local_loss = mi+ kl_weight * kl_loss
            #local_loss = mi
            local_loss.backward(retain_graph=True)
            optimizer_local.step()

        shuffle_g0, shuffle_g1, shuffle_g2 = g0[torch.randperm(g0.shape[0])], g1[torch.randperm(g1.shape[0])], g2[torch.randperm(g2.shape[0])]
        joint1, joint2 = contrast_model(g1, g2), contrast_model(g0, g1)
        margin1, margin2 = contrast_model(g1, shuffle_g2), contrast_model(g0, shuffle_g1)
        mi = - (torch.mean(joint1) - torch.log(torch.mean(torch.exp(margin1)))) + \
             (torch.mean(joint2) - torch.log(torch.mean(torch.exp(margin2))))
        contrast_loss = -(torch.log(torch.mean(F.cosine_similarity(z0, z2))))
        loss = mi+ kl_weight * kl_loss
        #loss = mi
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss, [encoder_model, contrast_model]

def train(args, data):

    # inizialize models
    GAE, GCL, opt_gae, opt_gcl = initialize_model(args)

    # errors = train_GAE(args, data, GAE, optimizer=opt_gae, is_training=True)
    # residal_data, candi_groups, sub_size = filter_subgraph(args, data, errors)
    im_GAE = ImprovedGAE(data.num_features, 256, 128).to("cuda:0")
    optimizer = torch.optim.Adam(im_GAE.parameters(), lr=args.gae_lr)
    im_loss = train_imGAE(args,im_GAE,data,optimizer=optimizer, is_training=True)

    # train GAE to locate subgraph

    # filter
    residal_data, candi_groups, sub_size = filter_subgraph(args, data, im_loss)

    # preprocessing for locate critical edges and nodes of each batch
    batch_list, cycle_edges_list, tree_root_list, path_middle_list, one_degree_list = [], [], [], [], []
    for rdata in [residal_data]:
        cycle_edges, tree_root_nodes, del_edge_index, one_degree_nodes = GraphProcessor().pattern_search(rdata)
        batch_list.append(rdata)
        cycle_edges_list.append(cycle_edges)
        tree_root_list.append(tree_root_nodes)
        path_middle_list.append(del_edge_index)
        one_degree_list.append(one_degree_nodes)

    # train GCL
    with tqdm(total=args.gcl_epochs, desc='(GCL)') as pbar:
        for epoch in range(1, args.gcl_epochs):
            loss, GCL = train_GCL(args, GCL[0], GCL[1],
        [batch_list, cycle_edges_list, tree_root_list, path_middle_list, one_degree_list], opt_gcl)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    return  GCL, batch_list,im_GAE

def test(i, args, GCL, data, im_GAE):

    GCL_encoder, contrast_model = GCL[0], GCL[1]
    GCL_encoder.eval()
    im_GAE.eval()

    # errors = train_GAE(args, data, im_GAE, optimizer=None, is_training=False)
    # residal_data, candi_groups, sub_size = filter_subgraph(args, data, errors)

    im_loss = train_imGAE(args, im_GAE, data,optimizer=None, is_training=False)
    residal_data, candi_groups, sub_size = filter_subgraph(args, data, im_loss)

    batch_list, cycle_edges_list, tree_root_list, path_middle_list, one_degree_list = [], [], [], [], []
    for rdata in [residal_data]:
        cycle_edges, tree_root_nodes, del_edge_index, one_degree_nodes = GraphProcessor().pattern_search(rdata)
        batch_list.append(rdata)
        cycle_edges_list.append(cycle_edges)
        tree_root_list.append(tree_root_nodes)
        path_middle_list.append(del_edge_index)
        one_degree_list.append(one_degree_nodes)
    _, g, _, _, _, _ = GCL_encoder(residal_data.x, residal_data.edge_index, residal_data.batch)
    x, y = [], []
    x.append(g)
    y.append(residal_data.y)
    x = torch.cat(x, dim=0).detach().cpu().numpy()
    y = torch.cat(y, dim=0).detach().cpu().numpy()

    def visualize_tsne(embeddings, labels, save_path):
        """使用t-SNE将子图嵌入降维并可视化"""
        # 检查嵌入维度是否需要降维
        if embeddings.shape[1] > 2:
            tsne = TSNE(
                n_components=2,
                perplexity=min(30, len(embeddings) - 1),  # 避免perplexity大于样本数
                random_state=42,
                init='pca',
                learning_rate='auto'
            )
            embeddings_2d = tsne.fit_transform(embeddings)
        else:
            # 如果已是2维则直接使用
            embeddings_2d = embeddings

        # 绘制散点图
        plt.figure(figsize=(8, 8))
        # 正常子图（0）用蓝色，异常子图（1）用红色
        normal_mask = labels == 0 # 正常子图（标签0）
        anomaly_mask = labels == 1  # 异常子图（标签1）

        # 正常子图：绿色
        plt.scatter(
            embeddings_2d[normal_mask, 0],
            embeddings_2d[normal_mask, 1],
            c='green',  # 正常子图绿色
            label='normal',
            s=100,
            alpha=0.9,
            edgecolors='white',
            linewidths=0.5
        )

        # 异常子图：红色
        plt.scatter(
            embeddings_2d[anomaly_mask, 0],
            embeddings_2d[anomaly_mask, 1],
            c='red',  # 异常子图红色
            label='anomaly',
            s=100,
            alpha=0.9,
            edgecolors='white',
            linewidths=0.5
        )

        # 添加图例
        plt.legend(fontsize=18, title='Subgraph Type')
        # plt.xlabel('Dim1',fontweight='bold')
        # plt.ylabel('Dim2',fontweight='bold')
        plt.title(
            f'(b) T-SNE SimML',
            y=-0.12,  # 关键参数：将标题放在图像下方
            fontsize=22,
            fontweight='bold'  # 标题加粗
        )
        # 保存图像（可选）
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    save_path = f"SimML-tsne_iter_{i}.png"
    # 调用可视化函数（可指定保存路径，如args.tsne_save_path）
    visualize_tsne(x, y,save_path=save_path)


    cls = COPOD(contamination=args.contamination,n_jobs=-1).fit(x)

    y_score, y_pre = cls.decision_scores_, cls.labels_
    test_micro = f1_score(y, y_pre, average='micro')

    try:
        auc = roc_auc_score(y, y_score)
    except:
        auc = 0
    cr = CR_calculator(data, candi_groups, y_pre)

    result = {'f1': test_micro, 'auc': auc, 'cr': cr, 'comp_size': sub_size}
    return result

def mask_features(x, mask_ratio, mask_value):
    """
    随机掩码节点特征的一部分

    参数:
        x: 节点特征矩阵 [num_nodes, num_features]
        mask_ratio: 掩码比例
        mask_value: 掩码后填充的值
    """
    num_nodes, num_features = x.shape
    mask = torch.rand(num_nodes, num_features) < mask_ratio
    x_masked = x.clone()
    x_masked[mask] = mask_value
    return x_masked, mask
