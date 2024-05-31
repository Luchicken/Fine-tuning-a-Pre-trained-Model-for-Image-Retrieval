import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import load_data
import time
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from finetune import load_model
from args import get_args


args = get_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_base_features(model, dataloader):
    """提取特征"""
    base_features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(DEVICE)
            features = model(images)
            features = features.view(features.size(0), -1)  # 展平
            base_features.append(features)
    base_features = torch.cat(base_features)
    return base_features.cpu()


def extract_base_features_latent(model, model_H, dataloader):
    """提取特征"""
    base_fine_features, base_H_features = [], []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(DEVICE)
            fine_features = model(images)
            H_features = model_H(fine_features)
            H_features = torch.where(H_features >= 0.5, 1, 0)
            fine_features = fine_features.view(fine_features.size(0), -1)  # 展平
            H_features = H_features.view(H_features.size(0), -1)  # 展平
            base_fine_features.append(fine_features)
            base_H_features.append(H_features)
    base_fine_features = torch.cat(base_fine_features)
    base_H_features = torch.cat(base_H_features)
    return base_fine_features.cpu(), base_H_features.cpu()


def retrieve(model, query_image, base_features, K=20, dist='cos'):
    """查询 query_image 的前 K 个相似图像索引"""
    with torch.no_grad():
        query_image = query_image.to(DEVICE)
        query_feature = model(query_image.unsqueeze(0))
        query_feature = query_feature.view(query_feature.size(0), -1)
    if dist == 'cos':
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(query_feature, base_features)  # shape: (6445)
        top_k_indices = torch.topk(cos_sim, K).indices
    else:
        # 计算欧氏距离
        euclidean_dist = torch.sqrt(torch.sum((query_feature - base_features) ** 2, dim=1))
        top_k_indices = torch.topk(euclidean_dist, K, largest=False).indices
    return top_k_indices.cpu().numpy()


def retrieve_latent(model, model_H, query_image, base_features, base_H_features, K=20, dist='cos'):
    """查询 query_image 的前 K 个相似图像索引--启用隐层"""
    with torch.no_grad():
        query_image = query_image.to(DEVICE)
        query_fine_feature = model(query_image.unsqueeze(0))
        query_H_feature = model_H(query_fine_feature)
        query_H_feature = torch.where(query_H_feature >= 0.5, 1, 0)
        query_fine_feature = query_fine_feature.view(query_fine_feature.size(0), -1)
        query_H_feature = query_H_feature.view(query_H_feature.size(0), -1)
    # 计算汉明距离
    hamming_distance = torch.sum(torch.abs(base_H_features - query_H_feature), dim=1)
    top_k_indices_Coarse = torch.where(hamming_distance < 20)[0]  # 粗检索
    # top_k_indices_Coarse = torch.topk(hamming_distance, 2 * K, largest=False).indices  # 粗检索
    if dist == 'cos':
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(query_fine_feature, base_features[top_k_indices_Coarse])
        top_k_indices = top_k_indices_Coarse[torch.topk(cos_sim, K).indices]  # 精检索
    else:
        # 计算欧氏距离
        euclidean_dist = torch.sqrt(torch.sum((query_fine_feature - base_features[top_k_indices_Coarse]) ** 2, dim=1))
        top_k_indices = top_k_indices_Coarse[torch.topk(euclidean_dist, K, largest=False).indices]  # 精检索
    # top_k_indices = torch.topk(hamming_distance, K, largest=False).indices
    return top_k_indices.cpu().numpy()


def denormalize(tensor, mean, std):
    """反标准化"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


# 评估检索性能并显示图像
def evaluate_and_display(model, dataset_query, base_features, base_labels, dataset, K=20, plot=False, plot_root=None, base_H_features=None, model_H=None, dist='cos'):
    last_query_label = None
    PK_list = {}
    PK, count_landmark = 0., 0
    plot_folder_K = os.path.join(plot_root, f'{K}')
    os.makedirs(plot_folder_K, exist_ok=True)
    for idx, (query_image, _) in enumerate(dataset_query):
        query_label = dataset_query.imgs[idx][0].split('/')[3]
        if base_H_features is None:
            top_k_indices = retrieve(model, query_image, base_features, K, dist)
        else:
            top_k_indices = retrieve_latent(model, model_H, query_image, base_features, base_H_features, K, dist)
        top_k_labels = base_labels[top_k_indices]
        # 计算前 K 个检索结果中与查询相关的样本占比
        relevant_count = (top_k_labels == query_label).sum()
        # print(f'Query {idx + 1}: {relevant_count / K:.2f}')

        if last_query_label == None:
            last_query_label = query_label
            PK = relevant_count / K
            count_landmark = 1
        elif last_query_label != query_label:
            PK_list[last_query_label] = PK / count_landmark
            last_query_label = query_label
            PK = relevant_count / K
            count_landmark = 1
        else:
            PK += relevant_count / K
            count_landmark += 1

        if plot:
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            num_rows = (K // 10) + 1
            fig, axes = plt.subplots(num_rows, 10, figsize=(20, 2 * num_rows))
            # 显示查询图像
            query_image_denorm = denormalize(query_image.clone(), mean, std).permute(1, 2, 0).numpy()
            axes[0, 0].imshow(query_image_denorm)
            axes[0, 0].set_title(f"Query {idx + 1}: {query_label}")
            axes[0, 0].axis('off')
            # 显示检索结果
            for i, base_idx in enumerate(top_k_indices):
                row = i // 10 + 1
                col = i % 10
                retrieved_image, _ = dataset[base_idx]
                retrieved_image_denorm = denormalize(retrieved_image.clone(), mean, std).permute(1, 2, 0).numpy()
                axes[row, col].imshow(retrieved_image_denorm)
                axes[row, col].set_title(f"Result {i + 1}: {base_labels[base_idx]}")
                axes[row, col].axis('off')
            for i in range(1, 10):
                axes[0, i].axis('off')  # 隐藏多余子图
            plt.savefig(f"{plot_folder_K}/{str(idx).zfill(2)}.png", bbox_inches='tight')
            plt.close()
    PK_list[last_query_label] = PK / count_landmark
    return PK_list


if __name__ == '__main__':
    # 加载模型 (AlexNet / ResNet-50)
    ckpt = './ckpt'
    if args.model == 'alexnet':
        if args.latent_layer:
            save_model = 'model_alexnet_20240529-220813.pkl'
        else:
            save_model = 'model_alexnet_20240530-095321.pkl'
    else:
        save_model = 'model_resnet_20240529-220809.pkl'
    model_name = args.model
    model = load_model(model_name).to(DEVICE)
    print(f"Loading model from '{os.path.join(ckpt, save_model)}'...")
    print(f"Model: {args.model}, Latent Layer: {args.latent_layer}")
    model.load_state_dict(torch.load(os.path.join(ckpt, save_model)))  # 加载最佳模型参数
    if args.model == 'alexnet':
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    else:
        model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    if args.model == 'alexnet' and args.latent_layer:
        model_H = model.classifier[-2:]
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])

    # 加载 base 数据
    dataset, dataloader = load_data(args.data, 'base', args.batchsize)
    # 提取并保存 base 特征和标签
    if not os.path.exists(f'base_features/features_{save_model}'):
        base_labels = np.array(dataset.imgs)[:, 0]
        base_labels = np.vectorize(lambda x: x.split('/')[3])(base_labels)
        if args.model == 'alexnet' and args.latent_layer:
            base_features, base_H_features = extract_base_features_latent(model, model_H, dataloader)
            os.makedirs('base_features', exist_ok=True)
            with open(f'base_features/features_{save_model}', 'wb') as f:
                pickle.dump({
                    'features': base_features,
                    'H_features': base_H_features,
                    'labels': base_labels,
                }, f)
        else:
            base_features = extract_base_features(model, dataloader)
            os.makedirs('base_features', exist_ok=True)
            with open(f'base_features/features_{save_model}', 'wb') as f:
                pickle.dump({
                    'features': base_features,
                    'labels': base_labels,
                }, f)

    # 加载 base 特征和标签
    print(f"Loading base features & labels from './base_features/features_{save_model}'...")
    with open(f'base_features/features_{save_model}', 'rb') as f:
        log = pickle.load(f)
        if args.model == 'alexnet' and args.latent_layer:
            base_H_features = log['H_features'].to(DEVICE)
        base_features = log['features'].to(DEVICE)
        base_labels = log['labels']

    # 加载 query 数据
    print(f"Loading query images...")
    dataset_query, _ = load_data(args.data, 'query', 1)

    # 保存图像的文件夹
    plot_root = os.path.join('./plots', save_model.split('.')[0])
    os.makedirs(plot_root, exist_ok=True)
    # 评估检索性能并显示图像
    print("\nStart image retrieval.")
    since = time.time()
    # PK_list = evaluate_and_display(model, dataset_query, base_features, base_labels, dataset, args.K, args.plot, plot_root)
    if args.model == 'alexnet' and args.latent_layer:
        PK_list = {K: evaluate_and_display(model, dataset_query, base_features, base_labels, dataset, K, args.plot, plot_root, base_H_features, model_H, dist=args.dist) for K in [20, 40, 60]}
    else:
        PK_list = {K: evaluate_and_display(model, dataset_query, base_features, base_labels, dataset, K, args.plot, plot_root, dist=args.dist) for K in [20, 40, 60]}
    time_pass = time.time() - since
    print(f"Time cost: {time_pass:.2f}s")
    print(f"K=20 {PK_list[20]}")
    print(f"K=40 {PK_list[40]}")
    print(f"K=60 {PK_list[60]}")

    # 绘制每个 landmark 的 P@K 图
    landmarks = list(PK_list[20].keys())
    cols = 3
    rows = (len(landmarks) - 1) // cols + 1
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    for i, landmark in enumerate(landmarks):
        row, col = i // cols, i % cols
        ax = axes[row, col]
        Ks = sorted(PK_list.keys())
        P_at_K = [PK_list[K][landmark] for K in Ks]
        ax.scatter(Ks, P_at_K, marker='o', s=140)
        for K, P in zip(Ks, P_at_K):
            ax.text(K, P - 0.03, f'{P:.2f}', fontsize=13)
        ax.set_xlabel('K', fontsize=15)
        ax.set_ylabel('Precision', fontsize=15)
        ax.set_title(f'Precision@K for {landmark}', fontsize=15)
        ax.set_xticks(Ks)
        ax.set_yticks([0.45, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
        # ax.set_yticks([0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
        ax.set_xlim(10, 70)
        ax.set_ylim(0.45, 1.05)
        # ax.set_ylim(0.20, 1.05)
        ax.grid(True, linestyle='--')
    if len(landmarks) < rows * cols:
        for i in range(len(landmarks), rows * cols):
            row = i // cols
            col = i % cols
            fig.delaxes(axes[row, col])
    plt.tight_layout()
    plt.savefig(f"{plot_root}/P@K.png", bbox_inches='tight')
    plt.close()

