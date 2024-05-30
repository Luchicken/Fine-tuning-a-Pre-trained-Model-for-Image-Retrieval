import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import load_data
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


def retrieve(model, query_image, base_features, K):
    """查询 query_image 的前 K 个相似图像索引"""
    with torch.no_grad():
        query_image = query_image.to(DEVICE)
        query_feature = model(query_image.unsqueeze(0))
        query_feature = query_feature.view(query_feature.size(0), -1)
    # 计算余弦相似度
    cos_sim = F.cosine_similarity(query_feature, base_features)  # shape: (6445)
    # 获取前 K 个相似的图像索引
    top_k_indices = torch.topk(cos_sim, K).indices
    return top_k_indices.cpu().numpy()


def denormalize(tensor, mean, std):
    """反标准化"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


# 评估检索性能并显示图像
def evaluate_and_display(model, dataset_query, base_features, base_labels, dataset, K=20, plot=False, plot_root=None):
    last_query_label = None
    PK_list = []
    PK, count_landmark = 0., 0
    plot_folder = os.path.join(plot_root, f'{K}')
    os.makedirs(plot_folder, exist_ok=True)
    for idx, (query_image, _) in enumerate(dataset_query):
        query_label = dataset_query.imgs[idx][0].split('/')[3]
        top_k_indices = retrieve(model, query_image, base_features, K)
        top_k_labels = base_labels[top_k_indices]
        # 计算前 K 个检索结果中与查询相关的样本占比
        relevant_count = (top_k_labels == query_label).sum()
        # print(f'Query {idx + 1}: {relevant_count / K:.2f}')

        if last_query_label == None:
            last_query_label = query_label
            PK = relevant_count / K
            count_landmark = 1
        elif last_query_label != query_label:
            PK_list.append(PK / count_landmark)
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
            plt.savefig(f"{plot_folder}/{str(idx).zfill(2)}.png")
            plt.close()
    PK_list.append(PK / count_landmark)
    return PK_list


if __name__ == '__main__':
    # 加载模型 (AlexNet / ResNet-50)
    ckpt = './ckpt'
    if args.model == 'alexnet':
        save_model = 'model_alexnet_20240530-095321.pkl'
    else:
        save_model = 'model_resnet_20240529-220809.pkl'
    model_name = args.model
    model = load_model(model_name).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(ckpt, save_model)))  # 加载最佳模型参数
    if args.model == 'alexnet':
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    else:
        model = nn.Sequential(*list(model.children())[:-1])
    model.eval()

    # 加载 base 数据
    dataset, dataloader = load_data(args.data, 'base', args.batchsize)
    # 提取并保存 base 特征和标签
    if not os.path.exists(f'base_features/features_{save_model}'):
        base_features = extract_base_features(model, dataloader)
        base_labels = np.array(dataset.imgs)[:, 0]
        base_labels = np.vectorize(lambda x: x.split('/')[3])(base_labels)
        os.makedirs('base_features', exist_ok=True)
        with open(f'base_features/features_{save_model}', 'wb') as f:
            pickle.dump({
                'features': base_features,
                'labels': base_labels,
            }, f)

    # 加载 base 特征和标签
    with open(f'base_features/features_{save_model}', 'rb') as f:
        log = pickle.load(f)
        base_features = log['features'].to(DEVICE)
        base_labels = log['labels']
    # 加载 query 数据
    dataset_query, _ = load_data(args.data, 'query', 1)

    # 保存图像的文件夹
    plot_folder_path = os.path.join('./plots', save_model.split('.')[0])
    os.makedirs(plot_folder_path, exist_ok=True)
    # 评估检索性能并显示图像
    # PK_list = evaluate_and_display(model, dataset_query, base_features, base_labels, dataset, args.K, args.plot)
    PK_list = {K: evaluate_and_display(model, dataset_query, base_features, base_labels, dataset, K, args.plot, plot_folder_path) for K in [20, 40, 60]}
    print(f"K=20 {PK_list[20]}")
    print(f"K=40 {PK_list[40]}")
    print(f"K=60 {PK_list[60]}")

