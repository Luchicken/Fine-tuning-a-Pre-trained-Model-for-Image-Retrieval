import argparse

def get_args():
    # 命令行设置
    parser = argparse.ArgumentParser(description='Fine-tuning')
    parser.add_argument('--model', type=str, default='alexnet', help='alexnet/resnet')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=25)  # 8(BJTU) + 17(util_pic)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--latent_layer', action='store_true', help='Enable latent layer')  # 是否启用隐层
    parser.add_argument('--latent_size', type=int, default=48)  # 隐层大小
    parser.add_argument('--K', type=int, default=20, help='20/40/60')  # top K
    parser.add_argument('--plot', action='store_true', help='Plot the retrieval results')  # 是否绘制检索结果
    parser.add_argument('--dist', type=str, default='cos', help='cos/euclidean')
    return parser.parse_args()

