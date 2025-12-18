import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_cross_similarity(Asq, A_sq_prime, save_path=None):
    """
    可视化原始跨相似度Asq和增强后的A_sq_prime
    Args:
        Asq: 原始跨相似度张量 [B, H_s*W_s, H_q*W_q]
        A_sq_prime: 增强后的跨相似度张量 [B, H_s*W_s, H_q*W_q]
        save_path: 保存路径（可选）
    """
    # 提取第一个样本和第一个支持位置
    Asq_sample = Asq[0, 0].detach().cpu().numpy()
    A_sq_prime_sample = A_sq_prime[0, 0].detach().cpu().numpy()

    # 创建1行2列的子图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 可视化Asq
    im1 = axes[0].imshow(Asq_sample, cmap='jet')
    axes[0].set_title('Original Cross-Similarity (Asq)', fontsize=14)
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 可视化增强后的A_sq_prime
    im2 = axes[1].imshow(A_sq_prime_sample, cmap='jet')
    axes[1].set_title('Enhanced Cross-Similarity (A_sq_prime)', fontsize=14)
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 调整布局
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    plt.close(fig)