### Experimental fairness
感谢审稿人的意见，本文严格按照了HSNet以及DCAMA的实验条件下进行，与之不同的是，我们改变了优化器SGD，修改成了Adam，学习率设置为1e-4，训练100轮次并遵循HSNet在没有数据增强的情况下训练我们的模型，直到收敛，以便与之前的最佳性能方法进行公平的比较。训练和推理是在四个NVIDIA RTX 4090上。我们与DCAMA采用一致的分辨率384*384，为了与Swin-B适配。对比实验中部分采用了473*473分辨率，我们进行等量实验使用分辨率为473*473，其他实验条件保持不变。训练过程中473*473分辨率显存占用明显增大，并且训练时间过长。其最终性能与384*384相比：？


### Response to PAAM module
我们确认Wa和Wp是通过端到端的反向传播自适应学习得到的，设计了一个轻量级的参数生成网络，该子网络包含全局平均池化层和两层MLP（Linear -> ReLU -> Linear -> Softmax）。

对于共享变换，我们使用线性层进行降维的时候使用了共享的线性层，而生成各自变换时使用独立的线性层，共享权重通过减少参数量起到了正则化作用，防止了小样本任务中的过拟合，强制幅度和相位特征在统一的潜空间中对齐。

### Cost Analysis of AMAM Calculation
AMAM模块是无参数的，其运算主要是由矩阵乘法得到，计算复杂度是O（N2）
<div align="center">
  <img width="729" height="107" alt="image" src="https://github.com/user-attachments/assets/68bc429a-8520-4baf-b6c8-b9604102277a" />
</div>

含噪声支持掩码实验：为了评估模型对不完美标注的鲁棒性，我们通过对 支持 掩码应用形态学膨胀来模拟噪声。具体来说，我们使用不同大小的核来扩展掩码边界。这一过程不可避免地将背景杂质引入到支持特征中，创造了一个具有挑战性的场景。
<div align="center">
  <img width="687" height="151" alt="image" src="https://github.com/user-attachments/assets/d1c860f0-1596-460c-95ea-789cb1352d34" />
</div>

由此表可以看到就算在膨胀率20支持掩码具有严重噪声情况下，我们的模型仍然保持一定的性能，证明了我们模型面对含噪声支持掩码的实验鲁棒性

### CTSGM and Generalization of the model
We followed the standard protocol by using the template "a photo of a {class}" for text embeddings. Additionally, we explicitly verified the generalization effectiveness of our method through cross-domain experiments transferred from COCO-20i to PASCAL-5i.
本文实验采用的是标准模板：a photo of a {class},进一步本文在跨数据集上实验COCO-20i to PAscal-5i验证其泛化有效性.


<div align="center">
  <img width="610" height="197" alt="image" src="https://github.com/user-attachments/assets/e1562c8b-da58-4333-b60a-5456b2d85937" />
</div>





# Frequency-enhanced Affinity Map Weighted Mask Aggregation for Few-Shot Semantic Segmentation

<div align="center">

<!-- You can add badges here if you have them, e.g., PyTorch version, License -->
<!-- ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) -->

</div>

## 📖 Introduction

This repository contains the implementation of **Frequency-enhanced Affinity Map Weighted Mask Aggregation (FAMANet)** for Few-Shot Semantic Segmentation.

## 🏗️ Network Architecture

The overall architecture of our proposed method is shown below:

<div align="center">
  <img width="1485" height="718" alt="Network Architecture" src="https://github.com/user-attachments/assets/0d70bd9b-b4d1-45c9-b0aa-41f1d0f80a1e" />
</div>

---

## 🧩 Key Modules

### 1. Phase and Amplitude Attention Module (PAAM)

PAAM is designed to enhance feature representation by utilizing frequency domain information.

- **Source Code**: The implementation of PAAM can be found in [`PhaseandAmplitudeAttention.py`](./PhaseandAmplitudeAttention.py). 

<div align="center">
  <img src="https://github.com/user-attachments/assets/0c070ff6-e029-42f0-a4ae-51cf7d82a6ef" width="700" alt="PAAM Structure">
</div>

#### Visualization of PAAM Effects
Visual comparison showing the effectiveness of the frequency-enhanced attention:

<div align="center">
  <img src="https://github.com/user-attachments/assets/292ded22-2696-48b3-a193-8cd544828303" width="600" alt="PAAM Visualization">
</div>

### 2. Affinity Map Aggregation Module (AMAM)

AMAM utilizes cross-attention mechanisms to aggregate mask weights based on affinity maps.

- **Source Code**: The implementation of AMAM can be found in [`CrossAttention.py`](./CrossAttention.py). 

<div align="center">
  <img src="https://github.com/user-attachments/assets/9a5b1505-7496-4217-952a-501e9bb5b236" width="700" alt="AMAM Structure">
</div>

---
###   DataSet
<div align="center">
  <img width="720" height="750" alt="image" src="https://github.com/user-attachments/assets/f9934672-845b-4ed1-b76d-d47d5afe33c5" />
</div>
<div align="center">
  <img width="722" height="752" alt="image" src="https://github.com/user-attachments/assets/41b72d33-ee1b-4ebb-9a3f-76b75caf60e0" />
</div>



## 🚀 Getting Started

### Training
To train the model, please verify the configurations in the script and run:

```bash
bash train.sh

### Testing
To test the model, please verify the configurations in the script and run:

```bash
bash test.sh

