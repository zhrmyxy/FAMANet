We thank the reviewer for the comments.

### Response to Reviewer#1
# Response to PAAM module
我们确认Wa和Wp是通过端到端的反向传播自适应学习得到的，设计了一个轻量级的参数生成网络，该子网络包含全局平均池化层和两层MLP（Linear -> ReLU -> Linear -> Softmax）。
对于共享变换，我们使用线性层进行降维的时候使用了共享的线性层，而生成各自变换时使用独立的线性层，共享权重通过减少参数量起到了正则化作用，防止了小样本任务中的过拟合，强制幅度和相位特征在统一的潜空间中对齐。
对于权重消融研究如下表所示：
<div align="center">
  <img width="676" height="139" alt="image" src="https://github.com/user-attachments/assets/c8a1b3d4-d058-4146-adf5-26b395a69606" />
</div>

### Cost Analysis of AMAM Calculation
AMAM模块是无参数的，其运算主要是由矩阵乘法得到，计算复杂度是O（N2）
<div align="center">
  <img width="621" height="90" alt="image" src="https://github.com/user-attachments/assets/6d0acfc7-adcb-4f26-ae37-f4d5e9acf01f" />
</div>
由实验结果可见，AMAM的计算成本极低，对于模型推理速度几乎不影响。

含噪声支持掩码实验：为了评估模型对不完美标注的鲁棒性，我们通过对 支持 掩码应用形态学膨胀来模拟噪声。具体来说，我们使用不同大小的核来扩展掩码边界。这一过程不可避免地将背景杂质引入到支持特征中，创造了一个具有挑战性的场景。
<div align="center">
  <img width="674" height="142" alt="image" src="https://github.com/user-attachments/assets/cf3a2bc5-8fe1-477f-a18f-dace70a3dc2a" />
</div>

由此表可见在膨胀率20，支持掩码具有严重噪声情况下，我们的模型仍然保持一定的性能，证明了我们模型面对含噪声支持掩码的实验鲁棒性

### CTSGM and Generalization of the model
We followed the standard protocol by using the template "a photo of a {class}" for text embeddings. Additionally, we explicitly verified the generalization effectiveness of our method through cross-domain experiments transferred from COCO-20i to PASCAL-5i.
本文实验采用的是标准模板：a photo of a {class},进一步本文在跨数据集上实验COCO-20i to PAscal-5i验证其泛化有效性.

<div align="center">
  <img width="615" height="190" alt="image" src="https://github.com/user-attachments/assets/718d1ec8-e904-4da9-9289-de64d345c1fd" />
</div>

### Response to Reviewer#2
# Experimental fairness
To address fairness concerns arising from differing resolution settings, we conducted balanced comparative experiments at the two commonly used resolutions in this field: 473×473 and 448×448. The experimental results demonstrate that FAMANet exhibits strong robustness to input size variations, with performance fluctuations across different resolutions remaining within 0.3%. More importantly, even after strictly aligning the resolution settings, our method maintains superior performance. This fully confirms that the performance improvement is primarily attributable to enhancements in network architecture rather than differences in experimental configuration. All subsequent ablation studies were conducted on the Pascal-5i dataset.
<div align="center">
  <img width="672" height="115" alt="image" src="https://github.com/user-attachments/assets/c03113d5-4521-4f0c-91ea-e38be0759450" />
</div>

### Implementation Details
To ensure result reproducibility, our experimental setup strictly follows the conditions of HSNet and DCAMA, with specific adjustments: we employ the Adam optimizer (initial learning rate set to 1e-4) and train for 100 epochs until convergence. Additionally, consistent with HSNet's configuration, no extra data augmentation strategies are introduced during this training phase. All experiments are conducted on four NVIDIA RTX 4090 GPUs.

-**Source Code**: The implementation of train can be found in [`train.sh`](./scripts/train.sh). 
- **Source Code**: The implementation of test can be found in [`test.sh`](./scripts/test.sh). 
- **Source Code**: The implementation of config can be found in [`config.py`](./common/config.py). 


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
  <img width="1489" height="728" alt="image" src="https://github.com/user-attachments/assets/0b97624d-6686-45bc-ae6b-e5a15daa0c33" />
</div>

---

## 🧩 Key Modules

### 1. Phase and Amplitude Attention Module (PAAM)

PAAM is designed to enhance feature representation by utilizing frequency domain information.

- **Source Code**: The implementation of PAAM can be found in [`PhaseandAmplitudeAttention.py`](./model/mymodule/PhaseandAmplitudeAttention.py). 

<div align="center">
  <img width="1525" height="472" alt="image" src="https://github.com/user-attachments/assets/b457744b-62f2-47bc-aea4-75f6335c04b2" />
</div>

#### Visualization of PAAM Effects
Visual comparison showing the effectiveness of the frequency-enhanced attention:

<div align="center">
  <img width="1000" height="705" alt="image" src="https://github.com/user-attachments/assets/c2b25ca6-6ef1-4290-86a7-5ecfeb1bd913" />
</div>

### 2. Affinity Map Aggregation Module (AMAM)

AMAM utilizes cross-attention mechanisms to aggregate mask weights based on affinity maps.

- **Source Code**: The implementation of AMAM can be found in [`CrossAttention.py`](./model/mymodule/CrossAttention.py). 

<div align="center">
  <img width="973" height="298" alt="image" src="https://github.com/user-attachments/assets/8e7f6930-a4b8-4616-8a27-2e431c096621" />
</div>

---
###   DataSet
<div align="center">
  <img width="729" height="753" alt="image" src="https://github.com/user-attachments/assets/55194873-f3ce-4b0e-83cf-45745eeb3098" />

</div>
<div align="center">
  <img width="729" height="753" alt="image" src="https://github.com/user-attachments/assets/aa96ecb9-886a-4430-860d-d4ce653b1ed5" />
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

