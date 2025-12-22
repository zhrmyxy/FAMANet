# We appreciate the reviewers' helpful comments.

# Response to Reviewer #3837:
### 1.PAAM module analysis
We confirm that Wa and Wp are adaptively learned via end-to-end backpropagation. A lightweight parameter generation network is designed, consisting of a global average pooling layer and a two-layer MLP (Linear → ReLU → Linear → Softmax). For shared transformations, a shared linear layer is employed for dimensionality reduction, while independent linear layers are used to generate individual transformations. The shared weights serve a regularizing effect by reducing the number of parameters, thereby preventing overfitting in few-shot tasks and enforcing alignment of magnitude and phase features within a unified latent space.
For the weight ablation study, refer to the following table:
<div align="center">
  <img width="676" height="139" alt="image" src="https://github.com/user-attachments/assets/c8a1b3d4-d058-4146-adf5-26b395a69606" />
</div>
Experiments demonstrate that the dynamic gating achieves optimal performance.

### 2.Cost Analysis of AMAM Calculation
The AMAM module is parameter-free, and its operations are primarily based on matrix multiplication, resulting in a computational complexity of O(N²), where N is determined by the height (H) and width (W) of the features.
<div align="center">
  <img width="621" height="90" alt="image" src="https://github.com/user-attachments/assets/6d0acfc7-adcb-4f26-ae37-f4d5e9acf01f" />
</div>
As demonstrated by the experimental results, AMAM has a low computational cost and exerts minimal impact on model inference speed.

#### Noisy support mask experiment: 
To evaluate the model's robustness to imperfect annotations, we simulate noise by applying morphological dilation to the support masks. Specifically, we expand the mask boundaries using kernels of varying sizes. This process inevitably introduces background regions into the support features, creating a challenging scenario.
<div align="center">
  <img width="674" height="142" alt="image" src="https://github.com/user-attachments/assets/cf3a2bc5-8fe1-477f-a18f-dace70a3dc2a" />
</div>

As shown in this table, even under a dilation rate of 20 and with severely noisy support masks, our model still maintains certain performance, demonstrating the robustness of our model in experiments involving noisy support masks.

### 3.CTSGM and Generalization of the model
We followed the standard protocol by using the template "a photo of a {class}" for text embeddings. Additionally, we explicitly verified the generalization effectiveness of our method through cross-domain experiments transferred from COCO-20i to PASCAL-5i.

<div align="center">
  <img width="615" height="190" alt="image" src="https://github.com/user-attachments/assets/718d1ec8-e904-4da9-9289-de64d345c1fd" />
</div>

# Response to Reviewer #4E42:
### 1.Experimental fairness and resolution effects
We first clarify that the resolution commonly used in state-of-the-art (SOTA) methods, 473×473, contains 52% more pixel information than the resolution of 384×384 adopted in this work, which typically benefits segmentation performance. To alleviate concerns, we re-evaluated under the SOTA setting and achieved 70.2% mIoU (473×473), still outperforming existing methods under the same configuration, which strongly demonstrates that the performance gain stems from improvements in the model architecture itself. Moreover, the original 384×384 setting maintains excellent performance (70.5%) while significantly improving inference speed compared to higher resolutions (18.93 FPS vs. 12.47 FPS).
<div align="center">
  <img width="716" height="115" alt="image" src="https://github.com/user-attachments/assets/e99a6ddb-8ce2-444a-a907-f17d63005d38" />
</div>

### 2.Implementation Details
As described in Section 3.0 (EXPERIMENTS AND RESULTS) of the original text. To ensure result reproducibility, our experimental setup strictly follows the conditions of HSNet and DCAMA, with specific adjustments: we employ the Adam optimizer (initial learning rate set to 1e-4) and train for 100 epochs until convergence. Additionally, consistent with HSNet's configuration, no extra data augmentation strategies are introduced during this training phase. All experiments are conducted on four NVIDIA RTX 4090 GPUs.

- **Source Code**: The implementation of train can be found in [`train.sh`](./scripts/train.sh). 
- **Source Code**: The implementation of test can be found in [`test.sh`](./scripts/test.sh). 
- **Source Code**: The implementation of config can be found in [`config.py`](./common/config.py). 


<br />
<br />
<br />
<br />
<br />


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

###   Visualization
<div align="center">
  <img width="1464" height="786" alt="image" src="https://github.com/user-attachments/assets/4d8bdb02-1ba7-485e-8407-fdffa1b5d68a" />
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

