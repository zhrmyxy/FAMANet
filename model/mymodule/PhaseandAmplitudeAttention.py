import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseAmplitudeAttention(nn.Module):
    """Dual Attention using both Phase and Amplitude information"""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.channel = channel

        # 共享的特征变换层
        self.fc_shared = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )

        # 相位和振幅各自独立的注意力分支
        self.fc_phase = nn.Linear(channel // reduction, channel, bias=False)
        self.fc_amplitude = nn.Linear(channel // reduction, channel, bias=False)

        # 特征融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(channel * 2, channel),
            nn.ReLU(),
            nn.Linear(channel, 2),
            nn.Softmax(dim=-1)
        )

        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 添加用于存储权重的占位符
        self.amp_weights = None
        self.phase_weights = None
        self.final_weights = None
        self.gate_weights = None

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # 1. 频域变换 - 添加shift和归一化处理
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))

        # 分离幅度和相位
        amplitude = torch.abs(x_fft) + 1e-8  # 防止除零
        phase = torch.angle(x_fft)

        # 2. 全局特征提取   通过全局平均池化压缩空间维度，获得通道级统计特征。
        amp_pooled = self.avg_pool(amplitude).view(batch_size, -1)
        phase_pooled = self.avg_pool(phase).view(batch_size, -1)

        # 3. 共享特征变换
        shared_amp = self.fc_shared(amp_pooled)
        shared_phase = self.fc_shared(phase_pooled)

        # 4. 独立注意力计算
        amp_weights = self.sigmoid(self.fc_amplitude(shared_amp))
        phase_weights = self.sigmoid(self.fc_phase(shared_phase))

        # 5. 特征融合决策
        combined = torch.cat([amp_pooled, phase_pooled], dim=1)
        gate_weights = self.fusion_gate(combined)  # [B, 2]

        # 6. 动态加权融合
        final_weights = (gate_weights[:, 0:1] * amp_weights +
                         gate_weights[:, 1:2] * phase_weights)


        # 7. 特征增强
        enhanced_x = x * final_weights.view(batch_size, self.channel, 1, 1)

        return enhanced_x

    # # 存储权重用于可视化
    # self.amp_weights = amp_weights.detach().clone()
    # self.phase_weights = phase_weights.detach().clone()
    # self.final_weights = final_weights.detach().clone()
    # self.gate_weights = gate_weights.detach().clone()