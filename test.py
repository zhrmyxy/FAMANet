r""" Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation """
import torch.nn as nn
import torch
from thop import profile
try:
    from model.mymodule.CrossAttention6 import CrossMultiHeadedAttention
except ImportError:
    pass # 如果这行报错，请检查你的文件路径
# === 定义自定义 FLOPs 计算规则 ===
def count_cross_attention_flops(m, x, y):
    # x 是输入 tuple: (Asq, Ass, Aqq, value)
    # 取 Asq (形状 [B, HW, HW]) 来计算 N
    Asq = x[0]
    B, N, _ = Asq.shape
    # 估算公式：Enhance操作 + 2次矩阵乘法 + Softmax ≈ 6 * B * N^2
    # N = 60*60 = 3600 (在473输入下)
    total_ops = 6 * B * N * N
    m.total_ops += torch.DoubleTensor([total_ops])

from model.DCAMA6 import DCAMA6
from model.DCAMA8 import DCAMA8
# from model.DCAMA2 import DCAMA2
# from model.DCAMA4 import DCAMA4
# from model.DCAMA6 import DCAMA6
# from model.DCAMA9 import DCAMA9
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset
import clip
import time


def benchmark_model(model, args):
    print(f"\n{'=' * 20} Model Efficiency Benchmark {'=' * 20}")

    # 1. 设置测试分辨率 (为了回复审稿人，建议用 473)
    H, W = 384, 384
    device = torch.device("cuda:0")

    # 2. 准备 Dummy Input (模拟一个 Batch 的输入)
    # 假设 Batch Size = 1 (推理通常是 1)
    dummy_query = torch.randn(1, 3, H, W).to(device)
    dummy_support = torch.randn(1, args.nshot, 3, H, W).to(device)
    dummy_mask = torch.randn(1, args.nshot, H, W).to(device)
    dummy_class = torch.tensor([1]).to(device)  # 假设 Class ID 为 1

    # A. 计算 Params 和 FLOPs
    # ==========================
    model_core = model.module if isinstance(model, nn.DataParallel) else model

    # --- 关键修正：模拟训练时的输入处理 ---
    # 必须和 model.forward 接收的参数一模一样
    # 如果 args.nshot 为 1，训练代码通常会去掉 shot 维度
    if args.nshot == 1:
        s_img_in = dummy_support.squeeze(1)  # [1, 1, 3, H, W] -> [1, 3, H, W]
        s_mask_in = dummy_mask.squeeze(1)  # [1, 1, H, W] -> [1, H, W]
    else:
        # 如果 nshot > 1，看你的 forward 怎么处理，保持一致即可
        s_img_in = dummy_support.squeeze(1)
        s_mask_in = dummy_mask.squeeze(1)

    inputs = (dummy_query, s_img_in, s_mask_in, dummy_class)

    # 这里的 custom_ops 必须确保 key 是类对象
    # 检查一下 globals() 里有没有 CrossMultiHeadedAttention
    ops_rules = {}
    if 'CrossMultiHeadedAttention' in globals():
        ops_rules[CrossMultiHeadedAttention] = count_cross_attention_flops
    else:
        print("Warning: CrossMultiHeadedAttention class not found in globals!")

    # 直接调用，不要 try-except，报错就让它报出来，我们好看日志
    print("Running thop.profile... (If it crashes here, check input shapes)")
    flops, params = profile(
        model_core,
        inputs=inputs,
        custom_ops=ops_rules,
        verbose=False
    )

    print(f"Input Size:       {H}x{W}")
    print(f"Total Params:     {params / 1e6:.2f} M")
    print(f"Total FLOPs:      {flops / 1e9:.2f} G")

    # ==========================
    # B. 计算 FPS (使用 predict_mask_nshot)
    # ==========================
    print("Measuring FPS...")

    # 构造 batch 字典，模拟 dataloader 输出
    batch = {
        'query_img': dummy_query,
        'support_imgs': dummy_support,
        'support_masks': dummy_mask,
        'class_id': dummy_class,
        'query_mask': torch.randn(1, H, W).to(device)  # 仅防报错
    }

    # 预热 GPU
    model.eval()
    with torch.no_grad():
        for _ in range(50):
            _ = model.module.predict_mask_nshot(batch, nshot=args.nshot)

    # 正式计时
    iterations = 200
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model.module.predict_mask_nshot(batch, nshot=args.nshot)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    fps = 1.0 / avg_time

    print(f"Latency:          {avg_time * 1000:.2f} ms")
    print(f"FPS:              {fps:.2f}")
    print(f"{'=' * 60}\n")


def test(model, dataloader, nshot):
    r""" Test """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. forward pass
        batch = utils.to_cuda(batch)
        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)

        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
                                                  iou_b=area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    args = parse_opts()

    Logger.initialize(args, training=False)

    # Model initialization
    device = torch.device("cuda", args.local_rank)
    clip_model, _ = clip.load('RN50', device=device, jit=False)
    model = DCAMA6(args.backbone, args.feature_extractor_path, args.use_original_imgsize, clip_model)
    model.eval()

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    params = model.state_dict()
    state_dict = torch.load(args.load)

    for k1, k2 in zip(list(state_dict.keys()), params.keys()):
        state_dict[k2] = state_dict.pop(k1)

    model.load_state_dict(state_dict)

    # =======================================================
    # 【在这里插入调用代码】
    # 确保只在主进程打印，避免多卡重复输出
    if args.local_rank == 0:
        benchmark_model(model, args)
    # =======================================================

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize, args.vispath)

    # Dataset initialization
    FSSDataset.initialize(img_size=384, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    # dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot, cam_train_path=args.traincampath, cam_val_path=args.valcampath)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    # Test
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
