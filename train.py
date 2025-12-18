r""" training (validation) code """
import torch.optim as optim
import torch.nn as nn
import torch

from model.DCAMA import DCAMA
from model.DCAMA1 import DCAMA1
from model.DCAMA2 import DCAMA2
from model.DCAMA3 import DCAMA3
from model.DCAMA4 import DCAMA4
from model.DCAMA5 import DCAMA5
from model.DCAMA6 import DCAMA6
from model.DCAMA8 import DCAMA8
from model.DCAMA9 import DCAMA9
from model.DCAMA10 import DCAMA10
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset
import clip
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")


def train(epoch, model, dataloader, optimizer, training):
    r""" Train """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. forward pass
        batch = utils.to_cuda(batch)
        # logit_mask, fg_mask = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1), class_id=batch['class_id'], support_cam=batch['support_cams'].squeeze(1))
        logit_mask = model(batch['query_img'], batch['support_imgs'].squeeze(1),
                                    batch['support_masks'].squeeze(1), class_id=batch['class_id'])
        # logit_mask = model(batch['query_img'], batch['support_imgs'].squeeze(1),
        #                             batch['support_masks'].squeeze(1))
        pred_mask = logit_mask.argmax(dim=1)

        # 2. Compute loss & update model parameters
        loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        # loss_fg = model.module.compute_objective2(fg_mask, batch['query_mask'])
        # loss = loss + loss_fg * 1
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


def evaluate_multiple_episodes(model, dataloader, optimizer, epoch, num_episodes=5):
    """多Episode验证取平均"""
    total_miou, total_fb_iou, total_loss = 0.0, 0.0, 0.0
    for _ in range(num_episodes):
        # 每次验证使用不同的随机支撑集
        avg_loss, miou, fb_iou = train(epoch, model, dataloader, optimizer, training=False)
        total_miou += miou
        total_fb_iou += fb_iou
        total_loss += avg_loss
    return total_loss/num_episodes, total_miou/num_episodes, total_fb_iou/num_episodes


if __name__ == '__main__':

    # Arguments parsing
    args = parse_opts()

    # ddp backend initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl')

    # Model initialization
    device = torch.device("cuda", args.local_rank)
    clip_model, _ = clip.load('RN50', device=device, jit=False)
    clip_model = clip_model.to(device)
    model = DCAMA6(args.backbone, args.feature_extractor_path, False, clip_model)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True)

    # Helper classes (for training) initialization
    # optimizer = optim.SGD([{"params": model.parameters(), "lr": args.lr,
    #                         "momentum": 0.9, "weight_decay": args.lr / 10, "nesterov": True}])
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,  # 保留原始学习率
    )
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=args.lr,
    #     betas=(0.9, 0.999),
    #     weight_decay=args.lr / 10
    # )

    Evaluator.initialize()
    if args.local_rank == 0:
        Logger.initialize(args, training=True)
        Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    # Dataset initialization
    FSSDataset.initialize(img_size=473, datapath=args.datapath, use_original_imgsize=False)
    # dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn', cam_train_path=args.traincampath, cam_val_path=args.valcampath)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    if args.local_rank == 0:
        # dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', cam_train_path=args.traincampath, cam_val_path=args.valcampath)
        dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

    # Train
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.nepoch):
        dataloader_trn.sampler.set_epoch(epoch)
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)
        # # ===== Warmup阶段 =====
        # if epoch < args.warmup_epochs:  # 默认warmup_epochs=5
        #     warmup_factor = (epoch + 1) / args.warmup_epochs
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = args.lr * warmup_factor
        #
        # # ===== 余弦退火阶段 =====
        # else:
        #     scheduler.step()  # 余弦退火更新

        # current_lr = optimizer.param_groups[0]['lr']
        # evaluation
        if args.local_rank == 0:
            # # 记录当前学习率（用于监控）
            # Logger.info(f'Epoch {epoch}: Learning Rate = {current_lr:.6f}')
            # Logger.tbd_writer.add_scalar('learning_rate', current_lr, epoch)
            with torch.no_grad():
                val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)

            # Save the best model
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                Logger.save_model_miou(model, epoch, val_miou)

            Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            Logger.tbd_writer.flush()

    if args.local_rank == 0:
        Logger.tbd_writer.close()
        Logger.info('==================== Finished Training ====================')
