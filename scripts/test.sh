#CUDA_LAUNCH_BLOCKING=1
#python ./test.py --datapath "../datasets" \
#                 --benchmark pascal \
#                 --fold 0 \
#                 --bsz 1 \
#                 --nworker 8 \
#                 --backbone swin \
#                 --feature_extractor_path "../backbones/swin_base_patch4_window12_384_22kto1k.pth" \
#                 --logpath "./logs" \
#                 --load "./logs/train/fold_0_0624_091317/best_model.pt" \
#                 --nshot 1 \
#                 --vispath "./vis_5" \
#                 --visualize

CUDA_LAUNCH_BLOCKING=1
python ./test.py --datapath "../datasets" \
                 --benchmark pascal \
                 --fold 0 \
                 --bsz 1 \
                 --nworker 8 \
                 --backbone resnet50 \
                 --feature_extractor_path "../backbones/resnet50_a1h-35c100f8.pth" \
                 --logpath "./logs" \
                 --load "./pascal_resnet50_fold0_1.pt" \
                 --nshot 1 \
                 --vispath "./vis_5" \
                 --visualize
