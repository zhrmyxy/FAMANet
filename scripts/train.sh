#python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_port=16005 \
# ./train.py --datapath "../datasets" \
#           --benchmark coco \
#           --fold 0 \
#           --bsz 12 \
#           --nworker 8 \
#           --backbone swin \
#           --feature_extractor_path "../backbones/swin_base_patch4_window12_384_22kto1k.pth" \
#           --logpath "./logs" \
#           --lr 1e-3 \
#           --nepoch 500
#torchrun --nnodes=1 --nproc_per_node=1 --master_port=16005 \
#./train.py --datapath "../datasets" \
#           --benchmark pascal \
#           --fold 0 \
#           --bsz 6 \
#           --nworker 8 \
#           --backbone resnet50 \
#           --feature_extractor_path "../backbones/resnet50_a1h-35c100f8.pth" \
#           --logpath "./logs" \
#           --lr 1e-3 \
#           --nepoch 100 \
##           --nshot 5 \
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nnodes=1 --nproc_per_node=4 --master_port=16005 \
./train.py --datapath "../datasets" \
           --benchmark pascal \
           --fold 2 \
           --bsz 4 \
           --nworker 8 \
           --backbone resnet50 \
           --feature_extractor_path "../backbones/resnet50_a1h-35c100f8.pth" \
           --logpath "./logs" \
           --lr 1e-4 \
           --nepoch 101 \
#           --nshot 5 \