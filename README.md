Install packages and activate conda environment via:
```
conda env create -f environment.yml -n isic2024
conda activate isic2024
```
then run
```
python -m src.training --model HierarchicalResNet18_smoteenn \
    --hier-freeze-backbone-epochs 20 --hier-partial-unfreeze 1 \
    --hier-pos-weight 100 --hier-trunk-dim-2 128 \
    --backbone-lr-mult 0.1 --use-ema 1 \
    --scheduler cosine_warm_restarts --restart-period 25 \
    --epochs 30 --use-smoteenn 1 --batch-size 128 \
    --backbone resnet18 \
    --libauc-mode two_stage --libauc-stage2-epoch 10 --libauc-num-pos 64 \
    --phase-suffix runA_2stage10 --hier-type-loss-weight 0.5
```
