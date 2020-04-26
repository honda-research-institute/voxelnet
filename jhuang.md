## Train
python train.py --alpha 1 --beta 10

## Test (kitti-val set)
python test.py --vis True

## Inference (random input)
python inference.py

## Important settings (config.py)
`config.py`
- __C.ANCHOR_Z
- __C.RPN_SCORE_THRESH (only output result scoring > this)

`preprocess.py`
- lidar_coord

`kitti_loader_jhuang.py`
- nusc_mode