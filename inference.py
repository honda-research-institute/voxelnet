#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import os
import time
import tensorflow as tf

from model import RPN3D
from config import cfg
from utils import *
from utils.kitti_loader_jhuang import iterate_data


if __name__ == '__main__':
    output_path = './predictions'
    data_dir = os.path.join('./jhuang/nusc')
    save_model_dir = os.path.join('./save_model', 'default')
    
    # create output folder
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'data'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'vis'), exist_ok=True)

    with tf.Graph().as_default():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
                            visible_device_list=cfg.GPU_AVAILABLE,
                            allow_growth=True)
    
        config = tf.ConfigProto(
            gpu_options=gpu_options,
            device_count={
                "GPU": cfg.GPU_USE_COUNT,
            },
            allow_soft_placement=True,
        )

        with tf.Session(config=config) as sess:
            model = RPN3D(
                cls=cfg.DETECT_OBJ,
                single_batch_size=1,
                avail_gpus=cfg.GPU_AVAILABLE.split(',')
            )
            if tf.train.get_checkpoint_state(save_model_dir):
                print("Reading model parameters from %s" % save_model_dir)
                model.saver.restore(
                    sess, tf.train.latest_checkpoint(save_model_dir))

            for batch in iterate_data(data_dir, shuffle=False, aug=False, \
                                      batch_size=cfg.GPU_USE_COUNT, multi_gpu_sum=cfg.GPU_USE_COUNT):

                tic = time.perf_counter()
                tags, results, _, bird_views, _ = \
                    model.predict_step(sess, batch, summary=False, vis=True, bev_only=True)

                toc = time.perf_counter()
                t_lapse = toc - tic
                print('Inference time = {:d} ms'.format(round(t_lapse*1000)))
                
                # ret: A, B
                # A: (N) tag
                # B: (N, N') (class, x, y, z, h, w, l, rz, score)
                for tag, result in zip(tags, results):
                    of_path = os.path.join(output_path, 'data', tag + '.txt')
                    with open(of_path, 'w+') as f:
                        labels = box3d_to_label([result[:, 1:8]], [result[:, 0]], [result[:, -1]], coordinate='lidar')[0]
                        for line in labels:
                            f.write(line)
                        print('write out {} objects to {}'.format(len(labels), tag))
                # dump visualizations
                for tag, bird_view in zip(tags, bird_views):
                    bird_view_path = os.path.join( output_path, 'vis', tag + '_bv.jpg'  )
                    cv2.imwrite( bird_view_path, bird_view )


