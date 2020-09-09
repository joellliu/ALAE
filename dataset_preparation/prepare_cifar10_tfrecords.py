# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Create a tfrecords for cifar10. """

from defaults import get_cfg_defaults
import sys
import logging
from net import *
import numpy as np
import argparse
import os
import tensorflow as tf
import random
import dlutils


def prepare_cifar10(cfg, logger, cifar10_images, cifar10_labels, train):
    im_size = 32

    if train:
        cifar10_images = cifar10_images[:50000]
        cifar10_labels = cifar10_labels[:50000]
    else:
        cifar10_images = cifar10_images[50000:]
        cifar10_labels = cifar10_labels[50000:]

    # cifar10_images = F.pad(torch.tensor(cifar10_images).view(cifar10_images.shape[0], 1, 28, 28), (2, 2, 2, 2)).detach().cpu().numpy()
    cifar10_images = torch.tensor(cifar10_images).view(cifar10_images.shape[0], 3, 32, 32).detach().cpu().numpy()

    if train:
        path = cfg.DATASET.PATH
    else:
        path = cfg.DATASET.PATH_TEST

    directory = os.path.dirname(path)

    os.makedirs(directory, exist_ok=True)

    folds = cfg.DATASET.PART_COUNT

    if not train:
        folds = 1

    cifar10_folds = [[] for _ in range(folds)]

    count = len(cifar10_images)

    count_per_fold = count // folds
    for i in range(folds):
        cifar10_folds[i] += (cifar10_images[i * count_per_fold: (i + 1) * count_per_fold],
                           cifar10_labels[i * count_per_fold: (i + 1) * count_per_fold])

    for i in range(folds):
        images = cifar10_folds[i][0]
        labels = cifar10_folds[i][1]
        tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        part_path = path % (2 + 3, i)
        tfr_writer = tf.python_io.TFRecordWriter(part_path, tfr_opt)

        for image, label in zip(images, labels):
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        tfr_writer.close()

        if train:
            for j in range(3):
                images_down = []

                for image, label in zip(images, labels):
                    h = image.shape[1]
                    w = image.shape[2]
                    image = torch.tensor(np.asarray(image, dtype=np.float32)).view(1, 3, h, w)

                    image_down = F.avg_pool2d(image, 2, 2).clamp_(0, 255).to('cpu', torch.uint8)

                    image_down = image_down.view(3, h // 2, w // 2).numpy()
                    images_down.append(image_down)

                part_path = cfg.DATASET.PATH % (5 - j - 1, i)
                tfr_writer = tf.python_io.TFRecordWriter(part_path, tfr_opt)
                for image, label in zip(images_down, labels):
                    ex = tf.train.Example(features=tf.train.Features(feature={
                        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))}))
                    tfr_writer.write(ex.SerializeToString())
                tfr_writer.close()

                images = images_down


def run():
    parser = argparse.ArgumentParser(description="ALAE. prepare cifar10")
    parser.add_argument(
        "--config-file",
        default="configs/cifar10.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    random.seed(0)

    dlutils.download.cifar10()
    cifar10 = dlutils.reader.Cifar10('cifar10/cifar-10-batches-bin', train=True, test=False).items
    random.shuffle(cifar10)

    cifar10_images = np.stack([x[1] for x in cifar10])
    cifar10_labels = np.stack([x[0] for x in cifar10])

    prepare_cifar10(cfg, logger, cifar10_images, cifar10_labels, train=False)
    prepare_cifar10(cfg, logger, cifar10_images, cifar10_labels, train=True)


if __name__ == '__main__':
    run()
