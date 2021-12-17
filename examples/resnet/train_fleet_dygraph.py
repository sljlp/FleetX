# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.vision.models import ResNet
from paddle.vision.models.resnet import BottleneckBlock
from paddle.io import Dataset, BatchSampler, DataLoader
import time
import os
#paddle.set_device("cpu")
output_dir="saved_model"
load_from_file=True

base_lr = 0.1
momentum_rate = 0.9
l2_decay = 1e-4

epoch = 10000
batch_num = 100
batch_size = 32
class_dim = 102

img_path="/code_lp/imgnet/val"

show_info_interval = 10

paddle.seed(1020)
import cv2
# define a random dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        np.random.seed(fleet.worker_index()+1000)
        import os
        self.labels = os.listdir(img_path)
        self.labels.sort()
        assert len(self.labels) == class_dim
        self.label_id_dict = dict()
        for i, l in enumerate(self.labels):
            self.label_id_dict[l] = i
        self.img_paths = []
        self.img_paths = os.popen(f"find {img_path} -type f").readlines()
        self.img_paths = [ path.strip() for path in self.img_paths ]
        self.img_paths.sort()

    def __getitem__(self, idx):
        img_idx = np.random.randint(len(self.img_paths))
        image = cv2.imread(self.img_paths[img_idx]).astype("float32")
        img2 = cv2.resize(image, (224,224))
        img3 = np.transpose(img2, (2, 0, 1)) / 128.0 - 1.0
        label_str = self.img_paths[img_idx].split("/")[-2]
        assert label_str in self.label_id_dict

        # image = np.random.random([3, 224, 224]).astype('float32')
        # label = np.random.randint(0, class_dim - 1, (1, )).astype('int64')
        return img3, np.array([self.label_id_dict[label_str]]).astype("int64")

    def __len__(self):
        return self.num_samples

def optimizer_setting(parameter_list=None):
    optimizer = paddle.optimizer.Momentum(
        learning_rate=base_lr,
        momentum=momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(l2_decay),
        parameters=parameter_list)
    return optimizer


def train_resnet():
    fleet.init(is_collective=True)

    resnet = ResNet(BottleneckBlock, 50, num_classes=class_dim)
    optimizer = optimizer_setting(parameter_list=resnet.parameters())
    optimizer = fleet.distributed_optimizer(optimizer)
    resnet = fleet.distributed_model(resnet)

    dataset = RandomDataset(batch_num * batch_size)
    train_loader = DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=True,
                    num_workers=0)
    t1 = time.time()

    if load_from_file and os.path.exists(output_dir):
        model = paddle.load(output_dir + "/model.pdparams")
        # opt = paddle.load(os.path.join(output_dir, "model_state.pdopt"))
        resnet.set_state_dict(model)
        # optimizer.set_state_dict(opt)

    for eop in range(epoch):
        resnet.train()
        
        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True

            out = resnet(img)
            loss = paddle.nn.functional.cross_entropy(input=out, label=label)
            avg_loss = paddle.mean(x=loss)
            acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

            avg_loss.backward()
            optimizer.step()
            resnet.clear_gradients()

            if (batch_id+1) % show_info_interval == 0:
                t2 = time.time()
                print("[Epoch %d, batch %d] loss: %.5f, acc1: %.5f, acc5: %.5f, speed: %.2f samples/s" % (eop, batch_id, avg_loss, acc_top1, acc_top5, (show_info_interval*32/(t2 - t1))))
                t1 = time.time()
    if fleet.worker_index() == 0:
        
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        model_to_save = resnet._layers if isinstance(
                                resnet, paddle.DataParallel) else resnet
        # paddle.save(model_to_save.state_dict(), output_dir+"/model.pdparams")
        # paddle.save(
        #     optimizer.state_dict(),
        #     os.path.join(output_dir, "opt_state.pdopt"))

if __name__ == '__main__':
    train_resnet()
