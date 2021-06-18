import cv2
import os
import pudb

import numpy as np
import matplotlib.pyplot as plt
from glob import glob

class FeatureVisualizer:
    def __init__(self, save_dir='/mnt_venus/luis.ll/venus/vis/channel/', max_vis=500):
        self.save_dir = save_dir
        self.save_id = 0
        self.max_vis = max_vis
        if not os.path.exists(self.save_dir):
            print(f'makedirs {self.save_dir}')
            os.makedirs(self.save_dir)
        elif len(glob(self.save_dir + '*.png')) > 5:
            info = 'exist {} imgs and '.format(
                len(glob(self.save_dir + '*.png')))
            command = 'rm {}*.png'.format(self.save_dir)
            print(info + command)
            os.system(command)

    def __call__(self, output=None, output_tags=[], input_=None, input_tags=[], save=True):
        """
        save feature to vis/
        ------
        output: dict, {'hm': torch.Tensor, 'wh': torch.Tensor}
        output_tags: list, []
        input_: dict {'input': torch.Tensor}
        input_tags: list, ['input']

        """
        if self.save_id > self.max_vis:
            return

        if not (output_tags or input_tags):
            return
        data = {}
        for t in output_tags:
            if t not in output:
                continue
            data[f'{t}_o'] = output[t]
            batch_size = len(data[f'{t}_o'])

        for t in input_tags:
            if t not in input_:
                continue
            data[f'{t}_i'] = input_[t]
            batch_size = len(data[f'{t}_i'])

        for idx_img in range(batch_size):
            import pdb
            pdb.set_trace()
            for tag in data:
                img = data[tag][idx_img].cpu().data.numpy()
                if tag == 'input_i':
                    tag = '0input_i'
                    c, h, w = img.shape
                    assert c == 3
                    out_img = np.zeros((h, w, 3)).astype(np.float)
                    for i in range(c):
                        out_img[:, :, c-i-1] = (
                            img[i] - img[i].min())*255/(img[i].max()-img[i].min() + 1e-6)
                    out_img = np.uint8(out_img)
                else:
                    c, h, w = img.shape
                    cc = int(np.ceil(float(c)**0.5))
                    out_img = np.zeros((cc*h, cc*w)).astype(np.float)
                    bound_value = img.max()/2.
                    for idx, channel in enumerate(img):
                        idx_h = int(idx/cc)
                        idx_w = int(idx % cc)
                        out_img[h*idx_h: h*(idx_h+1), w*idx_w: w *
                                (idx_w+1)] = channel.copy()
                        if cc > 1:
                            out_img[h*idx_h+h-1, w*idx_w: w *
                                    (idx_w+1)] = bound_value
                            out_img[h*idx_h: h*(idx_h+1), w *
                                    idx_w+w-1] = bound_value

                save_name = '{}{}_{}.png'.format(
                    self.save_dir, str(self.save_id).zfill(3), tag)
                plt.imshow(out_img)
                if save:
                    plt.savefig(save_name)
                if idx_img == 0:
                    print('\nsave fea: {}'.format(save_name))
            self.save_id += 1
