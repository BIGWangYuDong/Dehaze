from Dehaze.core.Datasets.builder import PIPELINES
import torchvision.transforms as transforms
import numpy as np

# transforms.XXX((H,W))
# Image.open  ->  XX.size ->  W,H
@PIPELINES.register_module()
class Resize(object):
    def __init__(self,
                 img_scale=None,
                 keep_ratio=True
                 ):
        if img_scale is None:
            self.img_scale = None
        else:
            assert isinstance(img_scale, (int, tuple))
            self.img_scale = img_scale
        self.keep_ratio = keep_ratio

    def __call__(self, results):
        image, gt = results['image'], results['gt']
        if isinstance(self.img_scale, int):
            h, w = self.img_scale, self.img_scale
        else:
            h, w = self.img_scale
        osize = [h, w]
        transform = transforms.Resize(osize)
        results['image'] = transform(image)
        results['gt'] = transform(gt)
        return results


@PIPELINES.register_module()
class RandomCrop(object):
    def __init__(self, img_scale=None):
        if img_scale is None:
            self.img_scale = None
        else:
            assert isinstance(img_scale, (int, tuple))
            self.img_scale = img_scale

    def __call__(self, results):
        image, gt = results['image'], results['gt']
        if isinstance(self.img_scale, int):
            h, w = self.img_scale, self.img_scale
        else:
            h, w = self.img_scale
        osize = [h, w]
        transform = transforms.RandomCrop(osize)
        results['image'] = transform(image)
        results['gt'] = transform(gt)
        return results

@PIPELINES.register_module()
class RandomFlip(object):
    def __init__(self, flip_ratio=0.5):
        if flip_ratio is None:
            self.flip_ratio = None
        else:
            assert isinstance(flip_ratio, float)
            self.flip_ratio = flip_ratio

    def __call__(self, results):
        image, gt = results['image'], results['gt']
        transform = transforms.RandomHorizontalFlip(p=self.flip_ratio)
        results['image'] = transform(image)
        results['gt'] = transform(gt)
        return results


@PIPELINES.register_module()
class Pad(object):
    def __init__(self, size_divisor=None, mode='pad'):
        '''
        mode = pad or resize
        '''
        if size_divisor is None:
            self.divisor = None
        else:
            assert isinstance(size_divisor, int)
            self.divisor = size_divisor
        self.mode = mode

    def __call__(self, results):
        image, gt = results['image'], results['gt']

        pad_w = int(np.ceil(image.size[0] / self.divisor)) * self.divisor
        pad_h = int(np.ceil(image.size[1] / self.divisor)) * self.divisor
        if self.mode == 'pad':
            w = image.size[0]
            h = image.size[1]
            padding = (0, 0, pad_w - w, pad_h - h)
            transform = transforms.Pad(padding)
        else:
            transform = transforms.Resize((pad_h, pad_w))
        results['image'] = transform(image)
        results['gt'] = transform(gt)
        return results

# 再建一个 resize到可整除的

@PIPELINES.register_module()
class ImageToTensor(object):
    def __call__(self, results):
        image, gt = results['image'], results['gt']
        totensor = transforms.ToTensor()
        results['image'] = totensor(image)
        results['gt'] = totensor(gt)
        return results


@PIPELINES.register_module()
class Normalize(object):
    def __init__(self,
                 mean=(0.5, 0.5, 0.5),
                 std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
    def __call__(self, results):
        image, gt = results['image'], results['gt']
        Normalize = transforms.Normalize(mean=self.mean, std=self.std)
        results['image'] = Normalize(image)
        results['gt'] = Normalize(gt)
        return results
