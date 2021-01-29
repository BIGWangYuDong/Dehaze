import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict
from Dehaze.core.Models import BACKBONES
from Dehaze.core.Models.weight_init import normal_init, xavier_init
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from collections import OrderedDict


@BACKBONES.register_module()
class DenseNet192(nn.Module):
    def __init__(self, growth_rate=32, block_config=(3, 6, 12, 24, 48),
                 num_init_features=32, bn_size=4, drop_rate=0, memory_efficient=False):
        super(DenseNet192, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True))
        ]))
        num_features = num_init_features
        # block1
        num_layers = block_config[0]
        self.denseblock1 = nn.Sequential(OrderedDict([]))
        block1 = DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient
        )
        self.denseblock1.add_module('denseblock%d' % 1, block1)
        num_features = num_features + num_layers * growth_rate
        trans1 = Transition(num_input_features=num_features,
                            num_output_features=num_features // 2)
        num_features = num_features // 2
        self.denseblock1.add_module('transition%d' % 1, trans1)
        self.denseblock1_pool = nn.Sequential(OrderedDict([
            ('pool1', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        # block2
        num_layers = block_config[1]
        self.denseblock2 = nn.Sequential(OrderedDict([]))
        block2 = DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient
        )
        self.denseblock2.add_module('denseblock%d' % 2, block2)
        num_features = num_features + num_layers * growth_rate
        trans2 = Transition(num_input_features=num_features,
                            num_output_features=num_features // 2)
        num_features = num_features // 2
        self.denseblock2.add_module('transition%d' % 2, trans2)
        self.denseblock2_pool = nn.Sequential(OrderedDict([
            ('pool2', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        # block3
        num_layers = block_config[2]
        self.denseblock3 = nn.Sequential(OrderedDict([]))
        block3 = DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient
        )
        self.denseblock3.add_module('denseblock%d' % 3, block3)
        num_features = num_features + num_layers * growth_rate
        trans3 = Transition(num_input_features=num_features,
                            num_output_features=num_features // 2)
        num_features = num_features // 2
        self.denseblock3.add_module('transition%d' % 3, trans3)
        self.denseblock3_pool = nn.Sequential(OrderedDict([
            ('pool3', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        # block4
        num_layers = block_config[3]
        self.denseblock4 = nn.Sequential(OrderedDict([]))
        block4 = DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient
        )
        self.denseblock4.add_module('denseblock%d' % 4, block4)
        num_features = num_features + num_layers * growth_rate
        trans4 = Transition(num_input_features=num_features,
                            num_output_features=num_features // 2)
        num_features = num_features // 2
        self.denseblock4.add_module('transition%d' % 4, trans4)
        self.denseblock4_pool = nn.Sequential(OrderedDict([
            ('pool4', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        # block5
        num_layers = block_config[4]
        self.denseblock5 = nn.Sequential(OrderedDict([]))
        block5 = DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient
        )
        self.denseblock5.add_module('denseblock%d' % 5, block5)
        num_features = num_features + num_layers * growth_rate
        trans5 = Transition(num_input_features=num_features,
                            num_output_features=num_features // 2)
        self.denseblock5.add_module('transition%d' % 5, trans5)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)

    def forward(self, x):
        out1_in = self.features(x)

        out1_out = self.denseblock1(out1_in)
        out2_in = self.denseblock1_pool(out1_out)

        out2_out = self.denseblock2(out2_in)
        out3_in = self.denseblock2_pool(out2_out)

        out3_out = self.denseblock3(out3_in)
        out4_in = self.denseblock3_pool(out3_out)

        out4_out = self.denseblock4(out4_in)
        out5_in = self.denseblock4_pool(out4_out)

        out5_out = self.denseblock5(out5_in)
        return out1_out, out2_in, out2_out, out3_in, \
               out3_out, out4_in, out4_out, \
               out5_in, out5_out


class DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(*inputs)

        return cp.checkpoint(closure, input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))

