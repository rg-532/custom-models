import torch
import torch.nn as nn


def conv3x3x3(in_chans, out_chans, groups=1):
    return nn.Conv3d(in_chans, out_chans, kernel_size=3, stride=1,
                     padding=1,  bias=False, groups=groups)


def conv1x1x1(in_chans, out_chans, stride=1):
    return nn.Conv3d(in_chans, out_chans, kernel_size=1,
                     stride=stride,  bias=False)


class ResNeXtBlockC(nn.Module):
    chans_multiplier = 2

    def __init__(self, in_chans, mid_chans, cardinality,
                 stride=1, bypass=None):
        super().__init__()

        self.in_chans = in_chans
        self.mid_chans = mid_chans
        self.cradinality = cardinality
        self.out_chans = mid_chans * self.chans_multiplier
        self.stride = stride

        self.conv1 = conv1x1x1(in_chans, mid_chans, stride)
        self.bn1 = nn.BatchNorm3d(mid_chans)

        self.conv2 = conv3x3x3(mid_chans, mid_chans, groups=cardinality)
        self.bn2 = nn.BatchNorm3d(mid_chans)

        self.conv3 = conv1x1x1(mid_chans, self.out_chans)
        self.bn3 = nn.BatchNorm3d(self.out_chans)

        self.relu = nn.ReLU(inplace=True)
        self.bypass = bypass

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.bypass is not None:
            residual = self.bypass(x)

        out += residual
        out = self.relu(out)

        return out


class IdentityBypass(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    

class ConvProjectionBypass(nn.Module):
    def __init__(self, in_chans, out_chans, stride=(1, 2, 2)):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.stride = stride

        self.conv = conv1x1x1(in_chans, out_chans, stride)
        self.bn = nn.BatchNorm3d(out_chans)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        return out


class ResNeXt(nn.Module):
    def __init__(self,
                 block_type,
                 num_blocks_in_layers,
                 num_chans_in_layers,
                 cardinality=32,
                 in_chans=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 projection_bypass_type="conv",
                 widen_factor=1.0,
                 num_classes=1000):
        super().__init__()

        self.num_chans_in_layers = [int(x * widen_factor) for x in num_chans_in_layers]
        self.in_chans = in_chans
        self.conv1_t_size = conv1_t_size

        self.conv1 = nn.Conv3d(in_chans, num_chans_in_layers[0],
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(num_chans_in_layers[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        in_chans = num_chans_in_layers[0]
        self.layer1, in_chans = self._make_layer(block_type, in_chans,
                                                 num_chans_in_layers[0],
                                                 cardinality,
                                                 num_blocks_in_layers[0],
                                                 projection_bypass_type,
                                                 stride=(1, 1, 1),
                                                 has_projection=True)
        self.layer2, in_chans = self._make_layer(block_type, in_chans,
                                                 num_chans_in_layers[1],
                                                 cardinality,
                                                 num_blocks_in_layers[1],
                                                 projection_bypass_type,
                                                 stride=(1, 2, 2),
                                                 has_projection=True)
        self.layer3, in_chans = self._make_layer(block_type, in_chans,
                                                 num_chans_in_layers[2],
                                                 cardinality,
                                                 num_blocks_in_layers[2],
                                                 projection_bypass_type,
                                                 stride=(1, 2, 2),
                                                 has_projection=True)
        self.layer4, in_chans = self._make_layer(block_type, in_chans,
                                                 num_chans_in_layers[3],
                                                 cardinality,
                                                 num_blocks_in_layers[3],
                                                 projection_bypass_type,
                                                 stride=(1, 2, 2),
                                                 has_projection=True)

        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc = nn.Linear(in_chans, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block_type, in_chans, mid_chans, cardinality, num_blocks,
                    projection_bypass_type, stride, has_projection):
        out_chans = mid_chans * block_type.chans_multiplier

        first_bypass_block = IdentityBypass()
        if has_projection:
            if projection_bypass_type == "conv":
                first_bypass_block = ConvProjectionBypass(in_chans, out_chans, stride)

        layers = [
            block_type(in_chans, mid_chans, cardinality, stride=stride,
                       bypass=first_bypass_block)
        ]
        
        for _ in range(1, num_blocks):
            layers.append(block_type(out_chans, mid_chans, cardinality))

        return nn.Sequential(*layers), out_chans

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def generate_model(model_depth, **kwargs):
    assert model_depth in [50, 101, 152, 200]
    inner_channels = [128, 256, 512, 1024]

    if model_depth == 50:
        layer_sizes = [3, 4, 6, 3]
    elif model_depth == 101:
        layer_sizes = [3, 4, 23, 3]
    elif model_depth == 152:
        layer_sizes = [3, 8, 36, 3]
    elif model_depth == 200:
        layer_sizes = [3, 24, 36, 3]
    
    model = ResNeXt(ResNeXtBlockC, layer_sizes, inner_channels, **kwargs)

    return model
