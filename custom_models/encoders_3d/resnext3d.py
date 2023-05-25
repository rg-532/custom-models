import torch.nn as nn

from custom_models.basic_models_3d import resnext3d

channels = [128, 256, 512, 1024, 2048]


class ResNeXtEncoder(resnext3d.ResNeXt):
    def __init__(self, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth

        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


def generate_encoder(depth, model_depth, **kwargs):
    assert model_depth in [50, 101, 152, 200]
    inner_channels = channels[:-1]

    if model_depth == 50:
        layer_sizes = [3, 4, 6, 3]
    elif model_depth == 101:
        layer_sizes = [3, 4, 23, 3]
    elif model_depth == 152:
        layer_sizes = [3, 8, 36, 3]
    elif model_depth == 200:
        layer_sizes = [3, 24, 36, 3]

    encoder = ResNeXtEncoder(depth, block_type=resnext3d.ResNeXtBlockC, num_blocks_in_layers=layer_sizes,
                             num_chans_in_layers=inner_channels, **kwargs)

    return encoder
