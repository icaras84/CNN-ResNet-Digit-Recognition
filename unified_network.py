"""
William Phomphakdee
CSE 163

This file contains code for defining a flexible neural network
that utilizes blocks and stages to process images.
"""

from typing import Type

import torch
import torch.nn as nn


class NetworkBlock(nn.Module):
    """
    A subclass of a PyTorch module that specifies what a block's constructor
    parameters are.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Constructor stub for subclasses.
        :param in_channels:
        :param out_channels:
        :param stride:
        """
        super().__init__()


class ConvolveNetworkBlock(NetworkBlock):
    """
    A block that processes like a normal convolution layer.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(in_channels, out_channels, stride)
        self._block: nn.Sequential = nn.Sequential(
            # construct a convolution layer (w/o bias)
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # set ReLU as the activation function
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor
        :param x: a tensor that is passed through the block for processing
        :return: another tensor that is the result of convolve, normalization,
        and activation with this block
        """
        return self._block(x)


class ResidualNetworkBlock(NetworkBlock):
    """
    A block that represents a block in a ResNet. Contains two convolutional
    layers and one passthrough layer.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(in_channels, out_channels, stride)

        # reuse the Convolve block because first "layer" of this residual
        # block is "normal"
        self._conv_block: ConvolveNetworkBlock = \
            ConvolveNetworkBlock(in_channels, out_channels, stride)

        # manually define the second convolution layer because
        # residual skip/shortcut
        self._conv2: nn.Module = nn.Conv2d(out_channels, out_channels,
                                           kernel_size=3, stride=1,
                                           padding=1, bias=False)
        self._bn2: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)

        # construct the shortcut layer
        self._shortcut: nn.Sequential = nn.Sequential()

        # if there is downsampling, then we must define the shortcut as a
        # down-sample layer to match output dims
        if in_channels != out_channels or stride != 1:
            self._shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self._activation: nn.Module = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor
        :param x: a tensor that is passed through the block for processing
        :return: another tensor that is the sum of the pass-through and the
        result from two convolutional layers
        """
        shortcut_result = self._shortcut(x)

        # layer 1 conv; nothing special
        out = self._conv_block(x)

        # layer 2 conv is special since we need add residual
        # after result, but before activation
        out = self._bn2(self._conv2(out))

        out += shortcut_result
        return self._activation(out)


class UnifiedNetwork(nn.Module):
    """
    A neural network that can have stages of arbitrary block types
    of any number.
    """
    def __init__(self,
                 stage_configs: list[tuple[
                     Type[NetworkBlock], int, int, int, int
                 ]],
                 num_classes: int):
        super().__init__()

        self._stages: nn.Sequential = nn.Sequential()

        for i, (factory, num_blocks, in_ch, out_ch, stride) in \
                enumerate(stage_configs):

            blocks = [factory(in_ch, out_ch, stride)]
            blocks += [
                factory(out_ch, out_ch, 1) for _ in range(num_blocks - 1)
            ]

            self._stages.add_module(f'stage_{i}', nn.Sequential(*blocks))

        # some processing after the stages are all processed
        last_stage = stage_configs[-1][3]
        self._post_processing: nn.Module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_stage, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._post_processing(self._stages(x))
