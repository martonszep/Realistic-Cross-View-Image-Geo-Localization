import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from networks.tps import TPSGridGen

class AffineLocalization(nn.Module):
    def __init__(self, in_channels, input_size) -> None:
        super(AffineLocalization, self).__init__()

        # Calculation of fully connected layer size based on conv and maxpool layers below
        fc_size = np.prod(((((np.asarray(input_size) -6) //2 -4) //2 -2) //2 -2) //2)

        # Affine spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            
            # Regressor for the 3 * 2 affine matrix
            # Lambda(lambda x: print(x.size())),
            Lambda(lambda x: x.view(-1, 10 * fc_size)),
            nn.Linear(10 * fc_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2) # 2D affine transf param
        )

        # Initialize the weights/bias with identity transformation
        self.localization[-1].weight.data.zero_()        
        self.localization[-1].bias.data.copy_(torch.tensor([1, 0, 0, 
                                                            0, 1, 0], dtype=torch.float)) # 2D affine transf param
    def forward(self, x):
        return self.localization(x)

class TPSLocalization(nn.Module):
    def __init__(self, in_channels, input_size, grid_height, grid_width, target_control_points, bounded = False) -> None:
        super(TPSLocalization, self).__init__()

        self.bounded = bounded

        # Calculation of fully connected layer size based on conv and maxpool layers below
        fc_size = np.prod(((((np.asarray(input_size) -6) //2 -4) //2 -2) //2 -2) //2)

        # Thin plate spline spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            
            # Lambda(lambda x: print(x.size())),
            Lambda(lambda x: x.view(-1, 10 * fc_size)),
            nn.Linear(10 * fc_size, 512),
            nn.ReLU(True),
            nn.Linear(512, grid_height * grid_width * 2)
        )

        # Initialize the weights/bias
        bias = torch.atanh(target_control_points).view(-1) if self.bounded else target_control_points.view(-1)
        self.localization[-1].bias.data.copy_(bias)
        self.localization[-1].weight.data.zero_()        

    def forward(self, x):
        batch_size = x.size(0)
        points = F.tanh(self.localization(x)) if self.bounded else self.localization(x)
        return points.view(batch_size, -1, 2)

# Extension of https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
class SpatialTransf (nn.Module):
    def __init__(self, in_channels, output_size, input_size, use_tps=True, span_range=0.9, grid_height=6, grid_width=10):
        super(SpatialTransf, self).__init__()

        self._in_ch = in_channels 
        self.output_size = output_size
        self.use_tps = use_tps

        if self.use_tps == True:
            r1 = r2 = span_range # span_range_width and span_range_height
            image_height, image_width = self.output_size

            assert r1 < 1 and r2 < 1 # if >= 1, atanh will cause error in TPSLocalization with bounded=True
            target_control_points = torch.Tensor(list(itertools.product(
                np.arange(-r1, r1 + 0.00001, 2.0  * r1 / (grid_height - 1)),
                np.arange(-r2, r2 + 0.00001, 2.0  * r2 / (grid_width - 1)),
            )))
            Y, X = target_control_points.split(1, dim = 1)
            target_control_points = torch.cat([X, Y], dim = 1)
            self.localization = TPSLocalization(self._in_ch, input_size, grid_height, grid_width, target_control_points)
            self.tps = TPSGridGen(image_height, image_width, target_control_points)
        else:
            self.localization = AffineLocalization(in_channels=in_channels, input_size=input_size)

    def forward(self, x):
        if self.use_tps == True:
            source_control_points = self.localization(x)
            source_coordinate = self.tps(source_control_points)
            grid = source_coordinate.view(x.size(0), *self.output_size, 2) # required input size for grid_sampler
        else:
            theta = self.localization(x)
            theta = theta.view(-1, 2, 3) # 2D affine transf param
            grid = F.affine_grid(theta, (x.size(0), self._in_ch, *self.output_size), align_corners=False)
        
        x = F.grid_sample(x, grid, mode='bilinear', align_corners=False)
        return x     

class Lambda(nn.Module):
    """
    This is just a convenience function to make e.g. `nn.Sequential` more flexible.
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class ResidualBlock(nn.Module):
    """
    The residual block used by ResNet.
    
    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first convolution
        stride: Stride size of the first convolution, used for downsampling
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()        
        if stride > 1 or in_channels != out_channels:
            # Add strides in the skip connection and zeros for the new channels.
            self.skip = Lambda(lambda x: F.pad(x[:, :, ::stride, ::stride],
                                               (0, 0, 0, 0, 0, out_channels - in_channels),
                                               mode="constant", value=0))
        else:
            self.skip = nn.Sequential()
            
        # Initialize the required layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, input):
        # Execute the required layers and functions
        xb = F.relu(self.conv1_bn(self.conv1(input)))
        z = self.conv2(xb) + self.skip(input)
        xb = F.relu(self.conv2_bn(z))
        return xb # [batch_size, num_classes] 


class ComposedSpatialTransf (nn.Module):
    def __init__(self, in_channels, output_size, input_size, latent_channels=8, use_tps=True):
        super(ComposedSpatialTransf, self).__init__()

        self.use_tps = use_tps

        self.spatial_block = nn.Sequential(
            SpatialTransf(in_channels, output_size, input_size=input_size, use_tps=self.use_tps)
        )

        # possibility to stack multiple spatial transformers with residual blocks in between
        # self.spatial_block = nn.Sequential(
        #     SpatialTransf(in_channels, output_size, input_size=True, use_tps=self.use_tps),
        #     ResidualBlock(in_channels, latent_channels),
        #     SpatialTransf(latent_channels, output_size, input_size=output_size, use_tps=self.use_tps),
        #     ResidualBlock(latent_channels, in_channels),
        #     SpatialTransf(in_channels, output_size, input_size=output_size, use_tps=self.use_tps)
        # )

    def forward(self, x):
        return self.spatial_block(x)