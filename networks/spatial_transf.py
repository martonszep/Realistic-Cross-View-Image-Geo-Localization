import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import torchvision
from data.cvusa_utils import convert_image_np

# based on https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
class SpatialTransf (nn.Module):
    def __init__(self, in_channels, spatial_dims=None, sate_input=True):
        super(SpatialTransf, self).__init__()

        self._in_ch = in_channels 
        self.spatial_dims = spatial_dims
        fc_size = 13*13 if sate_input==True else 4*36

        # Spatial transformer localization-network
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
        theta = self.localization(x)
        theta = theta.view(-1, 2, 3) # 2D affine transf param

        if self.spatial_dims is not None:
            grid = F.affine_grid(theta, (x.size(0), self._in_ch, *self.spatial_dims))
        else:
            grid = F.affine_grid(theta, (x.size()))
        
        x = F.grid_sample(x, grid, mode='bilinear')

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
    def __init__(self, in_channels, spatial_dims, latent_channels=8):
        super(ComposedSpatialTransf, self).__init__()

        # self.spatial_block = nn.Sequential(
        #     SpatialTransf(in_channels, spatial_dims, sate_input=True)
        # )

        self.spatial_block = nn.Sequential(
            SpatialTransf(in_channels, spatial_dims, sate_input=True),
            ResidualBlock(in_channels, latent_channels),
            SpatialTransf(latent_channels, spatial_dims, sate_input=False),
            ResidualBlock(latent_channels, in_channels),
            SpatialTransf(in_channels, spatial_dims, sate_input=False)
        )

    def forward(self, x):
        return self.spatial_block(x)  

    # def visualize_stn(self, data):
    #     with torch.no_grad():

    #         input_tensor = data.cpu()
    #         transformed_input_tensor = self.forward(data).cpu()

    #         in_grid = convert_image_np(
    #             torchvision.utils.make_grid(input_tensor))

    #         out_grid = convert_image_np(
    #             torchvision.utils.make_grid(transformed_input_tensor))

    #         # Plot the results side-by-side
    #         f, axarr = plt.subplots(1, 2)
    #         axarr[0].imshow(in_grid)
    #         axarr[0].set_title('Dataset Satellite Images')

    #         axarr[1].imshow(out_grid)
    #         axarr[1].set_title('Transformed Satellite Images')
    #     return f