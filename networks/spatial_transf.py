import torch.nn as nn
import torch.nn.functional as F
import torch

# based on https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
class SpatialTransf (nn.Module):
    def __init__(self, in_channels, spatial_dims=None):


        super(SpatialTransf, self).__init__()
        self._in_ch = in_channels 
        self.spatial_dims = spatial_dims

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            #nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            #nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            #nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            #nn.BatchNorm2d(10),
            nn.ReLU(True),

            
            # Regressor for the 3 * 2 affine matrix
            # Lambda(lambda x: print(x.size())),
            Lambda(lambda x: x.view(-1, 10 * 13 * 13)), # calculated from satellite size 256x256, use the line above
            nn.Linear(10 * 13 * 13, 64),
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
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)   