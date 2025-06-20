import torch
import torch.nn as nn




def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    # if n_calss == 1 -> directly predict the [0-1] depth
    # if n_class == 64 -> divide the [0-1] depth into 64 classes
    def __init__(self, n_class, ref=True):
        super().__init__()
        if ref:
            self.input_dim = 2
        else:
            self.input_dim = 1

        self.dconv_down1 = double_conv(self.input_dim, 64)
        self.dconv_down2 = double_conv(64, 128)
        # self.dconv_down3 = double_conv(128, 256)
        # self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        # self.dconv_up3 = double_conv(256 + 512, 256)
        # self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)

        self.n_class = n_class
        
        
    def forward(self, x, ref):
        if self.input_dim == 2:
            x = torch.cat([x, ref], dim=1) 

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        x = self.dconv_down2(x)
        # conv2 = self.dconv_down2(x)
        # x = self.maxpool(conv2)
        
        # conv3 = self.dconv_down3(x)
        # x = self.maxpool(conv3)   
        
        # x = self.dconv_down4(x)
        
        # x = self.upsample(x)        
        # x = torch.cat([x, conv3], dim=1)
        
        # x = self.dconv_up3(x)
        # x = self.upsample(x)        
        # x = torch.cat([x, conv2], dim=1)       

        # x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
    






class ConvMLP(nn.Module):
    def __init__(self,  n_class, ref=True):
        super(ConvMLP, self).__init__()
        if ref:
            self.input_dim = 2
        else:
            self.input_dim = 1
        
        self.n_class = n_class

        # Convolutional layers
        self.conv1 = nn.Conv2d(self.input_dim, 64, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3)
        self.relu = nn.ReLU(inplace=True)

        # MLP
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_class)

    def forward(self, x, ref=None):
        if self.input_dim == 2:
            x = torch.cat([x, ref], dim=1) 
        
        # Pass the input through the CNN layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Reshape the tensor so it can be passed through the MLP
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)

        # Pass the output of the CNN through the MLP
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        x = x.permute(0, 2, 1).view(b, self.n_class, h, w)

        return x


if __name__ == '__main__':
    # random 480*640 tensor
    x = torch.randn(1, 1, 480, 640)
    model = UNet(n_class=1)
    pred = model(x)
    print(pred.shape)
