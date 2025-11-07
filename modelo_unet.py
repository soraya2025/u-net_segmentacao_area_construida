import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        self.enc_conv1 = self.double_conv(in_channels, 64)
        self.enc_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv2 = self.double_conv(64, 128)
        self.enc_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottom_conv = self.double_conv(128, 256)
        self.dec_upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv1 = self.double_conv(256, 128)
        self.dec_upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = self.double_conv(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Inicialização de pesos Kaiming
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.enc_conv1(x)
        x2 = self.enc_conv2(self.enc_pool1(x1))
        x = self.bottom_conv(self.enc_pool2(x2))
        x = self.dec_upconv1(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec_conv1(x)
        x = self.dec_upconv2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec_conv2(x)
        x = self.final_conv(x)
        return x # <--- O retorno deve ser o output bruto
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # Adicionado Batch Normalization
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # Adicionado Batch Normalization
            nn.ReLU(inplace=True)
        )


