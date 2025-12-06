

import torch
import torch.nn as nn
import torch.nn.functional as F

# FF:
# Image (64x64x1)
# Conv layer 1 (ks = 7, st = 4, chn = 32), ReLU -> out_shape = (16x16x32), eff_RF = ks = 7, eff_st = 4
# Conv layer 2 (ks = 3, st = 2, chn = 64), ReLU -> out_shape = (8x8x64), eff_RF = 7 + (3-1)*4 = 15, eff_st = 8
# Conv layer 3 (ks = 3, st = 1, chn = 10+10), ReLU -> out_shape = (8x8x20), eff_RF = 15 + (3-1)*8 = 31, eff_st = 8
# Avgpool on [:10] -> out_shape = (1x1x10)

# RNN:
# build a transposed conv layer to map from conv3 to input
# Conv3 -> Input (TD): transpose conv with ks=9, st=8, out_chn=1, padding=1, output_padding=1
# Conv2 -> Input (TD): transpose conv with ks=5, st=4, out_chn=1, padding=1, output_padding=1
# Conv3 -> Conv3 (L) : conv layer with ks=3, st=1, out_chn=10+10
# Conv2 -> Conv2 (L): conv layer with ks=3, st=1, out_chn=64

class RCNN(nn.Module):
    def __init__(self, num_classes=10, modulation_type='multiplicative'):
        super().__init__()
        # FF
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=4, padding=3)  # (B,32,16,16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # (B,64,8,8)
        self.conv3 = nn.Conv2d(64, num_classes + 10, kernel_size=3, stride=1, padding=1)  # (B,10+10,8,8)
        # RNN
        self.deconv3_to_input = nn.ConvTranspose2d(num_classes + 10, 1, kernel_size=9, stride=8, padding=1, output_padding=1)
        self.deconv2_to_input = nn.ConvTranspose2d(64, 1, kernel_size=9, stride=8, padding=1, output_padding=1)
        self.conv3_lateral = nn.Conv2d(num_classes + 10, num_classes + 10, kernel_size=3, padding='same')
        self.conv2_lateral = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.modulation_type = modulation_type

    def forward(self, x, timesteps=5, return_actvs=False):

        activations = {}

        if 'input' not in activations:
            activations['input'] = []
            activations['input'].append(x)
        if 'conv1' not in activations:
            activations['conv1'] = []
            activations['conv1'].append(F.relu(self.conv1(activations['input'][0])))
        if 'conv2' not in activations:
            activations['conv2'] = []
            activations['conv2'].append(F.relu(self.conv2(activations['conv1'][0])))
        if 'conv3' not in activations:
            activations['conv3'] = []
            activations['conv3'].append(F.relu(self.conv3(activations['conv2'][0])))
        if 'output' not in activations:
            activations['output'] = []
            activations['output'].append(F.adaptive_avg_pool2d(activations['conv3'][0][:,:10,:,:], (1,1)).squeeze(-1).squeeze(-1))

        for t in range(1,timesteps):
            # Reconstruct input from conv3 and conv2
            print(activations['conv3'][t-1].shape, activations['conv2'][t-1].shape)
            print(self.deconv3_to_input.weight.shape, self.deconv2_to_input.weight.shape)
            recon_from_conv3 = self.deconv3_to_input(activations['conv3'][t-1])
            recon_from_conv2 = self.deconv2_to_input(activations['conv2'][t-1])
            print(recon_from_conv3.shape, recon_from_conv2.shape)
            recon_input = torch.sigmoid(recon_from_conv3 + recon_from_conv2)

            activations['input'].append(x * (2*recon_input) if self.modulation_type == 'multiplicative' else F.relu(x + (2*recon_input-1)))

            # Forward pass
            activations['conv1'].append(F.relu(self.conv1(activations['input'][t])))
            conv2_ff_out = self.conv2(activations['conv1'][t])
            conv2_l_out = self.conv2_lateral(activations['conv2'][t-1])
            activations['conv2'].append(F.relu(conv2_ff_out + conv2_l_out) if self.modulation_type == 'additive' else F.relu(conv2_ff_out) * (2*torch.sigmoid(conv2_l_out)))
            conv3_ff_out = self.conv3(activations['conv2'][t])
            conv3_l_out = self.conv3_lateral(activations['conv3'][t-1])
            activations['conv3'].append(F.relu(conv3_ff_out + conv3_l_out) if self.modulation_type == 'additive' else F.relu(conv3_ff_out) * (2*torch.sigmoid(conv3_l_out)))

            # Readout
            activations['output'].append(F.adaptive_avg_pool2d(activations['conv3'][t][:,:10,:,:], (1,1)).squeeze(-1).squeeze(-1))

        if return_actvs:
            return activations
        else:
            return activations['output']
        


if __name__ == "__main__":
    abc = RCNN()
    out = abc(torch.randn(2,1,64,64), timesteps=3, return_actvs=False)

