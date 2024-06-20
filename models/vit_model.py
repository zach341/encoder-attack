import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16,vit_b_32,vit_h_14,vit_l_16,vit_l_32

class Vitencoder(nn.Module):
    def __init__(self, feature_dim=512):
        super(Vitencoder, self).__init__()

        self.f = []
        self.model = vit_b_16(pretrained=False) # Remove the classification linear layer
        
        projection_model = nn.Sequential(nn.Linear(768, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        
        self.g = projection_model

    def forward(self, x):
        
        x = self.model(x)
        x = self.g(x)

        return x

    
if __name__ == "__main__":
    model_1 = Vitencoder()
    # model_1 = vit_b_16()
    img = torch.randn(64, 3, 32, 32)
    y = model_1(img)
    print(y.shape)
