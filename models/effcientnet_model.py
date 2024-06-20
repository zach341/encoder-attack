import torch
import torch.nn as nn
from torchvision.models.efficientnet import efficientnet_b0

class Effcientencoder(nn.Module):
    def __init__(self, feature_dim = 512):
        super(Effcientencoder, self).__init__()
        self.f = []
        model_name = efficientnet_b0(pretrained=False)
        # print(model_name)
        for name, module in model_name.named_children():
            if name != 'classifier':
                self.f.append(module)
        self.f = self.f[0:-1]

        self.f = nn.Sequential(*self.f)

        projection_model = nn.Sequential(nn.Linear(1280, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        
        self.g = projection_model
    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        feature = self.g(feature)
        return feature
    
if __name__ == "__main__":
    model = Effcientencoder()
    img = torch.randn(64,3,32,32)
    y = model(img)
    print(y.shape)



