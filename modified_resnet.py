import torch
import torch.nn as nn

class ModifiedResNet(nn.Module):
    """
    Resnet50 model structure, with modification that it also returns the image features
    before the fully connected layer. This is used for loss calculation.
    """
    def __init__(self, original_model):
        super(ModifiedResNet, self).__init__()

        # add layers from the original model
        for name, module in original_model.named_children():
            setattr(self, name, module)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = x # [Batch_size, number of filters, feature_map_height, feature_map_width]

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.fc(x)
        return logits, features