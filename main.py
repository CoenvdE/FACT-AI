import torch
import torchvision.models as models
from clip import clip
from data_loaders import *
from prompt_learner import *
from text_encoder_mlp import *
from modified_resnet import *  
from sinkhorn_ot_loss import *
from FACT.src.train_val_test_loops import *
from names_and_label_functions import *
from mm_loss import *
from train_val_test_loops import *
from data_loaders import *
import argparse

# Path to the dataset
# If made use of scratch files, uncomment the TMPDIR lines and use those to make the dataloaders
parser = argparse.ArgumentParser()
parser.add_argument('-DIR', type=str, help='Path to the temporary directory', default='none')
parser.add_argument('-ablation1', type=str, help='Ablation 1', default='none')
parser.add_argument('-ablation2', type=str, help='Ablation 2', default='none')
args = parser.parse_args()
path_dir = args.DIR
ablation1 = args.ablation1
ablation2 = args.ablation2

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# create dataloaders
data_path = path_dir 

# make dataloaders with train set only
train_loader, val_loader, test_loader = load(data_path + "/train")

# Get text features from the text encoder
clip_model, _ = clip.load("ViT-B/32", device)
classnames = get_ImageNet_ClassNames()

text_encoder = TextEncoder(clip_model).to(device)
prompt_learner = PromptLearner(classnames, clip_model).to(device)
total_prompt_from_labels = prompt_learner()
tokenized_total_prompt = prompt_learner.tokenized_prompts.to(device)

with torch.no_grad():
    all_prompt_features = text_encoder(total_prompt_from_labels, tokenized_total_prompt)

all_prompt_features = all_prompt_features.to(device)

# create the resnet model
resnet = models.resnet50(pretrained=True)
resnet = resnet.to(device)
modified_resnet = ModifiedResNet(resnet)
modified_resnet = modified_resnet.to(device)

# create mlp
input_dim = 512 # text encoder CLIP
hidden_dim = 512 # COEN: chat said the notation is hidden dim, output dim, so hidden dim = 512
output_dim = 49 # to match 7x7 dimension of the feature maps
text2img_dim_transform = MLP(input_dim, hidden_dim, output_dim)
text2img_dim_transform = text2img_dim_transform.to(device)

# Initialize SinkhornDistance module for OT loss
sinkhorn_loss = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean').to(device)

# train model
num_epochs = 90

# train model with learnable prompts, to be tested. ajust line in train_val_test_loops.py aswell
# train_model(modified_resnet, train_loader, manifold_matching_loss, sinkhorn_loss, text2img_dim_transform, num_epochs, device, val_loader, get_encoded_labels, ablation1, ablation2, prompt_learner, text_encoder)

# train model with fixed prompts, works
train_model(modified_resnet, train_loader, manifold_matching_loss, sinkhorn_loss, text2img_dim_transform, num_epochs, device, val_loader, get_encoded_labels, ablation1, ablation2, all_prompt_features)

# Initialize the model and test performance
lico_model = models.resnet50(pretrained=False).to(device)
lico_model.load_state_dict(torch.load(f'/home/scur1049/FACT/models/modified_resnet_{ablation1}{ablation2}.pth'))
test_model(lico_model, test_loader, device)