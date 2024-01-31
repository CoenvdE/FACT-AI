from tqdm import tqdm
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_model(modified_resnet, dataloader, manifold_matching_loss, sinkhorn_loss, text2img_dim_transform, num_epochs, device, all_prompt_features, validation_loader, get_encoded_labels, ablation1, ablation2):
    """
    Train the model.
    """

    # Check validation before training
    validate_model(modified_resnet, validation_loader, device)

    # initialize the optimizer
    optimizer = optim.SGD([
        {'params': modified_resnet.parameters()},
        {'params': text2img_dim_transform.parameters()},
    ], lr=0.03, momentum=0.9, weight_decay=0.0001)

    # Initialize the learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, num_epochs)

    # Training loop
    for epoch in range(num_epochs):
        modified_resnet.train()
        text2img_dim_transform.train()


        if (epoch+1) % 2 == 1: # Change this to show which epoch we are
            print(f"Epoch {epoch + 1}/{num_epochs}")

        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass through models
            encoded_labels = get_encoded_labels(labels, all_prompt_features)
            predictions, features_resnet = modified_resnet(images)

            # feature_maps for OT loss
            feature_maps = features_resnet.view(features_resnet.shape[0], features_resnet.shape[1], -1)
            feature_maps = F.normalize(feature_maps, dim = 2)

            # image_features for manifold loss
            image_features = F.adaptive_avg_pool2d(features_resnet, 1)
            image_features = image_features.view(images.shape[0], -1)
            image_features = F.normalize(image_features, dim = -1)

            # transform text_features dimension to match thos of the image encoder's output
            text_features = text2img_dim_transform(encoded_labels)
            text_features = F.normalize(text_features, dim = -1)

            # get temperature parameter
            temperature = text2img_dim_transform.temp

            # calculate losses
            CE_loss = torch.nn.functional.cross_entropy(predictions, labels)
            MM_loss = manifold_matching_loss(image_features, text_features, temperature)
            OT_loss = sinkhorn_loss(feature_maps, text_features)

            if ablation1 == 'mm' or ablation2 == 'mm':
                MM_loss = 0
            if ablation1 == 'ot' or ablation2 == 'ot':
                OT_loss = 0

            # params according to the paper
            alpha = 10
            beta = 1

            # Combine the losses or use them as needed
            total_loss = CE_loss + alpha * MM_loss + beta * OT_loss

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Clipping the parameter value to be within a min_val and max_val #CHOSEN BY OURSELVES
            with torch.no_grad():  # This makes sure the operation is not tracked by autograd
                text2img_dim_transform.temp.clamp_(min=0.1, max=3)


        print(f"temperature after last batch of epoch was:{temperature.item()}")

        # Evaluate on validation set or perform any other actions at the end of each epoch
        validate_model(modified_resnet, validation_loader, device)
        scheduler.step()

        print(f"Loss of last epoch in batch is: CE: {CE_loss}, OT: {OT_loss}, MM: {MM_loss}")
        # Save the model after training
        torch.save(modified_resnet.state_dict(), f'/home/scur1049/FACT/models/modified_resnet_{ablation1}{ablation2}_{epoch}.pth')
        torch.save(text2img_dim_transform.state_dict(), f'/home/scur1049/FACT/models/text2img_dim_transform_{ablation1}{ablation2}_{epoch}.pth') 


    # Save the model after training
    torch.save(modified_resnet.state_dict(), f'/home/scur1049/FACT/models/modified_resnet_{ablation1}{ablation2}.pth')
    torch.save(text2img_dim_transform.state_dict(), f'/home/scur1049/FACT/models/text2img_dim_transform_{ablation1}{ablation2}.pth') 


def validate_model(modified_resnet, dataloader, device):
    """
    Validate the model.
    """

    modified_resnet.eval()  # Set the model to evaluation mode

    # Initialize variables to track metrics
    total_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():  # No need to track gradients during validation
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass through models
            predictions, _ = modified_resnet(images)

            # Calculate accuracy
            _, predicted = torch.max(predictions.data, 1)
            total_accuracy += (predicted == labels).sum().item()
            num_batches += 1

    # Compute average losses and accuracy
    avg_accuracy = total_accuracy / (num_batches * dataloader.batch_size)

    print(f'Validation results: Accuracy: {avg_accuracy}')

    # Return to training mode
    modified_resnet.train()
    return avg_accuracy


def test_model(trained_model, test_loader, device):
    # Ensure the model is in evaluation mode
    trained_model.eval()
    
    # Initialize loss and accuracy tracking
    total_loss = 0.0
    total_correct = 0
    total_images = 0
    
    # Disable gradient computation for efficiency and reduced memory usage
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing', leave=False):
            # Move the images and labels to the device (CPU or GPU)
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass: compute predicted outputs by passing images to the model
            outputs = trained_model(images)
            
            # Convert output probabilities to predicted class
            _, preds = torch.max(outputs, 1)
            
            # Compare predictions to true label
            total_correct += (preds == labels).sum().item()
            total_images += images.size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / total_images
    accuracy = total_correct / total_images
    
    print(f'Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
