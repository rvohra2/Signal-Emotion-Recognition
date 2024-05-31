import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import config
import torch.nn.functional as F

# Create a SummaryWriter to log training metrics and hyperparameters
writer = SummaryWriter('logs')

# Validation loop
def validate(model, val_loader, criterion):
    # Set the model to evaluation mode
    model.eval()

    # Initialize total loss, correct predictions, total samples, and counter
    total_loss = 0.0
    correct = 0
    total_samples = 0
    cnt = 0

    # Create a tqdm progress bar for the validation loader
    tqdm_loader = tqdm(val_loader, desc='Validation')

    # Disable gradient tracking for validation
    with torch.no_grad():
        # Loop through each batch in the validation loader
        for inputs, labels in tqdm_loader:
            # Move inputs and labels to the specified device (e.g. GPU or CPU)
            device = torch.device(config.DEVICE)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Increment counter
            cnt +=1
            # Forward pass: get outputs from the model
            outputs = model(inputs)
            # One-hot encode the labels
            t = F.one_hot(labels, config.OUTPUT_LAYER)
            # Calculate the loss
            loss = criterion(outputs, t)
            total_loss += loss.item()

            # Get the max of predicted labels
            _, predicted = torch.max(outputs, 1)
            
            # Count the correct predictions
            correct += (predicted == t).sum().item()

            # Update the tqdm progress bar with the current loss
            tqdm_loader.set_postfix(loss=loss.item())

            # Accumulate the total samples
            total_samples += labels.size(0)

            # Log the validation loss to the SummaryWriter
            writer.add_scalar('Validation/loss', loss, cnt)
    
    # Calculate the validation accuracy
    accuracy = 100.0 * correct / total_samples

    # Print the validation loss and accuracy
    print(f"Validation Loss: {total_loss / len(val_loader)}, Accuracy: {accuracy:.2f}%")