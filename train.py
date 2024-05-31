import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import config
import torch.nn.functional as F

# Create a SummaryWriter to log training metrics and hyperparameters
writer = SummaryWriter('logs')

# Define the training loop function
def train(model, train_loader, criterion, optimizer, num_epochs):
    # Loop through each epoch
    for epoch in range(num_epochs):
        # Create a tqdm progress bar for the training loader
        tqdm_loader = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        # Set the model to training mode
        model.train()
        # Initialize total loss and counter
        total_loss = 0.0
        cnt = 0

        # Loop through each batch in the training loader
        for inputs, labels in tqdm_loader:
            # Increment counter
            cnt+=1

            # Zero the gradients
            optimizer.zero_grad()

            #Move inputs and labels to the specified device (e.g. GPU or CPU)
            device = torch.device(config.DEVICE)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass: get outputs from the model
            outputs = model(inputs)
            # One-hot encode the labels
            t = F.one_hot(labels, config.OUTPUT_LAYER)

            # Calculate the loss
            loss = criterion(outputs, t)
            
            # Backward pass: compute gradients
            loss.backward()
            optimizer.step()

            # Update the tqdm progress bar with the current loss
            tqdm_loader.set_postfix(loss=loss.item())

            # Log the training loss to the SummaryWriter
            writer.add_scalar('Training/loss', loss, cnt)

            # Accumulate the total loss
            total_loss += loss.item()
        
        # Print the average training loss for the epoch
        print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader)}")

