from tqdm.auto import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from MOSDataset import MOSDataset
from DenseMOS import DenseMOS

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # assuming that the embeddings have already been extracted... 

    # Create the training DataLoader
    train_csv_path = "/home/aleph/tesis/classifier/train.csv"
    train_dataset = MOSDataset(train_csv_path, split='train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)  # DataLoader with batching and shuffling

    # Create the validation DataLoader
    val_csv_path = "/home/aleph/tesis/classifier/val.csv"
    val_dataset = MOSDataset(val_csv_path, split='val')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Create the testing DataLoader
    test_csv_path = "/home/aleph/tesis/classifier/test.csv"
    test_dataset = MOSDataset(test_csv_path, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2) 

    # Define the model parameters
    input_dim = 768  # Single 768-dimensional input
    hidden_dim = 128  # Hidden dimension for dense layers
    dropout_prob = 0.5  # Dropout probability
    num_layers = 12  # Number of layers in the Wav2Vec2 model

    # Instantiate the DenseMOS model
    dense_mos = DenseMOS(input_dim, hidden_dim, dropout_prob, num_layers)
    dense_mos.to(device)  # Move the model to the device

    # Instantiate the DenseMOS model
    print(summary(dense_mos, input_size=(12,768)))

    # Number of epochs and batch size
    num_epochs = 5 # Number of training epochs
    batch_size = 32  # Batch size for training

    # DataLoader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Ensure batch size and shuffle
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for validation

    # define loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(dense_mos.parameters(), lr=1e-4)
    # uncomment for L2 regularization
    # optimizer = torch.optim.Adam(dense_mos.parameters(), lr=1e-4, weight_decay=1e-5)

    # Track training and validation loss
    train_losses = []
    val_losses = []

    # For early stopping
    best_val_loss = float('inf')  # To track the best validation loss
    patience = 10  # Patience for early stopping
    no_improvement_count = 0  # Counter for epochs without improvement

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Decay LR every 10 epochs

    for epoch in range(num_epochs):
        # Training phase with progress bar
        dense_mos.train()  # Set model to training mode
        train_loss = 0.0  # Initialize the training loss
        
        # Use tqdm for progress tracking
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for inputs, targets in train_loader:
                inputs = inputs.to(device)  # Move inputs to the GPU
                targets = targets.to(device)  # Move targets to the GPU

                optimizer.zero_grad()  # Zero out the gradients
                
                outputs = dense_mos(inputs)  # Forward pass
                loss = loss_fn(outputs, targets)  # Compute the loss
                loss.backward()  # Backpropagation
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(dense_mos.parameters(), 1.0)  # Adjust value if needed
                
                optimizer.step()  # Update the weights
                
                train_loss += loss.item()  # Accumulate the loss
                
                pbar.update(1)  # Update the progress bar

        train_loss /= len(train_loader)  # Average loss over all batches
        train_losses.append(train_loss)  # Save the training loss

        # Validation phase
        dense_mos.eval()  # Set model to evaluation mode
        val_loss = 0.0  # Initialize the validation loss

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)  # Move inputs to the GPU
                targets = targets.to(device)  # Move targets to the GPU

                outputs = dense_mos(inputs)  # Forward pass
                loss = loss_fn(outputs, targets)  # Calculate loss
                val_loss += loss.item()  # Accumulate the validation loss

        val_loss /= len(val_loader)  # Average validation loss over all batches
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        # scheduler.step() # Step the scheduler

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(dense_mos.state_dict(), "best_model.pth")  # Save the best model
            no_improvement_count = 0  # Reset counter
        else:
            no_improvement_count += 1  # Increment counter if no improvement

        # Early stopping check
        if no_improvement_count >= patience:
            print(f"Stopping early after {epoch + 1} epochs due to no improvement in validation loss.")
            break

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model (optional)
            torch.save(dense_mos.state_dict(), "worst_model.pth")
        # save train and val losses to csv
        losses = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses})
        losses.to_csv('losses_je.csv', index=False)
