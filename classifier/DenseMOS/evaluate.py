import torch
from torch.utils.data import DataLoader


from DenseMOS import DenseMOS
from MOSDataset import MOSDataset
loss_fn = torch.nn.MSELoss() 

if __name__ == "__main__":
    # Load the best model
    best_model_path = "/home/aleph/tesis/classifier/best_model.pth"
    dense_mos = DenseMOS(input_dim=768, hidden_dim=128, dropout_prob=0.2)  # Ensure correct model initialization
    dense_mos.load_state_dict(torch.load(best_model_path))  # Load the saved model
    dense_mos.eval()  # Set model to evaluation mode

    # Evaluate the model on the test set
    test_csv_path = "test_set_embeddings.csv"
    test_dataset = MOSDataset(test_csv_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # DataLoader for testing
    test_loss = 0.0
    with torch.no_grad():  # No gradients needed during evaluation
        for inputs, targets in test_loader:
            outputs = dense_mos(inputs)  # Forward pass
            loss = loss_fn(outputs, targets)  # Calculate loss
            test_loss += loss.item()  # Accumulate the loss

    # Average test loss over all batches
    avg_test_loss = test_loss / len(test_loader)
    print("Test Loss:", avg_test_loss)  # Evaluate the model's performance on the test set
