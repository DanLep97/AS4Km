from torch import nn
import torch
from torch.utils.data import DataLoader
from torcheval.metrics.functional import r2_score
from torchmetrics import PearsonCorrCoef

# Define the gated layer
class ConditionedGatedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, with_gate):
        super(ConditionedGatedLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        if with_gate:
            self.gate = nn.Linear(input_dim, output_dim)
        self.gated = with_gate

    def forward(self, x):
        h = self.linear(x) # regular transformation
        if self.gated:
            g = torch.sigmoid(self.gate(x))  # Gate output between 0 and 1
            return h * g  # Element-wise multiplication
        else:
            return h

# Neural Network Class 
class Network(nn.Module):
    def __init__(
        self, 
        hidden_dim1, 
        hidden_dim2, 
        hidden_dim3, 
        dropout1, 
        dropout2, 
        input_size=5316,
        with_gate=True,
    ): #input_size=4292):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            ConditionedGatedLayer(input_size, hidden_dim1, with_gate),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Linear(hidden_dim3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

    @property
    def device(self):
        return next(self.parameters()).device
    xaxes={
        "title": {
            "text": "Feature",
            "title_font": {"size": 18}
        }
    }
def tester(
        test_dataset, 
        net,
):
    net.eval()

    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        num_workers=16, 
        drop_last=False
    )

    device = net.device

    all_pred = []
    all_y = []
    all_idx = []
    with torch.no_grad():
        for x,y,idx in test_loader:
            x = x.to(device)
            pred = net(x)
            all_pred.append(pred.cpu())
            all_y.append(y)
            all_idx.append(idx)
        all_pred = torch.stack(all_pred).flatten()
        all_y = torch.stack(all_y).flatten()
        all_idx = torch.stack(all_idx)
    return all_pred, all_y, all_idx

class WeightSaver():
    # Evaluate if the trainer should continue to train based on epoch patience
    # and best R2 score.
    def __init__(self, net:Network, best_model_path, patience=5):
        self.best_r2 = float("-inf")
        self.patience = patience
        self.running_epochs = 0
        self.net = net
        self.best_model_path = best_model_path
    
    def save_weights(self):
        weights = self.net.state_dict()
        torch.save({
            "model_state_dict": weights,
            "best_r2": self.best_r2,  # Save the best R² value alongside the model
        }, self.best_model_path)  # Save the model's state_dict and the best R²
        print(f"Saved weights at {self.best_model_path}")

    def should_continue_train(self, score): # return True if it should continue train
        if score > self.best_r2:
            self.best_r2 = score
            self.save_weights()
            self.running_epochs = 0
            return True
        elif self.running_epochs > self.patience:
            return False
        self.running_epochs+=1
        return True

def trainer(
        epochs: int, 
        train_dataset, 
        val_dataset, 
        best_model_path: str,
        metrics_path: str,
        net: Network,
        optimizer,
        criterion
    ):
    weight_saver = WeightSaver(net=net, best_model_path=best_model_path, patience=7)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=12, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=12, 
        drop_last=True
    )

    loss_values = []
    r2_values = []
    pearson_values = []
    loss_values_val = []
    r2_values_val = []
    pearson_values_val = []

    y_true_train = []
    y_pred_train = []
    y_true_val = []
    y_pred_val = []

    device = net.device

    # Training loop
    for epoch in range(epochs):
        net.train() 
        running_loss = 0.0
        running_r2 = 0.0
        running_pearson = 0.0
        for i, data in enumerate(train_loader):
            if data is None:
                continue

            inputs, labels = data

            # loads tensor into GPU
            inputs = inputs.to(device)                                              
            labels = labels.to(device)

            if torch.isnan(inputs).any() or torch.isnan(labels).any():
                print("Found NaN in inputs or labels!")
                continue

            optimizer.zero_grad()   # Zero the parameter gradients

            # Forward pass
            outputs = net(inputs)   # Get the network outputs

            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward() # Backward pass (compute gradients)
            optimizer.step()  

            # Rename for statistics
            y_pred = outputs
            y_true = labels

            y_true_train.append(y_true.to("cpu"))
            y_pred_train.append(y_pred.to("cpu"))

            r2 = r2_score(y_pred, y_true)

            pcc = PearsonCorrCoef().to(device)
            pearson = pcc(y_pred, y_true)
            
            running_loss += loss.item()
            running_r2 += r2.item()
            running_pearson += pearson.item()

            if i % 100 == 0 :
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}, R2: {r2}, Pearson: {pearson}")

        # Loss calculation and storing for plotting
        average_loss = running_loss / len(train_loader)
        loss_values.append(average_loss)

        # R2 calculation and storing for plotting
        average_r2 = running_r2 / len(train_loader)
        r2_values.append(average_r2)

        # Pearson calculation and storing for plotting
        average_pearson = running_pearson / len(train_loader)
        pearson_values.append(average_pearson)

        # Validation loop
        net.eval()
        val_loss = 0.0
        val_r2 = 0.0
        val_pearson = 0.0
        with torch.no_grad():
            for val_data in val_loader:
                if val_data is None:
                    continue

                val_inputs, val_labels = val_data
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                val_outputs = net(val_inputs)

                y_pred_val.append(val_outputs.to("cpu"))
                y_true_val.append(val_labels.to("cpu"))

                val_loss += criterion(val_outputs, val_labels.view(-1, 1)).item()
                val_r2 += r2_score(val_outputs, val_labels).item()
                val_pearson += pcc(val_outputs, val_labels).item()

        # Average validation loss and R2 score
        average_val_loss = val_loss / len(val_loader)
        loss_values_val.append(average_val_loss)

        average_val_r2 = val_r2 / len(val_loader)
        r2_values_val.append(average_val_r2)

        average_val_pearson = val_pearson / len(val_loader)
        pearson_values_val.append(average_val_pearson)

        # # Save the model if the current validation R² is better than the best one saved
        print(f"Epoch {epoch+1}, Training Loss: {average_loss}, Validation Loss: {average_val_loss}, Training R2: {average_r2}, Validation R2: {average_val_r2}, Training Pearson: {average_pearson}, Validation Pearson: {average_val_pearson}")
        if not weight_saver.should_continue_train(average_val_r2):
            print(f"Finished training at epoch {epoch+1}. Best R2 at epoch {epoch+1-weight_saver.patience}")
            break

    # save metrics:
    torch.save({
        "train_losses": loss_values,
        "train_r2": r2_values,
        "train_pearson": pearson_values,
        "train_y": y_true_train,
        "train_pred": y_pred_train,

        "val_losses": loss_values_val,
        "val_r2": r2_values_val,
        "val_pearson": pearson_values_val,
        "val_y": y_true_val,
        "val_pred": y_pred_val
    }, metrics_path)

if __name__ == "__main__":
    # to get the number of parameters:
    from hyperparameters import hyperparameters
    from torchsummary import summary
    import torch

    n_features = 5316
    net = Network(
        hidden_dim1=hyperparameters["hidden_dim1"], 
        hidden_dim2=hyperparameters["hidden_dim2"], 
        hidden_dim3=hyperparameters["hidden_dim3"], 
        dropout1=hyperparameters["dropout1"], 
        dropout2=hyperparameters["dropout2"],
        with_gate=True,
        input_size=n_features
    ).cpu()
    input = torch.randn(1, n_features)
    summary(net, input)