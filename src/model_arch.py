import torch.nn as nn


# Define the Multitask Model with 2 heads: Regression and Classification
class MultitaskModel(nn.Module):
    def __init__(self):
        super(MultitaskModel, self).__init__()
        
        # Shared layers (can be customized to your architecture)
        self.shared = nn.Sequential(
            nn.Linear(770, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Output a single value for regression
        )
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Output a single value for binary classification
            nn.Sigmoid(),  # Sigmoid to map to probabilities
        )

    def forward(self, x):
        shared_rep = self.shared(x)
        
        # Separate paths for tasks
        reg_output = self.regression_head(shared_rep)
        class_output = self.classification_head(shared_rep)
        
        return reg_output, class_output