from stock_dataset import StockDataset
from model_arch import MultitaskModel

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score, root_mean_squared_error

import matplotlib.pylab as plt

# Feat Eng Helper Functions
def norm_col_nme(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = ["_".join(c.lower().split(" ")) for c in df.columns]
    return df

def get_list_of_tweets(df: pd.DataFrame, stock_nme: str, date_val: str) -> list[str]:
    return df[(df.stock_name == stock_nme) & (df.date == pd.to_datetime(date_val))]["tweet"].iloc[0]

def get_sentiments_for_row(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, txt: list[str]) -> list[int]:
    mdl_inputs = tokenizer(txt, padding="longest", return_tensors="pt")
    output2 = model(**mdl_inputs)
    proba = torch.nn.functional.softmax(output2.logits, dim=-1)
    pos, neu, neg = torch.mean(proba, dim=0).tolist()
    
    return pd.Series({"pos": pos, "neu": neu, "neg": neg})

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for MTL DNN"""
    cols = [c for c in df.columns if "dim" in c] + ["open", "close"]
    df2 = df.drop(columns=["if_higher", "stock_name", "tweet", "company_name","date",
                           'open_target', 'high_target', 'low_target', 'close_target',
                           'adj_close_target', 'volume_target'])
    X = torch.tensor(df2[cols].values, dtype=torch.float32)
    cls_target_if_higher = torch.tensor(df.if_higher.values, dtype=torch.float32)
    reg_target_close = torch.tensor(df.close_target.values, dtype=torch.float32)

    return StockDataset(X, reg_target_close, cls_target_if_higher)

# Model Training Function
def train_model(model, dataloader, criterion_reg, criterion_class, optimizer, num_epochs, epoch):
    train_losses = []
    total_loss = 0.0

    model.train()
    for batch in dataloader:
        inputs, reg_targets, class_targets = batch
            
        optimizer.zero_grad()

        # Forward pass
        reg_output, class_output = model(inputs)

        # Compute loss for both tasks
        loss_reg = criterion_reg(reg_output, reg_targets.reshape(-1, 1))
        loss_class = criterion_class(class_output, class_targets.reshape(-1, 1))
        loss = loss_reg + loss_class

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Average loss per batch
    avg_loss = total_loss / len(dataloader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return train_losses

# Evaluation Function
def evaluate_model(model, dataloader, criterion_reg, criterion_class):

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs, reg_targets, class_targets = batch
            
            # Forward pass
            reg_output, class_output = model(inputs)

            # Compute loss for both tasks
            loss_reg = criterion_reg(reg_output, reg_targets.reshape(-1, 1))
            loss_class = criterion_class(class_output, class_targets.reshape(-1, 1))
            loss = loss_reg + loss_class

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate_test_data(dataloader: DataLoader, model: MultitaskModel):
    reg_ouputs, cls_output = [], []
    reg_true, cls_true = [], []
    for batch in dataloader:
        inputs, reg_targets, class_targets = batch
        reg_true.extend(reg_targets)
        cls_true.extend(class_targets)
        # Forward pass
        with torch.no_grad():
            reg_output, class_output = model(inputs)
            reg_ouputs.extend(reg_output)
            cls_output.extend(class_output)
    
    print("root_mean_squared_error for test", root_mean_squared_error(y_true=reg_true, y_pred=reg_ouputs))
    print("f1 score for test",f1_score(cls_true, [1 if i > 0.2 else 0 for i in cls_output]))

# Plot training and Validation Loss
def plot_losses(train_losses, val_losses):
    epochs = len(train_losses)
    plt.plot(range(epochs), train_losses, label="Training Loss")
    plt.plot(range(epochs), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()