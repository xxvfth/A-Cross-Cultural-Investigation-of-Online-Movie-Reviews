import re
import os
import torch
from torch.optim import Adam
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torch import autocast
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification


model_path = 
tokenizer = AutoTokenizer.from_pretrained(model_path)


def preprocess_text(text):
    text = text.lower() 
    text = re.sub(r'[^\w\s]', '', text)  
    return text  

def preprocess_data(file_path: str):
    data = pd.read_csv(file_path)
    data = data.dropna()  
  
    data[["H", "E", "R", "A"]] = data[["H", "E", "R", "A"]].astype(int)
    data["review"] = data["review"].apply(preprocess_text)  
    return data


class MyDataset(Dataset):
    def __init__(self, data, label_column):
        self.data = data.reset_index(drop=True)
        self.label_column = label_column

    def __getitem__(self, index):
  
        review = self.data.loc[index, "review"]
        label = self.data.loc[index, self.label_column]
        label = torch.tensor(label, dtype=torch.long) 
        return review, label

    def __len__(self):
        return len(self.data)


def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=512, padding="max_length",
                       truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels, dtype=torch.long)  
    return inputs


def load_and_split_data(file_path: str, label_column: str):
    data = preprocess_data(file_path)

    dataset = MyDataset(data, label_column=label_column)

    total_size = len(dataset)
    train_size = int(0.80 * total_size)
    valid_size = int(0.10 * total_size)
    test_size = total_size - train_size - valid_size
    train_data, valid_data, test_data = random_split(
        dataset, [train_size, valid_size, test_size])
    return DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_func), \
           DataLoader(valid_data, batch_size=8, shuffle=False, collate_fn=collate_func), \
           DataLoader(test_data, batch_size=8, shuffle=False, collate_fn=collate_func)


def evaluate(model, dataloader):
    model.eval()
    correct_pred, total_pred = 0, 0
    with torch.no_grad():
        for batch in dataloader:
          
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
          
            outputs = model(**batch)
            logits = outputs.logits  # 形状：(batch_size, num_labels)
        
            preds = torch.argmax(logits, dim=-1)  # 形状：(batch_size,)
    
            labels = batch["labels"]  # 形状：(batch_size,)

            batch_correct = (preds == labels).sum().item()
            batch_total = labels.size(0)
            correct_pred += batch_correct
            total_pred += batch_total

    accuracy = correct_pred / total_pred if total_pred > 0 else 0
    return accuracy


def train(model, trainloader, validloader, label, epoch=10):
    optimizer = Adam(model.parameters(), lr=2e-5)
    scaler = GradScaler()
    for ep in range(epoch):
        model.train()
        for i, batch in enumerate(trainloader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
     
        acc = evaluate(model, validloader)
        print(f"Epoch {ep + 1}/{epoch}, Accuracy: {acc:.4f}")

    model_save_path = f"D:/BERTE/saved_model_{label}"
    model.save_pretrained(model_save_path)
    print(f"Model for label {label} saved to {model_save_path}")


if __name__ == "__main__":

    os.makedirs("D:/BERTE", exist_ok=True)

    for label in ["H", "E", "R", "A"]:
        print(f"Training model for label: {label}")
      
        trainloader, validloader, testloader = load_and_split_data("en_IMDb_600_data.csv", label_column=label)
  
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,  
            problem_type="single_label_classification" 
        )
        if torch.cuda.is_available():
            model = model.cuda()
  
        train(model, trainloader, validloader, label)

        acc = evaluate(model, testloader)
        print(f"Final accuracy for label {label}: {acc:.4f}")
