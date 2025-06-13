
import re
import os
import torch
from torch.optim import Adam
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler  
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import BCEWithLogitsLoss 


model_path = 
tokenizer = AutoTokenizer.from_pretrained(model_path)


def preprocess_text(text):
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)  
    return text  


def preprocess_data(file_path: str):
    data = pd.read_csv(file_path)
    data = data.dropna()  
    data[["H", "E", "R", "A"]] = data[["H", "E", "R", "A"]].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    data["review"] = data["review"].apply(preprocess_text)  
    data[["H", "E", "R", "A"]] = data[["H", "E", "R", "A"]].astype(float) 
    return data


class MyDataset(Dataset):
    def __init__(self, data, label_column):
        self.data = data
        self.label_column = label_column

    def __getitem__(self, index):

        review = self.data.iloc[index]["review"]
        label = self.data.iloc[index][self.label_column] 
        label = torch.tensor(label, dtype=torch.float)  
        return review, label

    def __len__(self):
        return len(self.data)


def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = torch.stack(labels)
    return inputs


def load_and_split_data(file_path: str, label_column: str):
    data = preprocess_data(file_path)

    dataset = MyDataset(data, label_column=label_column)

    total_size = len(dataset)
    train_size = int(0.80 * total_size)
    valid_size = int(0.10 * total_size)
    test_size = total_size - train_size - valid_size
    train_data, valid_data, test_data = random_split(dataset, [train_size, valid_size, test_size])
    return DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_func), \
           DataLoader(valid_data, batch_size=8, shuffle=False, collate_fn=collate_func), \
           DataLoader(test_data, batch_size=8, shuffle=False, collate_fn=collate_func)


def evaluate(model, validloader):
    model.eval()
    correct_pred, total_pred = 0, 0
    with torch.no_grad():
        for batch in validloader:

            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits  

            pred = (torch.sigmoid(logits) > 0.5).long() 

            labels = batch["labels"].long().view(-1, 1)  

            batch_correct = (pred == labels).sum().item()
            batch_total = labels.size(0)
            correct_pred += batch_correct
            total_pred += batch_total

    accuracy = correct_pred / total_pred if total_pred > 0 else 0
    return accuracy




def train(model, trainloader, validloader, label, epoch=5):
    optimizer = Adam(model.parameters(), lr=2e-5)
    scaler = GradScaler()
    criterion = BCEWithLogitsLoss()
    for ep in range(epoch):
        model.train()
        for i, batch in enumerate(trainloader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}

            labels = batch["labels"].float().view(-1, 1) 
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(**batch)
                logits = outputs.logits  # 形状：(batch_size, 1)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        acc = evaluate(model, validloader)
        print(f"Epoch {ep + 1}/{epoch}, Accuracy: {acc:.4f}")

    model_save_path = f"saved_model_{label}"
    model.save_pretrained(model_save_path)
    print(f"Model for label {label} saved to {model_save_path}")


def test_and_save_results(model, testloader, label, tokenizer, output_dir="predictions"):
    model.eval()
    results = []
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for batch in testloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            pred = torch.sigmoid(output.logits) > 0.5
            for i in range(pred.size(0)):
                review = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
                predicted_label = int(pred[i].item())
                original_label = int(batch["labels"][i].item())
                results.append([review, original_label, predicted_label])

    output_file = os.path.join(output_dir, f"predictions_{label}.csv")
    df = pd.DataFrame(results, columns=["review", "original_label", f"predicted_{label}"])
    df.to_csv(output_file, index=False)
    print(f"Results for label {label} saved to {output_file}")


if __name__ == "__main__":
    for label in ["H", "E", "R", "A"]:
        print(f"Training model for label: {label}")

        trainloader, validloader, testloader = load_and_split_data("zn_douban_700_data.csv", label_column=label)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
            problem_type="binary_classification"
        )
        if torch.cuda.is_available():
            model = model.cuda()

        train(model, trainloader, validloader, label)

        acc = evaluate(model, testloader)
        print(f"Final accuracy for label {label}: {acc}")

        test_and_save_results(model, testloader, label, tokenizer)
