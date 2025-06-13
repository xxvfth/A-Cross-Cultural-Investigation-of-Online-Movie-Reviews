import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os


model_base_path = 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_reviews(input_file, output_file):
    data = pd.read_csv(input_file)
    data = data.dropna()
    reviews = data["review"].tolist()


    tokenizer = AutoTokenizer.from_pretrained(model_base_path)

    predictions = {f"pred_{label}": [] for label in ["H", "E", "R", "A"]}

    for label in ["H", "E", "R", "A"]:
        print(f"Predicting for label: {label}")


        model_save_path = f"D:/BERTE/saved_model_{label}"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_save_path,
            num_labels=2,  
            problem_type="single_label_classification"
        )
        model.to(device)
        model.eval()

        label_predictions = []

        with torch.no_grad():
            batch_size = 32
            for i in range(0, len(reviews), batch_size):
                batch_reviews = reviews[i:i+batch_size]
                inputs = tokenizer(
                    batch_reviews,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                logits = outputs.logits  

        
                preds = torch.argmax(logits, dim=-1).cpu().tolist()

                label_predictions.extend(preds)

        predictions[f"pred_{label}"] = label_predictions

    for label in ["H", "E", "R", "A"]:
        data[f"pred_{label}"] = predictions[f"pred_{label}"]


    data.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    input_csv_file =  
    output_xlsx_file = 
    predict_reviews(input_csv_file, output_xlsx_file)
