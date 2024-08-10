#Test fine-tuning script only training robBERT with EmotioNL
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from typing import Dict
import os
import shutil
from scipy.stats import pearsonr
os.environ["WANDB_DISABLED"] = "true"

sentences_df = pd.read_csv('~/corpus_fulltext_captions.txt', sep='\t') #EmotioNL dataset
sentences_df = sentences_df[['ID', 'Valence', 'Arousal']]
sentences_df = sentences_df.rename(columns={'ID': 'Text'})

tweets_df = pd.read_csv('~/corpus_fulltext_tweets.txt', sep='\t') #EmotioNL dataset
tweets_df = tweets_df[['Text', 'Valence', 'Arousal']]

#Combine EmotioNL captions and tweets, then shuffle to create single larger dataset
combined_sentences_df = pd.concat([sentences_df, tweets_df], ignore_index=True)
combined_sentences_df = combined_sentences_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_cv_df, final_val_df = train_test_split(combined_sentences_df, test_size=0.2, random_state=42) #Split combined EmotioNL dataset into train and test

class ArousalValenceDataset(Dataset):
    def __init__(self, texts, arousal_labels, valence_labels, tokenizer, max_length=128):
        self.texts = texts
        self.arousal_labels = arousal_labels
        self.valence_labels = valence_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        arousal = self.arousal_labels[idx]
        valence = self.valence_labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor([arousal, valence], dtype=torch.float)
        }

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_losses = []
        self.eval_losses = []

    def log(self, logs: Dict[str, float]) -> None: #Logging training and eval losses for plotting
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
        
        super().log(logs)
    
def plot_losses(train_losses=None, eval_losses=None, trainerobj=None, fold=None):
    plt.figure(figsize=(10, 6))
    if trainerobj:
        plt.plot(trainerobj.train_losses, label='Training Loss')
        plt.plot(trainerobj.eval_losses, label='Validation Loss')
        # if trainerobj.eval_losses:
        #     #eval_steps = len(trainerobj.train_losses) // len(trainerobj.eval_losses)
        #     #plt.plot(range(0, len(trainerobj.train_losses), eval_steps), trainerobj.eval_losses, label='Validation Loss')
        # else:
        #     print("No evaluation losses to plot.")
    else:
        plt.plot(train_losses, label='Training Loss')
        plt.plot(eval_losses, label='Validation Loss')
    plt.title(f'Training and Validation Losses {"- Fold " + str(fold) if fold else ""}')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'~/ArValWordSent/plots/loss_plot{"_fold_" + str(fold) if fold else ""}.png')
    plt.close()

def bin_values(values, num_bins=3):
    return pd.cut(values, bins=num_bins, labels=False)

tokenizer = RobertaTokenizer.from_pretrained("DTAI-KULeuven/robbert-2023-dutch-base")

print("Fine-tuning on sentences:")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []
all_fold_losses = defaultdict(list)

best_model = None
best_val_loss = float('inf')

for fold, (train_idx, val_idx) in enumerate(kf.split(train_cv_df)):
    print(f"Fold {fold + 1}")
    
    train_data = train_cv_df.iloc[train_idx]
    val_data = train_cv_df.iloc[val_idx]

    train_dataset = ArousalValenceDataset(train_data['Text'].tolist(), train_data['Arousal'].tolist(), train_data['Valence'].tolist(), tokenizer)
    val_dataset = ArousalValenceDataset(val_data['Text'].tolist(), val_data['Arousal'].tolist(), val_data['Valence'].tolist(), tokenizer)

    model = RobertaForSequenceClassification.from_pretrained(
        "DTAI-KULeuven/robbert-2023-dutch-base",
        num_labels=2,
        problem_type="regression"
    )

    training_args = TrainingArguments(
        output_dir=f'~/ArValWordSent/Sent_results/results_fold_{fold}',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'~/ArValWordSent/Sent_logs/logs_fold_{fold}',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=10,
        #save_steps=100,
        save_strategy='no',
        load_best_model_at_end=False,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    plot_losses(trainerobj=trainer, fold=fold+1)
    all_fold_losses['train'].extend(trainer.train_losses)
    all_fold_losses['eval'].extend(trainer.eval_losses)
    
    #Evaluate on validation set
    val_results = trainer.evaluate()
    cv_results.append(val_results)
    
    if val_results['eval_loss'] < best_val_loss:
        best_val_loss = val_results['eval_loss']
        best_model = model
        best_model.save_pretrained(f"~/ArValWordSent/best_model_fold_{fold}")


plot_losses(train_losses=all_fold_losses['train'], eval_losses=all_fold_losses['eval'])

avg_cv_loss = np.mean([result['eval_loss'] for result in cv_results])
print(f"Average Cross-Validation Loss: {avg_cv_loss}")

print("Final evaluation on held-out validation set:")
final_val_dataset = ArousalValenceDataset(final_val_df['Text'].tolist(), final_val_df['Arousal'].tolist(), final_val_df['Valence'].tolist(), tokenizer)

final_trainer = Trainer(
    model=best_model,
    args=training_args,
    eval_dataset=final_val_dataset,
)

# Predict on final validation dataset
predictions = final_trainer.predict(final_val_dataset)
predicted_labels = predictions.predictions

true_labels = np.column_stack((final_val_df['Arousal'], final_val_df['Valence']))

mse = mean_squared_error(true_labels, predicted_labels)
rmse = np.sqrt(mse)
mae = mean_absolute_error(true_labels, predicted_labels)
r2 = r2_score(true_labels, predicted_labels)
accuracy = np.mean(np.abs(true_labels - predicted_labels) < 0.1)

#For calculating f1 score
true_arousal_binned = bin_values(true_labels[:, 0])
pred_arousal_binned = bin_values(predicted_labels[:, 0])
f1_arousal = f1_score(true_arousal_binned, pred_arousal_binned, average='weighted')

true_valence_binned = bin_values(true_labels[:, 1])
pred_valence_binned = bin_values(predicted_labels[:, 1])
f1_valence = f1_score(true_valence_binned, pred_valence_binned, average='weighted')

#Pearson r
pearson_arousal, _ = pearsonr(true_labels[:, 0], predicted_labels[:, 0])
pearson_valence, _ = pearsonr(true_labels[:, 1], predicted_labels[:, 1])

metrics = {
    "Mean Squared Error": mse,
    "Root Mean Squared Error": rmse,
    "Mean Absolute Error": mae,
    "R-squared": r2,
    "Accuracy (within 0.1)": accuracy,
    "F1 Score (Arousal)": f1_arousal,
    "F1 Score (Valence)": f1_valence,
    "Pearson's r (Arousal)": pearson_arousal,
    "Pearson's r (Valence)": pearson_valence
}

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")
print(f"Accuracy (within 0.1): {accuracy}")
print(f"F1 Score (Arousal): {f1_arousal}")
print(f"F1 Score (Valence): {f1_valence}")
print(f"Pearson's r (Arousal): {pearson_arousal}")
print(f"Pearson's r (Valence): {pearson_valence}")

best_model.save_pretrained("~/ArValWordSent/final_model")
tokenizer.save_pretrained("~/ArValWordSent/final_model")
with open('~/ArValWordSent/final_metrics.txt', 'w') as f:
    json.dump(metrics, f, indent=4)