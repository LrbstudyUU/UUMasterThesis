#Test script for fine-tuning RobBERT Dutch model for specific emotions
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from transformers import Trainer, TrainingArguments, TrainerCallback
from torch import nn, cuda
import os
import matplotlib.pyplot as plt

os.environ["WANDB_DISABLED"] = "true"

# Set device
device = 'cuda' if cuda.is_available() else 'cpu'

# Load data
data_emotions = pd.read_excel('~/RobBERT/All_Primary_Emotions.xlsx', sheet_name="MeanEmotionPerWord")

# Filter and rename columns
emotion_columns = ['Average of met blijheid', 'Average of met kwaadheid', 'Average of met angst', 
                   'Average of met bedroefdheid', 'Average of met walging', 'Average of met verrassing']
new_data_emotions = data_emotions[['Row Labels'] + emotion_columns]

# Ensure that emotion columns are of type float and drop rows with NaNs
new_data_emotions[emotion_columns] = new_data_emotions[emotion_columns].apply(pd.to_numeric, errors='coerce')
new_data_emotions = new_data_emotions.dropna(subset=emotion_columns)
#print(new_data_emotions[emotion_columns].dtypes)  # Should all be float64
#print(new_data_emotions[emotion_columns].isna().sum())  # Check for NaNs, should be 0

# Split data
train_df_emotions, val_df_emotions = train_test_split(new_data_emotions, test_size=0.2, random_state=42)

# Load tokenizer
tokenizer_emotions = RobertaTokenizer.from_pretrained('DTAI-KULeuven/robbert-2023-dutch-base')

# Tokenize data
train_encodings_emotions = tokenizer_emotions(list(train_df_emotions['Row Labels']), truncation=True, padding=True)
val_encodings_emotions = tokenizer_emotions(list(val_df_emotions['Row Labels']), truncation=True, padding=True)

# Dataset class
class EmotionsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx].astype(np.float32))  # Ensure labels are float32
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_labels_emotions = train_df_emotions[emotion_columns].values
val_labels_emotions = val_df_emotions[emotion_columns].values
train_dataset_emotions = EmotionsDataset(train_encodings_emotions, train_labels_emotions)
val_dataset_emotions = EmotionsDataset(val_encodings_emotions, val_labels_emotions)

# Load model
model_emotions = RobertaModel.from_pretrained('DTAI-KULeuven/robbert-2023-dutch-base')

# Define a custom regression head for the model
class RobertaRegressionHead(nn.Module):
    def __init__(self):
        super(RobertaRegressionHead, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(768, 768)
        self.out_proj = nn.Linear(768, 6)  # 6 output values for the 6 emotions

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # Take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

# Attach the regression head to the model
class RobertaForEmotionRegression(nn.Module):
    def __init__(self, model):
        super(RobertaForEmotionRegression, self).__init__()
        self.model = model
        self.regression_head = RobertaRegressionHead()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.regression_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)
        return (loss, logits) if loss is not None else logits

model_emotions = RobertaForEmotionRegression(model_emotions).to(device)

# Training arguments
training_args_emotions = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',  # directory for storing logs
    logging_steps=10,      # log every 10 steps
)

# Callback to capture losses
class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        self.train_loss = []
        self.eval_loss = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.train_loss.append(logs['loss'])
        if 'eval_loss' in logs:
            self.eval_loss.append(logs['eval_loss'])

# Initialize the callback
loss_history = LossHistoryCallback()

# Trainer
trainer_emotions = Trainer(
    model=model_emotions,
    args=training_args_emotions,
    train_dataset=train_dataset_emotions,
    eval_dataset=val_dataset_emotions,
    callbacks=[loss_history],
)

# Train and evaluate
trainer_emotions.train()
trainer_emotions.evaluate()

# Save the model and tokenizer
model_save_path = '~/RobBERT/emotions/emotions_model'
tokenizer_save_path = '~/RobBERT/emotions/emotions_tokenizer'
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(tokenizer_save_path, exist_ok=True)
torch.save(model_emotions.state_dict(), os.path.join(model_save_path, 'pytorch_model.bin'))
tokenizer_emotions.save_pretrained(tokenizer_save_path)

# Save the config
config = RobertaConfig.from_pretrained('DTAI-KULeuven/robbert-2023-dutch-base')
config.num_labels = 6
config.save_pretrained(model_save_path)

# Plotting loss graphs
plt.figure(figsize=(10, 5))
plt.plot(loss_history.train_loss, label='Training Loss')
plt.plot(loss_history.eval_loss, label='Validation Loss')
plt.ylim(0, 1)  # Adjust y-axis limit
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Time (Adjusted)')
plt.show()
plt.savefig('~/RobBERT/emotions/emotionsTrain.png')