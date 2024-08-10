#TEst script for fine-tuning RobBERT Dutch model

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
# from torch import cuda
# import os
# os.environ["WANDB_DISABLED"] = "true"

# # Set device
# device = 'cuda' if cuda.is_available() else 'cpu'

# # Load data
# data_arousal = pd.read_excel('/RobBERT/All_Arousal.xlsx', sheet_name="MeanArousalPerWord")

# # Filter and rename columns
# new_data_arousal = data_arousal[['Word', 'Arousal']]
# new_data_arousal = new_data_arousal.rename(columns={'Arousal': 'label'})

# # Split data
# train_df_arousal, val_df_arousal = train_test_split(new_data_arousal, test_size=0.2, random_state=42)

# # Load tokenizer
# tokenizer_arousal = RobertaTokenizer.from_pretrained('DTAI-KULeuven/robbert-2023-dutch-base')

# # Tokenize data
# train_encodings_arousal = tokenizer_arousal(list(train_df_arousal['Word']), truncation=True, padding=True)
# val_encodings_arousal = tokenizer_arousal(list(val_df_arousal['Word']), truncation=True, padding=True)

# # Dataset class
# class ArousalDataset(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# # Create datasets
# train_dataset_arousal = ArousalDataset(train_encodings_arousal, list(train_df_arousal['label']))
# val_dataset_arousal = ArousalDataset(val_encodings_arousal, list(val_df_arousal['label']))

# # Load model
# model_arousal = RobertaForSequenceClassification.from_pretrained('DTAI-KULeuven/robbert-2023-dutch-base', num_labels=1)
# model_arousal.to(device)

# # Training arguments
# training_args_arousal = TrainingArguments(
#     output_dir='./results',
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=5,
#     weight_decay=0.01,
# )

# # Trainer
# trainer_arousal = Trainer(
#     model=model_arousal,
#     args=training_args_arousal,
#     train_dataset=train_dataset_arousal,
#     eval_dataset=val_dataset_arousal,
# )

# # Train and evaluate
# trainer_arousal.train()
# trainer_arousal.evaluate()

# model_arousal.save_pretrained('~/RobBERT/arousal/arousal_model')
# tokenizer_arousal.save_pretrained('~/RobBERT/arousal/arousal_tokenizer')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from torch import cuda
import os
import matplotlib.pyplot as plt

os.environ["WANDB_DISABLED"] = "true"

# Set device
device = 'cuda' if cuda.is_available() else 'cpu'

# Load data
data = pd.read_excel('~/RobBERT/All_Arousal.xlsx', sheet_name="MeanArousalPerWord")
#data = pd.read_excel('~/RobBERT/All_Valence.xlsx', sheet_name="Means")

# Filter and rename columns
new_data = data[['Word', 'Arousal']]
new_data = new_data.rename(columns={'Arousal': 'label'})
# new_data = data[['Word', 'Valence']]
# new_data = new_data.rename(columns={'Valence': 'label'})

min_val = new_data['label'].min()
max_val = new_data['label'].max()
new_data['label'] = (new_data['label'] - min_val) / (max_val - min_val)

# Split data
train_df_arousal, val_df_arousal = train_test_split(new_data, test_size=0.2, random_state=42)

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained('DTAI-KULeuven/robbert-2023-dutch-base')

# Tokenize data
train_encodings_arousal = tokenizer(list(train_df_arousal['Word']), truncation=True, padding=True)
val_encodings_arousal = tokenizer(list(val_df_arousal['Word']), truncation=True, padding=True)

# Dataset class
class ArousalorValenceDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = ArousalorValenceDataset(train_encodings_arousal, list(train_df_arousal['label']))
val_dataset = ArousalorValenceDataset(val_encodings_arousal, list(val_df_arousal['label']))

# Load model
model = RobertaForSequenceClassification.from_pretrained('DTAI-KULeuven/robbert-2023-dutch-base', num_labels=1)
model.to(device)

# Training arguments
training_args = TrainingArguments(
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
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[loss_history],
)

# Train and evaluate
trainer.train()
trainer.evaluate()

model.save_pretrained('~/RobBERT/arousal/arousal_model')
tokenizer.save_pretrained('~/RobBERT/arousal/arousal_tokenizer')
# model.save_pretrained('~/RobBERT/valence/valence_model')
# tokenizer_arousal.save_pretrained('~/RobBERT/valence/valence_tokenizer')

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
plt.savefig('~/RobBERT/arousal/arousalTrain.png')