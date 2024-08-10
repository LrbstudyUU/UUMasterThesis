#Helper script to test Arousal and valence models on input sentences
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

model_path = '~/ArValWordSent(Original)/final_model'
tokenizer_path = '~/ArValWordSent(Original)/final_model'

tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
model = RobertaForSequenceClassification.from_pretrained(model_path, use_safetensors=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)
# model.eval()

def predict_arousal_valence(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    predictions = outputs.logits.detach().numpy()[0]
    return {"arousal": predictions[0], "valence": predictions[1]}

print(predict_arousal_valence('Het verlies van mijn huisdier heeft me diep bedroefd.'))