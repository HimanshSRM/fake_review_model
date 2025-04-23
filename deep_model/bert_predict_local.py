import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Set model path
model_path = r"C:\Users\samra\Desktop\fake_review_app - Copy - Copy\deep_model"

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.to(device)
model.eval()  # Set to evaluation mode

def predict_fake_review(review_text):
    inputs = tokenizer(review_text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU

    with torch.no_grad():  # Disable gradient calculation
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    return "🔴 Fake Review" if prediction == 1 else "🟢 Real Review"

# Test Predictions
print(predict_fake_review("This product is amazing! Best purchase ever!"))
print(predict_fake_review("C"))
