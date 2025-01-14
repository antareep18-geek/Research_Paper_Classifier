#TASK 1

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm
import re
from nltk.corpus import stopwords
import nltk
import warnings
warnings.filterwarnings("ignore")

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Load datasets
df_papers = pd.read_csv("papers_texts.csv")
df_ref = pd.read_csv("ref_texts.csv")

# Labeling the reference dataset
df_ref.loc[df_ref["filename"].isin(["R001.pdf", "R002.pdf", "R003.pdf", "R004.pdf", "R005.pdf"]), "label"] = "non-pushable"
df_ref.loc[df_ref["filename"].isin(["R006.pdf", "R007.pdf", "R008.pdf", "R009.pdf", "R010.pdf", "R011.pdf", "R012.pdf", "R013.pdf", "R014.pdf", "R015.pdf"]), "label"] = "pushable"

# Conference assignment for the reference dataset
df_ref.loc[df_ref["filename"].isin(["R001.pdf", "R002.pdf", "R003.pdf", "R004.pdf", "R005.pdf"]), "conference"] = "none"
df_ref.loc[df_ref["filename"].isin(["R006.pdf", "R007.pdf"]), "conference"] = "cvpr"
df_ref.loc[df_ref["filename"].isin(["R008.pdf", "R009.pdf"]), "conference"] = "emnlp"
df_ref.loc[df_ref["filename"].isin(["R010.pdf", "R011.pdf"]), "conference"] = "kdd"
df_ref.loc[df_ref["filename"].isin(["R012.pdf", "R013.pdf"]), "conference"] = "neurips"
df_ref.loc[df_ref["filename"].isin(["R014.pdf", "R015.pdf"]), "conference"] = "tmlr"

df_papers = df_papers.drop("filename", axis=1)
df_ref = df_ref.drop("filename", axis=1)

# Label encoding for 'pushable' and 'non-pushable'
df_ref['label'] = df_ref['label'].map({"pushable": 1, "non-pushable": 0})

# Preprocess text (remove stopwords, punctuation, etc.)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df_papers['text'] = df_papers['text'].apply(preprocess_text)
df_ref['text'] = df_ref['text'].apply(preprocess_text)

# Upsample to balance classes
n_samples = 20
class_0 = df_ref[df_ref['label'] == 0]
class_1 = df_ref[df_ref['label'] == 1]
class_0_oversampled = resample(class_0, replace=True, n_samples=n_samples, random_state=42)
class_1_oversampled = resample(class_1, replace=True, n_samples=n_samples, random_state=42)
balanced_df = pd.concat([class_0_oversampled, class_1_oversampled])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
df_ref = balanced_df

# Prepare train and test data
X = df_ref['text']
y = df_ref['label']
X_train, X_test, y_train, y_test = train_test_split(X.to_list(),
                                                    y.to_list(),
                                                    test_size=0.2,
                                                    random_state=53)

# Load SciBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

# Tokenize the data with truncation and padding to max_length=512
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=512)

# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, chunks, labels):
        self.chunks = chunks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.chunks[idx], dtype=torch.long)
        attention_mask = (input_ids != 0).long()  # Create attention mask (1 for real tokens, 0 for padding)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

# Collate function to pad sequences dynamically during batching
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]

    # Pad input_ids and attention_mask to the max sequence length in the batch
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {'input_ids': padded_input_ids, 'attention_mask': padded_attention_mask, 'labels': torch.tensor(labels, dtype=torch.long)}

# Create the datasets
train_dataset = CustomDataset(train_encodings['input_ids'], y_train)
test_dataset = CustomDataset(test_encodings['input_ids'], y_test)

# Create data loaders with collate_fn for dynamic padding
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=5, collate_fn=collate_fn)

# Load the pre-trained SciBERT model
model = AutoModelForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=2)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-6)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

# Evaluation
model.eval()
preds = []
true_labels = []
with torch.no_grad():
    for batch in test_loader:
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        preds.extend(torch.argmax(outputs.logits, axis=1).cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

# Print classification report and confusion matrix
print(classification_report(true_labels, preds, target_names=["non-pushable", "pushable"]))
cm = confusion_matrix(true_labels, preds)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

# Predict labels for df_papers
papers_encodings = tokenizer(df_papers['text'].tolist(), truncation=True, padding=True, max_length=512)

class PapersDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

papers_dataset = PapersDataset(papers_encodings)
papers_loader = DataLoader(papers_dataset, batch_size=5)

model.eval()
predictions = []
with torch.no_grad():
    for batch in papers_loader:
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, axis=1)
        predictions.extend(preds.cpu().numpy())

df_papers['predicted_label'] = predictions
plt.show()

print(df_papers)

#TASK 2

import pandas as pd
import numpy as np
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm
import re
from nltk.corpus import stopwords
import nltk
import warnings
warnings.filterwarnings("ignore")

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Load datasets
df_ref = pd.read_csv("ref_texts.csv")

# Labeling the reference dataset
df_ref.loc[df_ref["filename"].isin(["R001.pdf", "R002.pdf", "R003.pdf", "R004.pdf", "R005.pdf"]), "label"] = "non-pushable"
df_ref.loc[df_ref["filename"].isin(["R006.pdf", "R007.pdf", "R008.pdf", "R009.pdf", "R010.pdf", "R011.pdf", "R012.pdf", "R013.pdf", "R014.pdf", "R015.pdf"]), "label"] = "pushable"

# Conference assignment for the reference dataset
df_ref.loc[df_ref["filename"].isin(["R001.pdf", "R002.pdf", "R003.pdf", "R004.pdf", "R005.pdf"]), "conference"] = "none"
df_ref.loc[df_ref["filename"].isin(["R006.pdf", "R007.pdf"]), "conference"] = "cvpr"
df_ref.loc[df_ref["filename"].isin(["R008.pdf", "R009.pdf"]), "conference"] = "emnlp"
df_ref.loc[df_ref["filename"].isin(["R010.pdf", "R011.pdf"]), "conference"] = "kdd"
df_ref.loc[df_ref["filename"].isin(["R012.pdf", "R013.pdf"]), "conference"] = "neurips"
df_ref.loc[df_ref["filename"].isin(["R014.pdf", "R015.pdf"]), "conference"] = "tmlr"

df_ref = df_ref.drop("filename", axis=1)

df_ref=df_ref[df_ref["conference"]!="none"]
# Label encoding for 'pushable' and 'non-pushable'
df_ref['conference'] = df_ref['conference'].map({"cvpr": 0,
                                                 "emnlp": 1,
                                                 "kdd": 2,
                                                 "neurips": 3,
                                                 "tmlr": 4})

# Preprocess text (remove stopwords, punctuation, etc.)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df_ref['text'] = df_ref['text'].apply(preprocess_text)

# Upsample
n_samples = 50
class_0 = df_ref[df_ref['conference'] == 0]
class_1 = df_ref[df_ref['conference'] == 1]
class_2 = df_ref[df_ref['conference'] == 2]
class_3 = df_ref[df_ref['conference'] == 3]
class_4 = df_ref[df_ref['conference'] == 4]
class_0_oversampled = resample(class_0, replace=True, n_samples=n_samples, random_state=42)
class_1_oversampled = resample(class_1, replace=True, n_samples=n_samples, random_state=42)
class_2_oversampled = resample(class_2, replace=True, n_samples=n_samples, random_state=42)
class_3_oversampled = resample(class_3, replace=True, n_samples=n_samples, random_state=42)
class_4_oversampled = resample(class_4, replace=True, n_samples=n_samples, random_state=42)
balanced_df = pd.concat([class_0_oversampled, class_1_oversampled, class_2_oversampled, class_3_oversampled, class_4_oversampled])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
df_ref = balanced_df

# Prepare train and test data
X = df_ref['text']
y = df_ref['conference']
X_train, X_test, y_train, y_test = train_test_split(X.to_list(),
                                                    y.to_list(),
                                                    test_size=0.2,
                                                    random_state=1000)

# Load SciBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

# Tokenize the data with truncation and padding to max_length=512
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=512)

# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, chunks, labels):
        self.chunks = chunks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.chunks[idx], dtype=torch.long)
        attention_mask = (input_ids != 0).long()  # Create attention mask (1 for real tokens, 0 for padding)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

# Collate function to pad sequences dynamically during batching
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]

    # Pad input_ids and attention_mask to the max sequence length in the batch
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {'input_ids': padded_input_ids, 'attention_mask': padded_attention_mask, 'labels': torch.tensor(labels, dtype=torch.long)}

# Create the datasets
train_dataset = CustomDataset(train_encodings['input_ids'], y_train)
test_dataset = CustomDataset(test_encodings['input_ids'], y_test)

# Create data loaders with collate_fn for dynamic padding
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=5, collate_fn=collate_fn)

# Load the pre-trained SciBERT model
model = AutoModelForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-6)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

# Evaluation
model.eval()
preds = []
true_labels = []
with torch.no_grad():
    for batch in test_loader:
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        preds.extend(torch.argmax(outputs.logits, axis=1).cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

# Print classification report and confusion matrix
print(classification_report(true_labels, preds))
cm = confusion_matrix(true_labels, preds)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

# Predict labels for df_papers
papers_encodings = tokenizer(df_papers['text'].tolist(), truncation=True, padding=True, max_length=512)

class PapersDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

papers_dataset = PapersDataset(papers_encodings)
papers_loader = DataLoader(papers_dataset, batch_size=5)

model.eval()
predictions = []
with torch.no_grad():
    for batch in papers_loader:
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, axis=1)
        predictions.extend(preds.cpu().numpy())


df_papers['predicted_conference'] = predictions
label_map = {0: "cvpr", 1: "emnlp", 2: "kdd", 3: "neurips", 4: "tmlr" }
df_papers['predicted_conference'] = df_papers['predicted_conference'].map(label_map)
df_papers.loc[df_papers["predicted_label"] == 0, "predicted_conference"] = "none"
plt.show()

df_papers

df_papers["predicted_conference"].value_counts()

import requests
import pandas as pd
import google.generativeai as genai
import time





#genai.configure(api_key="")
#Add your api_key above and uncomment the above the line
model = genai.GenerativeModel("gemini-1.5-flash")

# Define Gemini API endpoint and your API key
#GEMINI_API_URL = "https://us-central1-aiplatform.googleapis.com/v1"  # Replace with actual endpoint if different
#GEMINI_API_KEY = ""  # Replace with your API key and uncomment



# Function to generate rationale using Gemini API
def generate_rationale_gemini(text, conference):
    time.sleep(0)
    if conference.lower() == "none":
        return "No conference assigned, so no rationale is provided."

    prompt = f"""
    The following text has been categorized as suitable for the {conference.upper()} conference:
    Text: {text}
    Please provide a concise rationale (under 100 words) explaining why this text aligns with the focus of {conference.upper()}.
    """

    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.7
    }

    try:

        response = model.generate_content(prompt)
        #response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        #response.raise_for_status()
        #rationale = response.json()["choices"][0]["text"].strip()
        rationale=response.text.strip()
        return rationale
    except Exception as e:
        return f"Error: {str(e)}"

# Add a rationale column to the DataFrame
df_papers["Rationale"] = df_papers.apply(
    lambda row: generate_rationale_gemini(row["text"], row["predicted_conference"]), axis=1)

df_papers.drop(columns=['text'], inplace=True)
df_papers.rename(columns={'predicted_conference': 'Conference'}, inplace=True)
df_papers.rename(columns={'predicted_label': 'Publishable'}, inplace=True)

serial_numbers = [f'P{i:03}' for i in range(1, len(df_papers) + 1)]

# Add the serial numbers as a new column
df_papers.insert(0, 'Paper ID', serial_numbers)
# Save the updated DataFrame
df_papers.to_csv("updated_conferences_with_rationale_gemini.csv", index=False)

print("Updated DataFrame saved as 'updated_conferences_with_rationale_gemini.csv'")

import matplotlib.pyplot as plt
# Count the occurrences of each conference
conference_counts = df_papers["Conference"].value_counts()
# Create the horizontal bar plot
plt.figure(figsize=(10, 6))
plt.barh(conference_counts.index, conference_counts.values, color='skyblue')
# Add labels and title
plt.xlabel("Count")
plt.ylabel("Conference")
plt.title("Counts of Predicted Conferences")
# Show the plot
plt.show()
