import json
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset

# Đọc dữ liệu từ file JSONL
def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# Chuyển đổi nhãn sang token-level
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["text"], truncation=True, return_offsets_mapping=True)
    labels = [0] * len(tokenized_inputs["input_ids"])  # Mặc định là "O"

    for entity in example["labels"]:
        entity_start = entity["start"]
        entity_end = entity["end"]
        entity_label = entity["label"]

        for idx, (start, end) in enumerate(tokenized_inputs["offset_mapping"]):
            if start >= entity_start and end <= entity_end:
                labels[idx] = label2id[entity_label]

    tokenized_inputs["labels"] = labels
    tokenized_inputs.pop("offset_mapping")  # Xóa thông tin offset để giảm tải
    return tokenized_inputs

# Đọc dữ liệu từ file
file_path = "data_train.jsonl"
data = read_jsonl(file_path)

# Danh sách các nhãn
label_list = ["O", "ORGANIZATION", "ID_NUMBER", "FULL_NAME", "DATE_OF_BIRTH", "GENDER", "NATIONALITY", "PLACE_OF_ORIGIN", "PLACE_OF_RESIDENCE"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# Chuẩn bị dữ liệu cho Dataset
texts = [item["text"] for item in data]
labels = [item["labels"] for item in data]
dataset = Dataset.from_dict({"text": texts, "labels": labels})
tokenized_dataset = dataset.map(tokenize_and_align_labels)

# Tải tokenizer và mô hình
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(label_list), id2label=id2label, label2id=label2id)

# Thiết lập huấn luyện
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

# Huấn luyện mô hình
trainer.train()
