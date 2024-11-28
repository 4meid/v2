import json
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import Dataset

# Định nghĩa tokenizer trước
tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

# Đọc dữ liệu từ file JSONL
def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# Chuyển đổi nhãn sang token-level
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128, return_offsets_mapping=True)
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

# Tải mô hình
model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(label_list))

# Thiết lập huấn luyện
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Sử dụng DataCollatorForTokenClassification để đảm bảo padding
data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Huấn luyện mô hình
trainer.train()

# Lưu mô hình và tokenizer
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

print("Training and saving complete.")