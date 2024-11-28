import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# Tải mô hình và tokenizer
model = BertForTokenClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizerFast.from_pretrained("./saved_model")

# Chuẩn bị văn bản cần bóc tách
text = """
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tư do Hanh phúc
SOCIALIST REPUBLIC OF VIET NAM
Independence - Freedom Happiness
CĂN CƯỚC CÔNG DÂN
Citizen Identity Card
Số ID: 252458456854
Họ và tên / Full name: LÊ VĂN VIỆT
Ngày sinh / Date of birth: 28/01/1978
Giới tính / Sex: Nam
Quốc tịch / Nationality: Việt Nam
Quê quán / Place of origin: Tam Tiến, Núi Thành, Quảng Nam
Nơi thường trú / Place of residence: Thôn Quảng Bính, Cang nghiệt, Nghĩa Thắng, Đắk Nông
"""

# Tokenize văn bản
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

# Thực hiện dự đoán
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)

# Chuyển đổi nhãn từ ID sang tên nhãn
label_list = ["O", "ORGANIZATION", "ID_NUMBER", "FULL_NAME", "DATE_OF_BIRTH", "GENDER", "NATIONALITY", "PLACE_OF_ORIGIN", "PLACE_OF_RESIDENCE"]
id2label = {i: label for i, label in enumerate(label_list)}
predicted_labels = [id2label[id] for id in predictions[0].numpy()]

# In kết quả
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
for token, label in zip(tokens, predicted_labels):
    print(f"{token}: {label}")