from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mychen76/mistral7b_ocr_to_json_v1")
model = AutoModelForCausalLM.from_pretrained("mychen76/mistral7b_ocr_to_json_v1")

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

prompt = f"### Instruction: Parse the following OCR text into a structured JSON object.\n### Input: {text}\n### Output:"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result_text)