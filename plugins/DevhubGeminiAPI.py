import json
import requests
import sys
import os
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plugins.DevhubConfigManager import ConfigManager
from typing import List, Dict
def clean_json_string(raw_text: str) -> str:
    """
    Loại bỏ các phần mở đầu hoặc ký tự không hợp lệ như ```json.
    Args:
        raw_text (str): Chuỗi văn bản chứa JSON
    Returns:
        str: Chuỗi JSON hợp lệ
    """
    # Loại bỏ phần mở đầu không hợp lệ
    if raw_text.startswith("```json"):
        raw_text = raw_text[len("```json"):].strip()
    if raw_text.endswith("```"):
        raw_text = raw_text[:-len("```")].strip()

    return raw_text



def process_and_save_results(file_path: str, result: dict):
    """
    Process the API response and save new training data to the file.

    Args:
        file_path (str): Path to the file where data will be saved.
        result (dict): API response containing candidates for training data.
    """
    existing_data = []

    # Đọc dữ liệu cũ từ file nếu file tồn tại
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = [json.loads(line) for line in f]

    # Lọc dữ liệu để tránh trùng lặp
    existing_texts = {item['text'] for item in existing_data}
    new_data = []

    for candidate in result.get('candidates', []):
        raw_text = candidate['content']['parts'][0]['text']
        clean_text = clean_json_string(raw_text)
        try:
            # Parse JSON từ kết quả trả về
            parsed_content = json.loads(clean_text)
            if isinstance(parsed_content, list):  # Nếu kết quả là danh sách
                for item in parsed_content:
                    if item['text'] not in existing_texts:
                        new_data.append(item)
            elif isinstance(parsed_content, dict):  # Nếu kết quả là đối tượng
                if parsed_content['text'] not in existing_texts:
                    new_data.append(parsed_content)
        except json.JSONDecodeError as e:
            print(f"Lỗi khi parse JSON: {e}\nDữ liệu lỗi: {raw_text}")

    # Lưu dữ liệu mới vào file
    if new_data:
        with open(file_path, 'a', encoding='utf-8') as f:
            for item in new_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        print(f"Đã thêm {len(new_data)} bản ghi mới vào file '{file_path}'.")
    else:
        print("Không có dữ liệu mới để thêm.")


def call_generative_language_api(api_key: str, prompt: str) -> dict:
    """
    Gọi API ngôn ngữ sinh (Generative Language Model).

    Args:
        api_key (str): API key cho Google API
        prompt (str): Nội dung yêu cầu để tạo dữ liệu

    Returns:
        dict: Kết quả phản hồi từ API
    """
    import requests  # Import requests tại đây để giữ mã gọn

    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent'
    request_data = {
        'contents': [
            {
                'parts': [
                    {
                        'text': prompt
                    }
                ]
            }
        ]
    }

    headers = {
        'Content-Type': 'application/json',
    }

    try:
        response = requests.post(url, headers=headers, params={'key': api_key}, data=json.dumps(request_data))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return {
            'error': response.status_code,
            'message': f'HTTP error occurred: {http_err}'
        }
    except Exception as err:
        return {
            'error': 'Exception',
            'message': f'Other error occurred: {err}'
        }


def call_gemini(question: str) -> dict:
    """
    Gọi API Gemini để tạo nội dung dựa trên yêu cầu.

    Args:
        question (str): Nội dung cần tạo dữ liệu.

    Returns:
        dict: Kết quả phản hồi từ API
    """
    gemini_api_key = ConfigManager().get_api_key('API_KEY_GEMINI')
    return call_generative_language_api(gemini_api_key, question)


if __name__ == "__main__":
    # Prompt và file lưu dữ liệu
    prompt_text = (
        "tạo thêm 5 data training kết quả text ocr,tạo nhiều trường hợp sai chính tả,dữ liệu lộn xộn, sai ngày tháng, tên hoa thường, "
        "train trong phạm vi để nhận được id căn cước là 12 số, quốc tịch: việt nam,giới tính, họ tên, "
        "Trả về dữ liệu dạng JSON sau: "
        "{"
        "   \"text\": \"CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập - Tự do Hạnh phúc SOCIALIST REPUBLIC OF VIET NAM Independence - Freedom Happiness "
        "   CĂN CƯỚC CÔNG DÂN Citizen Identity Card Số ID: 252458456854 "
        "   Họ và tên / Full name: LÊ VĂN VIỆT "
        "   Ngày sinh / Date of birth: 28/01/1978 "
        "   Giới tính / Sex: Nam "
        "   Quốc tịch / Nationality: Việt Nam "
        "   Quê quán / Place of origin: Tam Tiến, Núi Thành, Quảng Nam "
        "   Nơi thường trú / Place of residence: Thôn Quảng Bính, Cang nghiệt, Nghĩa Thắng, Đắk Nông\","
        "   \"labels\": ["
        "       {\"start\": 0, \"end\": 10, \"label\": \"ORGANIZATION\"},"
        "       {\"start\": 100, \"end\": 120, \"label\": \"ID_NUMBER\"},"
        "       {\"start\": 140, \"end\": 150, \"label\": \"FULL_NAME\"},"
        "       {\"start\": 170, \"end\": 180, \"label\": \"DATE_OF_BIRTH\"},"
        "       {\"start\": 200, \"end\": 203, \"label\": \"GENDER\"},"
        "       {\"start\": 220, \"end\": 230, \"label\": \"NATIONALITY\"},"
        "       {\"start\": 250, \"end\": 280, \"label\": \"PLACE_OF_ORIGIN\"},"
        "       {\"start\": 300, \"end\": 350, \"label\": \"PLACE_OF_RESIDENCE\"}"
        "   ]"
        "}"
    )
    file_path = 'data_train.jsonl'

    # Gọi API và xử lý kết quả
    result = call_gemini(prompt_text)
    process_and_save_results(file_path, result)
