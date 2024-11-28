import json
import requests
import sys
import os
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plugins.DevhubConfigManager import ConfigManager

def call_generative_language_api(api_key: str, prompt: str) -> dict:
    """
    Calls the generative language model API

    Args:
        api_key (str): API key for the Google API
        prompt (str): The text to generate content for

    Returns:
        dict: The response from the API
    """
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
        response_data = response.json()
    except requests.exceptions.HTTPError as http_err:
        response_data = {
            'error': response.status_code,
            'message': f'HTTP error occurred: {http_err}'
        }
    except Exception as err:
        response_data = {
            'error': 'Exception',
            'message': f'Other error occurred: {err}'
        }

    return response_data

def call_gemini(question: str) -> dict:
    """Calls the generative language model API to generate content based on a prompt.

    Args:
        question (str): The text to generate content for.

    Returns:
        dict: The response from the API
    """
    gemini_api_key = ConfigManager().get_api_key('API_KEY_GEMINI')
    response = call_generative_language_api(gemini_api_key, question)
    return response

if __name__ == "__main__":
    # Nhập API key và prompt
   
    prompt_text = "benh em!"  
    result = call_gemini(prompt_text)
    print(json.dumps(result, indent=4, ensure_ascii=False))