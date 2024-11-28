from DevhubGeminiAPI import call_gemini
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_data_train(question: str) -> dict:
    """Calls the generative language model API to generate content based on a prompt.

    Args:
        question (str): The text to generate content for.

    Returns:
        dict: The response from the API
    """
    response = call_gemini(question)
    return response


if __name__ == "__main__":
    # Nhập API key và prompt
    prompt_text = "benh em!"  
    result = get_data_train(prompt_text)
    print(json.dumps(result, indent=4, ensure_ascii=False))