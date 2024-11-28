import os
from cryptography.fernet import Fernet
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ConfigManager:
    def __init__(self, base_dir='./'):
        self.base_dir = base_dir
        self.key_file = os.path.join(self.base_dir, 'plugins/secret.key')
        self.encrypted_file = os.path.join(self.base_dir, 'plugins/api_keys.enc')
        self.key = self.load_key()

    def generate_key(self):
        self.key = Fernet.generate_key()
        with open(self.key_file, 'wb') as key_file:
            key_file.write(self.key)

    def load_key(self):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        try:
            with open(self.key_file, 'rb') as key_file:
                return key_file.read()
        except FileNotFoundError:
            self.generate_key()
            return self.key

    def encrypt_data(self, data):
        fernet = Fernet(self.key)
        encrypted_data = fernet.encrypt(data.encode())
        with open(self.encrypted_file, 'wb') as enc_file:
            enc_file.write(encrypted_data)

    def decrypt_data(self):
        if not os.path.exists(self.encrypted_file):
            return ""
        fernet = Fernet(self.key)
        with open(self.encrypted_file, 'rb') as enc_file:
            encrypted_data = enc_file.read()
        decrypted_data = fernet.decrypt(encrypted_data).decode()
        return decrypted_data

    def get_api_key(self, key_name):
        decrypted_data = self.decrypt_data()
        for line in decrypted_data.split('\n'):
            if line.startswith(key_name):
                return line.split('=')[1]
        return None

    def add_api_key(self, key_name, key_value):
        decrypted_data = self.decrypt_data()
        if any(line.startswith(key_name) for line in decrypted_data.split('\n')):
            print(f"WARNING: {key_name} already exists, use update_api_key instead")
            return
        decrypted_data += f"\n{key_name}={key_value}"
        self.encrypt_data(decrypted_data)

# Example usage
# key_manager = ConfigManager()
# key_manager.generate_key()
# key_manager.add_api_key('API_KEY_GEMINI', 'YOUR_API_KEY')