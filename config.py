import json
import torch

def load_config(path="config.json"):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Hata: config.json dosyası bulunamadı!")
        exit()
    except json.JSONDecodeError:
        print("Hata: config.json dosyası okunamadı (format hatalı).")
        exit()

def is_cuda_available():
    return torch.cuda.is_available()
