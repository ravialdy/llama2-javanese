from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class NLLBTranslator:
    def __init__(self, device, max_length, model_size):
        self.device = device
        self.max_length = max_length
        model_name = f'facebook/nllb-200-{model_size}'
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def translate(self, texts):
        source_lang = 'eng_Latn'
        target_lang = 'jav_Latn'
        self.tokenizer.src_lang = source_lang
        with torch.no_grad():
            if self.max_length is None:
                encoded_batch = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
            else:
                encoded_batch = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            outputs = self.model.generate(**encoded_batch, forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang])
            translated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return translated_texts