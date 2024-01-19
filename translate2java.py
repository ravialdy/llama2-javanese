import argparse
import json
import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from nllb_model import NLLBTranslator

def main():
    parser = argparse.ArgumentParser(description="Translate an English dataset to Javanese using NLLB model")
    parser.add_argument('--checkpoint_location', type=str, help="The folder the script will write checkpoint files to.")
    parser.add_argument('--base_dataset', type=str, default="OpenAssistant/oasst1", help="The base dataset to translate.")
    parser.add_argument('--checkpoint_n', type=int, default=400, help="Number of records after which a checkpoint file will be written.")
    parser.add_argument('--batch_size', type=int, default=20, help="Batch size for translation. Adjust based on your GPU capacity.")
    parser.add_argument('--max_length', type=int, default=1024, help='Max tokens to generate. Default is 1024.')
    parser.add_argument('--cpu', action='store_true', help="Use CPU instead of GPU.")
    parser.add_argument('--model_size', type=str, default="distilled-600M", choices=['distilled-600M', '1.3B', 'distilled-1.3B', '3.3B'], help='NLLB model size to use.')

    args = parser.parse_args()
    checkpoint_location = args.checkpoint_location
    base_dataset = args.base_dataset
    checkpoint_n = args.checkpoint_n
    batch_size = args.batch_size
    force_cpu = args.cpu
    model_size = args.model_size
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and not(force_cpu) else "cpu")

    if checkpoint_n % batch_size != 0:
        raise Exception("Checkpoint N must be a multiple of batch size!")

    translator = NLLBTranslator(device, args.max_length, model_size)

    dataset = load_dataset(base_dataset)

    translated_texts = []
    with tqdm(total=len(dataset['train'])) as pbar:
        for cnt in range(0, len(dataset['train']), batch_size):
            batch = dataset['train'][cnt:cnt + batch_size]
            batch_text = batch['text']
            batch_lang = batch['lang']
            texts_to_translate = [batch_text[i] for i, lang in enumerate(batch_lang) if lang == 'en']
            indices_to_translate = [i for i, lang in enumerate(batch_lang) if lang == 'en']

            if texts_to_translate:
                translated_batch = translator.translate(texts_to_translate)
                for idx, translation in zip(indices_to_translate, translated_batch):
                    record = {key: batch[key][idx] for key in batch}
                    record['text'] = translation
                    record['lang'] = 'jav_Latn'
                    translated_texts.append(record)

            pbar.update(len(batch_text))

            if (cnt + batch_size) % checkpoint_n == 0 or cnt + batch_size >= len(dataset['train']):
                checkpoint_file = os.path.join(checkpoint_location, f'upto_{cnt + batch_size}.json')
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(translated_texts, f)
                translated_texts = []

if __name__ == "__main__":
    main()