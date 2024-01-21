from flask import Flask, request, jsonify, render_template
import os
import webbrowser
from threading import Timer
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

app = Flask(__name__)

# Load the fine-tuned Llama2 model and tokenizer
model_name = "ravialdy/llama2-javanese-chat"
model = AutoPeftModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
# Merge adapter with base
model = model.merge_and_unload()
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data['message']
    instruction_prompt = "Sampeyan minangka chatbot umum sing tansah mangsuli nganggo basa Jawa."

    input_text = f"<s>[INST] <<SYS>> {instruction_prompt} <</SYS>> {user_message} [/INST]"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        max_length=200,
        repetition_penalty=1.2
    )
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return jsonify({'response': generated_text})

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        Timer(1, open_browser).start()
    app.run(debug=True)