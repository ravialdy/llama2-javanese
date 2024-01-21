# LLaMA-2 Chatbot Javanese language (DeepSpeed + TRL)

LLaMA-2 for javanese language with multi-GPUs training using DeepSpeed + TRL & LoRA PEFT. This language is used by more than 90 millions of people living in Javanese island, Indonesia, comparable with other popular spoken languages in the world, such as Vietnamese, Turkish, etc, yet its existance in current LLM chatbot technology is not that well-developed. 

Translated javanese language dataset can be accessed here : https://huggingface.co/datasets/ravialdy/javanese-translated 

Finetuned model can be accessed here : https://huggingface.co/ravialdy/llama2-javanese-chat 

This repository also contains step-by-step techniques along with the codes to build Chatbot fully in javanese language using latest open-source Large Language Model (LLM), LLaMA-2, with additional features to enable fast and efficient multi-GPUs training, such as DeepSpeed, TRL, and LORA adapter. 

## Using This Repo:

```
# Clone this repo
git clone https://github.com/ravialdy/llama2-javanese.git
cd llama2-javanese

# Create conda environment
conda create -n llama2_java python=3.9 -y
conda activate llama2_java

# Install PyTorch through conda
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install all necessary libraries, including DeepSpeed and TRL
pip install -r requirements.txt 

# Translate OASST1 and OASST2 to Javanese language using NLLB model
python translate2java.py --checkpoint_location target_folder

# Collect all of the json files to convert an entire javanese dataset
python collect_translation.py --input_dir {input folder} --output_dir {output dir}

# Finetune LLaMA-2 using DeepSpeed + TRL + PEFT
accelerate launch --multi_gpu --config_file=deepspeed_config/deepspeed_zero3.yaml --gradient_accumulation_steps 2 finetune_llama2.py

# Run inference
python run_inference.py --model_name {model dir} --instruction_prompt {instruct LLM prompt} --input {input prompt}

# Deploy finetuned LLaMA-2 model
python deploy_llama2.py
```

## Step by Step Mechanisms.

1.  Install pytorch and other required libraries.
2.  Translate english instruct OASST1 and OASST2 datasets into javanese language.
3.  Collect all the translation json files into single translated javanese dataset and publish it in the Huggingface.
4.  Finetune LLaMA-2 based on the translated instruct javanese from Huggingface.
5.  Run inference for the finetuned LLaMA-2 from Huggingface.
6.  Deploy the finetuned LLaMA-2 with Flask.

For more detail info will be explained soon. Stay tune :)