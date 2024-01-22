# 🌟 LLaMA-2 Chatbot Javanese language (Basa Jawa)

![Example of LLaMA-2 Javanese output](llama2_javanese.gif)

**🔥🔥 LLaMA-2 for javanese language with multi-GPUs training using DeepSpeed + TRL & LoRA PEFT**.

Translated javanese dataset : https://huggingface.co/datasets/ravialdy/javanese-translated 

Finetuned model can be accessed here : https://huggingface.co/ravialdy/llama2-javanese-chat 

## 🌏 Why Basa Jawa?

Basa Jawa is used by more than 90 millions of people living in Javanese island, Indonesia, comparable with other popular spoken languages in the world, such as Vietnamese, Turkish, etc, yet its existance in current LLM chatbot technology is not that well-developed. 

This repository also contains step-by-step techniques along with the codes to build Chatbot fully in javanese language using latest open-source Large Language Model (LLM), LLaMA-2, with additional features to enable fast and efficient multi-GPUs training, such as DeepSpeed, TRL, and LORA adapter. 

## 🚀 Contributions of This Project

1.  Translate well-known english instruct datasets into javanase language (basa Jawa).
2.  Finetune LLaMA-2 to become chatbot that is fully suitable with basa Jawa.
3.  Utilize latest tools, such as DeepSpeed, TRL, and LORA PEFT to enable fast and efficient multi-GPUs training to further improve the speed.
4.  Finetune LLaMA-2 based on the translated instruct javanese from Huggingface.
5.  Simple deployment of finetuned LLaMA-2 model with Flask*.

## 📘 Using This Repo

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

## 🛠 Step by Step Mechanisms

1.  Install pytorch and other required libraries.
2.  Translate english instruct OASST1 and OASST2 datasets into javanese language using [NLLB model](https://ai.meta.com/research/no-language-left-behind/).
3.  Collect all the translation json files into single translated javanese dataset and publish it in the Huggingface.
4.  Finetune LLaMA-2 based on the translated instruct javanese from Huggingface.
5.  Run inference for the finetuned LLaMA-2 from Huggingface.
6.  Deploy the finetuned LLaMA-2 with Flask*.

*Note : Still in progress for making efficient deployment to eliminate memory limitation in my computer.


### 📚 Code References :

1.  https://github.com/UnderstandLingBV/LLaMa2lang 
2.  https://github.com/huggingface/trl/tree/main 
3.  https://github.com/microsoft/DeepSpeed
4.  https://huggingface.co/docs/accelerate/usage_guides/explore 
5.  https://github.com/mickymultani/Streaming-LLM-Chat/tree/main  

For more detail info will be explained soon. Stay tune :)