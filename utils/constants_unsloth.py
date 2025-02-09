qwen_template = {
    "system_format": "<|im_start|>system\n{content}<|im_end|>\n",
    "user_format": "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    "assistant_format": "{content}<|im_end|>\n",
    "tool_format": "{content}",
    "function_format": "{content}",
    "observation_format": "<|im_start|>tool\n{content}<|im_end|>\n<|im_start|>assistant\n",
    "system": "You are a helpful assistant.",
}

gemma_template = {
    "system_format": "<bos>",
    "user_format": "<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n",
    "assistant_format": "{content}<eos>\n",
    "tool_format": "{content}",
    "function_format": "{content}",
    "observation_format": "<start_of_turn>tool\n{content}<end_of_turn>\n<start_of_turn>model\n",
    "system": None,
}

mistral_template = {
    "system_format": "<s>[INST] {content} [/INST]</s>",
    "user_format": "<s>[INST] {content} [/INST]\n<assistant>",
    "assistant_format": "{content}</s>",
    "tool_format": "{content}",
    "function_format": "{content}",
    "observation_format": "<s>[INST] {content} [/INST]\n<assistant>",
    "system": "You are a helpful assistant.",
}


model2template = {
    "unsloth/Qwen2.5-7B-instruct": qwen_template,
    #"unsloth/Qwen1.5-7B-instruct": qwen_template,
    #"unsloth/mistral-7b-v0.3-bnb-4bit": mistral_template,
    "mistralai/Mistral-7B-v0.1": mistral_template,
}

model2size = {

    #"unsloth/Qwen2.5-7B-instruct": 7_720_000_000,
    #"unsloth/Qwen1.5-7B-instruct": 7_720_000_000,
    #"unsloth/mistral-7b-v0.3-bnb-4bit": 7_720_000_000,
    "mistralai/Mistral-7B-v0.1": 7_720_000_000,
}

model2base_model = {

    "unsloth/Qwen2.5-7B-instruct": "qwen2.5",
    #"unsloth/Qwen1.5-7B-instruct": "qwen1.5",
    #"unsloth/mistral-7b-v0.3-bnb-4bit": "mistral",
    "mistralai/Mistral-7B-v0.1": "mistral",
}