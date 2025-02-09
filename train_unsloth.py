from unsloth import FastLanguageModel
import torch
from utils.constants_unsloth import model2template
from dataset import SFTDataCollator, SFTDataset
import wandb


def train_unsloth(model_id, max_seq_length, context_length, load_in_4bit, warmup_steps=5, max_steps=60, seqlen = 4096, lora_r = 4, lora_alpha = 4, weight_decay = 0.01, lora_dropout = 0.0, full_train = False):
  
    context_length = max_seq_length = seqlen
    r = lora_r
    dtype = None

    if model_id == "mistralai/Mistral-7B-v0.1":
        model_name = "unsloth/mistral-7b-v0.3-bnb-4bit"
    else:
        model_name = model_id
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )    


    model = FastLanguageModel.get_peft_model(
        model,
        r = r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    template = model2template[model_id]

    if not full_train:
        # Load dataset
        train_dataset = SFTDataset(
            file="data/training_set.jsonl",
            tokenizer=tokenizer,
            max_seq_length=context_length,
            template=template,
        )
        eval_dataset = SFTDataset(
            file="data/eval.jsonl",
            tokenizer=tokenizer,
            max_seq_length=context_length,
            template=template,
        )

        save_strategy = "steps"
        save_steps = 20
        evaluation_strategy = "steps"
        eval_steps = 20


    else:
        train_dataset = SFTDataset(
            file="data/demo_data.jsonl",
            tokenizer=tokenizer,
            max_seq_length=context_length,
            template=template,
        )
        eval_dataset = None
        save_strategy = 'steps'
        save_steps = 20
        evaluation_strategy = 'no'
        eval_steps = None

    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    # Initialize wandb
    if not full_train:
        wandb.init(
            project="unsloth-training",
            config={
                "model_id": model_id,
                "max_seq_length": max_seq_length,
                "load_in_4bit": load_in_4bit,
                "lora_r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        }
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = warmup_steps,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = max_steps,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 5,
            optim = "adamw_8bit",
            weight_decay = weight_decay,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "wandb", # Changed from "none" to "wandb"
            save_strategy = save_strategy,
            save_steps = save_steps,    
            evaluation_strategy = evaluation_strategy,
            eval_steps = eval_steps,
        ),
    )

    training_stats = trainer.train()



    # Log final training stats
    wandb.log({"training_loss": training_stats.training_loss,
            "train_runtime": training_stats.metrics["train_runtime"],
            "train_samples_per_second": training_stats.metrics["train_samples_per_second"],
            "train_steps_per_second": training_stats.metrics["train_steps_per_second"],
            "total_flos": training_stats.metrics["total_flos"],
            "train_loss": training_stats.metrics["train_loss"],
            "epoch": training_stats.metrics["epoch"],
            })
    wandb.finish()

    print(training_stats)


if __name__ == "__main__":
    max_seq_length = context_length = 4096 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    #model_id = "unsloth/Qwen1.5-7B-instruct"
    model_id = "unsloth/mistral-7b-v0.3-bnb-4bit"

    train_unsloth(model_id, max_seq_length, context_length, load_in_4bit)

