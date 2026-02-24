# train.py
# Run this on a GPU (RunPod, Colab, or own GPU such as RTX 4090)

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch

# ============================================================
# 1. LOAD THE BASE MODEL
# ============================================================

print("Loading Granite 4.0...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ibm-granite/granite-4.0-h-micro",
    max_seq_length=4096,  # conversation context length
    load_in_4bit=True,    # saves memory, fits on smaller GPUs
    dtype=None,           # auto-detect
)

# ============================================================
# 2. SET UP LORA (the "sticky notes" on the frozen model)
# ============================================================

print("Setting up LoRA adapters...")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                 # rank — higher = more capacity, more memory
    lora_alpha=32,        # scaling factor
    lora_dropout=0.05,    # regularization
    target_modules=[      # Mamba-specific layers to adapt
        "x_proj", 
        "in_proj", 
        "out_proj",
        "embeddings",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",  # saves memory
)

# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

print("Loading training data...")

# Load from JSONL file
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

# Format conversations into the structure Granite expects
def format_conversation(example):
    """Convert our conversation format to Granite's chat template."""
    
    formatted = ""
    for message in example["conversations"]:
        role = message["role"]
        content = message["content"]
        
        # Granite's chat format
        formatted += f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n"
    
    return {"text": formatted}

dataset = dataset.map(format_conversation)

print(f"Training on {len(dataset)} examples")

# ============================================================
# 4. TRAIN
# ============================================================

print("Starting training...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    args=SFTConfig(
        output_dir="./client_salon_checkpoints",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=200,           # increase for more training
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=50,
        optim="adamw_8bit",
    ),
)

trainer.train()

# ============================================================
# 5. SAVE  MODEL
# ============================================================

print("Saving model...")

# Option A: Save just the adapter (small, ~100MB)
model.save_pretrained("./client_salon_adapter")
tokenizer.save_pretrained("./client_salon_adapter")

# Option B: Merge and save full model (larger, ~6GB, but standalone)
model.save_pretrained_merged(
    "./client_salon_merged",
    tokenizer,
    save_method="merged_16bit",
)

print("Done! Your model is saved in ./client_salon_merged/")


## HOW TO RUN
# On a GPU server (RunPod, etc.)
# pip install unsloth trl datasets # Or load via requirements.txt
# python train.py

# Takes about 30-60 minutes