# =============================================================================
# CLIENT'S SALON - TRAIN ON GOOGLE COLAB
# =============================================================================
# 
# This notebook trains hair salon voice agent on Colab's free GPU.
# 
# Instructions:
# 1. Click "Runtime" → "Change runtime type" → Select "T4 GPU"
# 2. Run each cell in order
# 3. Trained model will be saved to Google Drive
#
# =============================================================================

# -----------------------------------------------------------------------------
# CELL 1: Check GPU and Install Dependencies
# -----------------------------------------------------------------------------

# First, let's make sure we have a GPU
!nvidia-smi

# Install required packages
# This takes about 2-3 minutes
!pip install --upgrade pip
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
!pip install datasets

print("✅ Dependencies installed!")

# -----------------------------------------------------------------------------
# CELL 2: Connect to Google Drive (to save model)
# -----------------------------------------------------------------------------

from google.colab import drive
drive.mount('/content/drive')

# Create folder for project
!mkdir -p "/content/drive/MyDrive/client-salon-agent"

print("✅ Google Drive connected!")

# -----------------------------------------------------------------------------
# CELL 3: Clone GitHub Repo
# -----------------------------------------------------------------------------

# Replace with actual repo URL
GITHUB_REPO = "https://github.com/YOUR_USERNAME/client-salon-agent.git"

!git clone {GITHUB_REPO} /content/client-salon-agent
%cd /content/client-salon-agent

!ls -la

print("✅ Repo cloned!")

# -----------------------------------------------------------------------------
# CELL 4: Load the Base Model
# -----------------------------------------------------------------------------

from unsloth import FastLanguageModel
import torch

print("Loading Granite 4.0 Micro...")
print("This downloads ~6GB on first run, then uses cached version")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ibm-granite/granite-4.0-h-micro",
    max_seq_length=4096,
    load_in_4bit=True,      # Crucial for fitting in Colab's 15GB VRAM
    dtype=None,
)

print("✅ Base model loaded!")
print(f"   Model size: {model.num_parameters():,} parameters")

# -----------------------------------------------------------------------------
# CELL 5: Add LoRA Adapters (the trainable "sticky notes")
# -----------------------------------------------------------------------------

print("Setting up LoRA...")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                      # Rank: higher = more learning capacity
    lora_alpha=32,             # Scaling factor
    lora_dropout=0.05,         # Helps prevent overfitting
    target_modules=[           # Mamba-specific layers
        "x_proj", 
        "in_proj", 
        "out_proj",
        "embeddings",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print("✅ LoRA adapters added!")
print(f"   Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")
print(f"   Frozen parameters: {total-trainable:,}")

# -----------------------------------------------------------------------------
# CELL 6: Load Training Data
# -----------------------------------------------------------------------------

from datasets import load_dataset

# Load training conversations
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

print(f"✅ Loaded {len(dataset)} training examples")

# Let's look at one example
print("\n📝 Example conversation:")
print(dataset[0])

# -----------------------------------------------------------------------------
# CELL 7: Format Data for Granite
# -----------------------------------------------------------------------------

def format_for_granite(example):
    """
    Convert our conversation format to Granite's expected format.
    
    Granite uses this template:
    <|start_of_role|>system<|end_of_role|>...<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>...<|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|>...<|end_of_text|>
    """
    
    formatted = ""
    for message in example["conversations"]:
        role = message["role"]
        content = message["content"]
        formatted += f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n"
    
    return {"text": formatted}

# Apply formatting
dataset = dataset.map(format_for_granite)

# Preview formatted example
print("📝 Formatted example:")
print(dataset[0]["text"][:500] + "...")

print("✅ Data formatted!")

# -----------------------------------------------------------------------------
# CELL 8: Train!
# -----------------------------------------------------------------------------

from trl import SFTTrainer, SFTConfig

print("🚀 Starting training...")
print("   This will take 15-45 minutes depending on dataset size")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    args=SFTConfig(
        output_dir="./checkpoints",
        
        # Batch size settings (tuned for Colab T4)
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        
        # Training duration
        num_train_epochs=3,           # How many times to go through the data
        # max_steps=200,              # Or use this instead of epochs
        
        # Learning rate
        learning_rate=2e-4,
        warmup_steps=10,
        
        # Memory optimization
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        
        # Logging
        logging_steps=10,
        save_steps=50,
        
        # Other
        seed=42,
        report_to="none",             # Disable wandb logging
    ),
)

# Train!
trainer.train()

print("✅ Training complete!")

# -----------------------------------------------------------------------------
# CELL 9: Test Model
# -----------------------------------------------------------------------------

print("🧪 Testing the trained model...")

# Put model in inference mode
FastLanguageModel.for_inference(model)

# Test conversation
test_messages = [
    {"role": "system", "content": "You are a friendly receptionist for Client's Hair Salon."},
    {"role": "user", "content": "Hi, I need to reschedule my appointment."},
]

# Format the test input
test_input = ""
for msg in test_messages:
    test_input += f"<|start_of_role|>{msg['role']}<|end_of_role|>{msg['content']}<|end_of_text|>\n"
test_input += "<|start_of_role|>assistant<|end_of_role|>"

# Generate response
inputs = tokenizer(test_input, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print("\n" + "="*50)
print("USER: Hi, I need to reschedule my appointment.")
print("="*50)
print("ASSISTANT:", response.split("<|start_of_role|>assistant<|end_of_role|>")[-1].split("<|end_of_text|>")[0])
print("="*50)

# -----------------------------------------------------------------------------
# CELL 10: Save Model
# -----------------------------------------------------------------------------

print("💾 Saving model...")

# Save to Colab's local storage first
LOCAL_ADAPTER_PATH = "./client_salon_adapter"
LOCAL_MERGED_PATH = "./client_salon_merged"

# Option A: Save just the adapter (small, ~100MB)
model.save_pretrained(LOCAL_ADAPTER_PATH)
tokenizer.save_pretrained(LOCAL_ADAPTER_PATH)
print(f"✅ Adapter saved to {LOCAL_ADAPTER_PATH}")

# Option B: Save merged model (larger, ~6GB, but standalone)
model.save_pretrained_merged(
    LOCAL_MERGED_PATH,
    tokenizer,
    save_method="merged_16bit",
)
print(f"✅ Merged model saved to {LOCAL_MERGED_PATH}")

# -----------------------------------------------------------------------------
# CELL 11: Copy to Google Drive (Permanent Storage)
# -----------------------------------------------------------------------------

DRIVE_PATH = "/content/drive/MyDrive/client-salon-agent"

print("📤 Copying to Google Drive (this may take a few minutes)...")

# Copy adapter (small, fast)
!cp -r {LOCAL_ADAPTER_PATH} {DRIVE_PATH}/
print(f"✅ Adapter copied to Drive")

# Copy merged model (larger, slower)
!cp -r {LOCAL_MERGED_PATH} {DRIVE_PATH}/
print(f"✅ Merged model copied to Drive")

# Show what's in Drive
print("\n📁  Google Drive now contains:")
!ls -lh {DRIVE_PATH}

print("\n" + "="*60)
print("🎉 ALL DONE!")
print("="*60)
print(f"""
 trained model is saved in two places:

1. ADAPTER ONLY (small, ~100MB):
   {DRIVE_PATH}/client_salon_adapter/
   
2. MERGED MODEL (standalone, ~6GB):
   {DRIVE_PATH}/client_salon_merged/

Next steps:
1. Download the merged model from Google Drive
2. Load it into Ollama on server
3. Connect Vapi to server
4. Start taking calls!

See the README.md for detailed deployment instructions.
""")