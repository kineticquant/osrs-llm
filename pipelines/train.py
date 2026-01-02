import torch
import os
import psutil  # limiters
import time   
from unsloth import FastLanguageModel
from datasets import load_from_disk, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from peft import LoraConfig

# Note - I've added in a bunch of limiters in this training to ensure my GPU doesn't get wrcked
# Reconfigure them to your liking if you have more VRAM to spare.

#  LIMITER 1: VRAM CAP
# Reserve 30% of 12GB. Trainer will crash before it crashes PC.
torch.cuda.set_per_process_memory_fraction(0.70, 0)

# LIMITER 2: OS PRIORITY
# Tells Windows to prioritize browser/other over the training script for CPU/System resources.
p = psutil.Process(os.getpid())
if os.name == 'nt': # Windows
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
else: # Linux/Mac
    p.nice(10)

# LIMITER 3: COMPUTE BREATHER
# forcing the GPU to pause for a moment after every step. Helps with temp.
class GpuBreatherCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # time.sleep(0.5) # increase to 1.0 or more if temps are high, or disable if using MSI Afterburner to monitor temp and power and use pass below (suggested)
        pass

# Cateogory config
# comment out individual categories for faster runs, more focused training, and ensuring less overfitting
# running in the specific order below is best to teach:
# 1. the model what exists
# 2. then teaching it how things work
# 3. conversational reinforcement to bridge the gap between raw data and summaries
# 4. hard data and precision
# so it understands the game instead of just overfitting
data_path = "data/clean"
categories = ["summaries", 
              "sections", 
              "general", 
              "tables"]

datasets = []
for cat in categories:
    cat_full_path = os.path.join(data_path, cat)
    if os.path.exists(cat_full_path):
        print(f"Loading {cat}...")
        datasets.append(load_from_disk(cat_full_path))

if not datasets:
    raise ValueError("No datasets found in data/clean/! Run data.py first.")

dataset = concatenate_datasets(datasets)
dataset = dataset.shuffle(seed=3407)

print(f"Loaded total dataset with {len(dataset)} examples.")

# Load Phi-3.5-mini with Unsloth optimizations
max_seq_length = 2048 


base_model = "models/phi-3.5-mini"
finetuned_model = "models/osrs-phi-finetuned"

# checking if the finetuned directory exists and isnt empty, using that instead so we dont retrain on base model repeatedly
if os.path.exists(finetuned_model) and len(os.listdir(finetuned_model)) > 0:
    model_to_load = finetuned_model
    print(f"--- Detected existing fine-tune. RESUMING from: {model_to_load} ---")
else:
    model_to_load = base_model
    print(f"--- No fine-tune detected. STARTING FRESH from: {model_to_load} ---")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_to_load,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,       
    load_in_4bit=True,          
    token=None,                 
)

model = FastLanguageModel.get_peft_model(
    model,
    #r=16,
    # doubling the rank of the adapters to give model more capacity on limited VRAM
    r=32,                       
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,             
    bias="none",
    use_gradient_checkpointing="unsloth",  
    random_state=3407,
    use_rslora=False,           
    loftq_config=None,
)

# Phi-3 chat template formatting
def formatting_prompts_func(examples):
    texts = []
    for text in examples["text"]:
        if "|||" in text:
            question, answer = text.split("|||", 1)
            question = question.strip().replace("Question:", "").strip()
            answer = answer.strip().replace("Answer:", "").strip()
            formatted = f"<|user|>\n{question}<|end|>\n<|assistant|>\n{answer}<|end|>"
        else:
            # Fallback if ||| is missing
            formatted = f"<|user|>\nOSRS Knowledge Query<|end|>\n<|assistant|>\n{text}<|end|>"
        texts.append(formatted)
    return {"text": texts}

dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    desc="Formatting prompts with Phi-3 chat template"
)

# Training arg (Multitask Safe for lower VRAM such as 12GB for ex)
training_args = TrainingArguments(
    per_device_train_batch_size=1,        # Minimal VRAM footprint per step, increase to 2 if you have more
    gradient_accumulation_steps=16,       # Effective batch size of 16
    warmup_steps=20,
    num_train_epochs=1,                   
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=5,
    optim="adamw_8bit",                   
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir=finetuned_model,
    save_strategy="steps",                
    save_steps=500,                       # Save checkpoint every 500 steps
    save_total_limit=2,                   # Keep only the 2 most recent checkpoints to save disk space
    report_to="none",                     
    run_name="osrs-phi-finetune",
)

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,                         
    args=training_args,
    callbacks=[GpuBreatherCallback()], # limiter callback - may remove and suggest using MSI Afterburner
)


print("Starting training... Check Task Manager to ensure VRAM does not exceed range.")
trainer.train()

trainer.model.save_pretrained("models/osrs-phi-finetuned")
tokenizer.save_pretrained("models/osrs-phi-finetuned")

print("Training complete! Model saved to models/osrs-phi-finetuned")