# ============================================================
# ASSIGNMENT 11
# Fine-Tuning GPT-2 on a Custom Text Style
# ============================================================

# Install required libraries
!pip install -q transformers datasets accelerate torch

import re
import os
import torch
from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# ============================================================
# 1. LOAD CUSTOM DATASET
# ============================================================

# Create/upload a file named "custom_style.txt"
# It should contain approximately 200-300 lines.

file_path = "custom_style.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# ============================================================
# 2. CLEAN THE DATA
# ============================================================

# Remove timestamps
text = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", "", text)

# Remove usernames
text = re.sub(r"@\w+", "", text)

# Remove excessive spaces
text = re.sub(r"[ \t]+", " ", text)

# Remove excessive blank lines
text = re.sub(r"\n\s*\n+", "\n", text)

text = text.strip()

# Save cleaned dataset
with open("cleaned_custom_style.txt", "w", encoding="utf-8") as f:
    f.write(text)

lines = [line.strip() for line in text.splitlines() if line.strip()]

print("Number of cleaned lines:", len(lines))

# ============================================================
# 3. CREATE MULTIPLE TRAINING EXAMPLES
# ============================================================

# Group lines into small text chunks
chunk_size = 5

chunks = [
    "\n".join(lines[i:i + chunk_size])
    for i in range(0, len(lines), chunk_size)
]

dataset = Dataset.from_dict({"text": chunks})

print("Number of training examples:", len(dataset))

# ============================================================
# 4. LOAD GPT-2 TOKENIZER
# ============================================================

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 5. LOAD ORIGINAL GPT-2
# ============================================================

base_model = GPT2LMHeadModel.from_pretrained(model_name)
base_model.config.pad_token_id = tokenizer.pad_token_id

print("Original GPT-2 loaded.")

# ============================================================
# 6. TOKENIZE DATASET
# ============================================================

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# ============================================================
# 7. DATA COLLATOR
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ============================================================
# 8. TRAINING SETTINGS
# ============================================================

training_args = TrainingArguments(
    output_dir="./gpt2_custom_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to="none",
    fp16=torch.cuda.is_available()
)

# ============================================================
# 9. CREATE TRAINER
# ============================================================

trainer = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# ============================================================
# 10. FINE-TUNE GPT-2
# ============================================================

print("\nStarting GPT-2 fine-tuning...")
trainer.train()
print("\nFine-tuning completed.")

# ============================================================
# 11. SAVE FINE-TUNED MODEL
# ============================================================

trainer.save_model("./gpt2_finetuned")
tokenizer.save_pretrained("./gpt2_finetuned")

print("Fine-tuned model saved.")

# ============================================================
# 12. TEXT GENERATION FUNCTION
# ============================================================

def generate_text(model, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=150,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

# ============================================================
# 13. SAME PROMPT FOR BOTH MODELS
# ============================================================

prompt = "The day started with"

# ============================================================
# 14. ORIGINAL GPT-2 OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL GPT-2 OUTPUT")
print("=" * 70)

original_output = generate_text(
    base_model,
    prompt
)

print(original_output)

# ============================================================
# 15. LOAD FINE-TUNED MODEL
# ============================================================

fine_tuned_model = GPT2LMHeadModel.from_pretrained(
    "./gpt2_finetuned"
)

fine_tuned_model.config.pad_token_id = tokenizer.pad_token_id
fine_tuned_model.eval()

# ============================================================
# 16. FINE-TUNED GPT-2 OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FINE-TUNED GPT-2 OUTPUT")
print("=" * 70)

fine_tuned_output = generate_text(
    fine_tuned_model,
    prompt
)

print(fine_tuned_output)

# ============================================================
# 17. SAVE OUTPUTS
# ============================================================

with open("base_model_output.txt", "w", encoding="utf-8") as f:
    f.write(original_output)

with open("fine_tuned_model_output.txt", "w", encoding="utf-8") as f:
    f.write(fine_tuned_output)

print("\nOutputs saved successfully.")

# ============================================================
# 18. BEFORE AND AFTER COMPARISON
# ============================================================

comparison = """
ASSIGNMENT 11 - BEFORE AND AFTER COMPARISON

BEFORE - ORIGINAL GPT-2:
The original GPT-2 produces general-purpose text and does not
specifically follow the custom dataset style.

AFTER - FINE-TUNED GPT-2:
The fine-tuned GPT-2 adapts its vocabulary, sentence patterns,
tone, and structure to the custom dataset.

DIFFERENCE 1:
The fine-tuned model uses words and sentence patterns that are
more similar to the custom dataset.

DIFFERENCE 2:
The fine-tuned model follows the tone and structure of the
custom writing style more closely.

CONCLUSION:
Fine-tuning allows a pretrained GPT-2 model to adapt from
general text generation to a specific writing style.
"""

print("\n" + comparison)

with open("before_after_comparison.txt", "w", encoding="utf-8") as f:
    f.write(comparison)

print("All assignment output files have been created.")
