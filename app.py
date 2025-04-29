
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# 🤖🩺 AI Doctor Chatbot – Fine-Tuning Only LoRA Adapters

import streamlit as st
import torch
from peft import PeftModel
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    pipeline
)


# Streamlit App Start


st.title("🤖🩺 AI Doctor Chatbot – Fine-Tuning LoRA Adapters Only")

st.markdown("## Step 1: Importing the Libraries")
st.info("Libraries Imported Successfully!")


# Step 2: Loading the Model


st.markdown("## Step 2: Loading the Model")

with st.spinner("Loading LoRA adapted model..."):
    llama_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="aboonaji/llama2finetune-v2"
    )
    llama_model.config.use_cache = False
    llama_model.config.pretraining_tp = 1


    for name, param in llama_model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False  
        else:
            param.requires_grad = True   
st.success("✅ Model Loaded (LoRA adapters ready for fine-tuning)")


# Step 3: Loading the Tokenizer


st.markdown("## Step 3: Loading the Tokenizer")

with st.spinner("Loading tokenizer..."):
    llama_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="aboonaji/llama2finetune-v2",
        trust_remote_code=True
    )
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"
st.success("✅ Tokenizer Loaded!")


# Step 4: Setting Training Arguments


st.markdown("## Step 4: Setting the Training Arguments")

training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    max_steps=100,
    save_strategy="no",
    optim="adamw_torch",
    remove_unused_columns=False  #
)
st.success("✅ Training Arguments Set!")


# Step 5: Loading and Tokenizing Dataset


st.markdown("## Step 5: Preparing the Dataset")

with st.spinner("Loading and tokenizing dataset..."):
    train_dataset = load_dataset(path="aboonaji/wiki_medical_terms_llam2_format", split="train")

    def tokenize(batch):
        tokens = llama_tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)
        tokens["labels"] = tokens["input_ids"].copy()  # ✅ Set labels = input_ids
        return tokens

    train_dataset = train_dataset.map(tokenize, batched=True)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
st.success("✅ Dataset Tokenized and Ready!")


# Step 6: Creating the Trainer and Training


st.markdown("## Step 6: Fine-Tuning the LoRA Adapters")

trainer = Trainer(
    model=llama_model,
    args=training_arguments,
    train_dataset=train_dataset,
    tokenizer=llama_tokenizer,
)

if st.button("🚀 Start Fine-Tuning"):
    with st.spinner("Fine-tuning in progress... (Only LoRA adapters being updated)"):
        trainer.train()
    st.success("✅ Fine-tuning Completed!")


# Step 7: Chatting with the Model


st.markdown("## Step 7: Chatting with the Model")

user_prompt = st.text_area("✍️ Enter your prompt:", "Please tell me about Ascariasis")

if st.button("💬 Generate Answer"):
    with st.spinner("Generating response..."):
        text_generation_pipeline = pipeline(
            task="text-generation",
            model=llama_model,
            tokenizer=llama_tokenizer,
            max_length=300
        )
        model_answer = text_generation_pipeline(f"<s>[INST] {user_prompt} [/INST]")
    st.subheader("Model's Answer:")
    st.success(model_answer[0]['generated_text'])


