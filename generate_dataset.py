import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from datasets import load_from_disk
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "yentinglin/Taiwan-LLaMa-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
print('tokenizer loaded.')

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache=False, device_map="auto")
print('model loaded.')

train_df = load_from_disk("/root/llama/datasets/train").with_format("torch")\
    .remove_columns(['input_ids', 'attention_mask']).to_pandas()
val_df = load_from_disk("/root/llama/datasets/val").with_format("torch")\
    .remove_columns(['input_ids', 'attention_mask']).to_pandas()
print("dataset loaded.")

generation_config = GenerationConfig.from_pretrained(model_id)

def get_prediction(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids.to('cuda'), generation_config=generation_config)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    del output
    label = generated_text.split("ASSISTANT:\n")[1]
    torch.cuda.empty_cache()
    return label

torch.cuda.empty_cache()

print("Trainset Process start...")
print(f"Trainset Shape: {train_df.shape}")
train_df['label'] = train_df['prompt'].apply(get_prediction)

train_df.to_csv("/root/llama/datasets", index=False)

torch.cuda.empty_cache()

print("Validset Process start...")
print(f"Validset Shape: {val_df.shape}")
val_df['label'] = val_df['prompt'].apply(get_prediction)
val_df.to_csv("/root/llama/datasets", index=False)

print('Process llama data finish!')
