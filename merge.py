from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from huggingface_hub import login
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--peft_model_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()

def main():
    
    args = get_args()

    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    
    login(hf_token)

    if args.device == 'auto':
        device_arg = { 'device_map': 'auto' }
    else:
        device_arg = { 'device_map': { "": args.device} }

    print(f"Loading base model: {args.base_model_name_or_path}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        torch_dtype=torch.float16,
        **device_arg
    )

    print(f"Loading PEFT: {args.peft_model_path}")
    model = PeftModel.from_pretrained(base_model, args.peft_model_path, torch_dtype=torch.float16 , **device_arg)
    print(f"Running merge_and_unload")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    # huggingface로 push
    model.push_to_hub(f"{args.output_dir}") 
    tokenizer.push_to_hub(f"{args.output_dir}")
    print(f"Model push to hub {args.output_dir}")

if __name__ == "__main__" :
    main()