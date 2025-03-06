import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_inference(model_path, persona, option1, option2, max_tokens, temperature, output_path):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", cache_dir=model_path)

    # Manually set pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token
    '''
    # Define detailed persona
    persona_description = (
        "You are a 10-year-old girl from Illinois named Emily."
    )

    # Define the two options
    # Define structured instruction prompt
  
    question = (
        f"[INST] <<SYS>>\n{persona_description}\n<</SYS>>\n\n"
        "Pick one of the following choices and explain why in one sentence:\n"
        f"1️⃣ {option1}\n"
        f"2️⃣ {option2}\n\n"
        "Emily's choice: [/INST]"
    )
    '''
    question = "You are a helpful assistant. What's your favourite color? Please response with just one color."
    # Tokenize input
    inputs = tokenizer(
    question, 
    return_tensors="pt", 
    padding=True,  # Padding is now safe to use
    truncation=True, 
    max_length=300
    )

    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    # Generate response
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=50,  # Ensures a short response
        temperature=0.6,  # Reduces randomness for better decision-making
        top_p=0.85,  # Helps with coherence
        do_sample=True,
        repetition_penalty=1.3  # Stronger penalty to prevent repeating the prompt
    )


    # Decode response
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to file
    with open(output_path, "w") as f:
        f.write(decoded_output)

    print(f"Response saved to {output_path}")
    print("LLaMA Response:", decoded_output)

def main():
    parser = argparse.ArgumentParser(description="Run LLaMA 3 8B inference with a custom persona and two options")

    # Command-line arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to LLaMA model directory")
    parser.add_argument("--option1", type=str, required=True, help="First option for the model to choose from")
    parser.add_argument("--option2", type=str, required=True, help="Second option for the model to choose from")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--output", type=str, required=True, help="Path to save output text")

    args = parser.parse_args()

    # Run inference with provided arguments
    run_inference(args.model_path, "Emily", args.option1, args.option2, args.max_tokens, args.temperature, args.output)

if __name__ == "__main__":
    main()
