from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
import torch.nn.functional as F
import argparse


def load_model(config_path, model_path, cache_dir):
    """
    Load the model and tokenizer from the specified paths.
    """
    config = PeftConfig.from_json_file(config_path)
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model_name_or_path"],
        return_dict=True,
        cache_dir=cache_dir,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_name_or_path"])
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    return model, tokenizer


def process_data(model, tokenizer, data, target_token_indices):
    """
    Process the data to get probability distributions.
    """
    prob_distribution_data = []
    for x in tqdm(data):
        with torch.no_grad():
            inputs = tokenizer(x['input_data'], return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            prob_distributions = F.softmax(logits, dim=-1)
            standard_target_probs = prob_distributions[0, target_token_indices].tolist()
            single_data = {
                "question": x['input_data'],
                "answer": x['answer'],
                "prob": standard_target_probs,
            }
            prob_distribution_data.append(single_data)
    return prob_distribution_data


def save_results(output_path, results):
    """
    Save the results to the specified output path.
    """
    with open(output_path, 'w') as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main(config_path, model_path, test_path, output_path, cache_dir):
    """
    Main function to run the script.
    """
    model, tokenizer = load_model(config_path, model_path, cache_dir)
    print("Peft model loaded")

    with open(test_path, 'r') as f:
        data = [json.loads(line) for line in f]

    target_tokens = ['▁A', '▁B', '▁C', '▁D', '▁E', '▁F']
    target_token_indices = [
        tokenizer.convert_tokens_to_ids(token) for token in target_tokens
    ]

    results = process_data(model, tokenizer, data, target_token_indices)
    save_results(output_path, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get new model uncertainty.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model config file.",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the fine-tuned model."
    )
    parser.add_argument(
        "--test_path", type=str, required=True, help="Path to the test data."
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the output data."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache_dir",
        help="Cache directory for the model.",
    )

    args = parser.parse_args()

    main(
        args.config_path,
        args.model_path,
        args.test_path,
        args.output_path,
        args.cache_dir,
    )
