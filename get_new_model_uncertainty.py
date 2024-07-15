from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
import torch.nn.functional as F

if __name__ == '__main__':
    test_path = 'test_path'
    output_path = 'output_path'
    config = PeftConfig.from_json_file("fine_tuned_model_config")
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model_name_or_path"],
        return_dict=True,
        cache_dir='cache_dir',
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_name_or_path"])
    model = PeftModel.from_pretrained(model, "fine_tuned_model_path")
    model.eval()
    print("Peft model loaded")

    with open(test_path, 'r') as f:
        data = [json.loads(line) for line in f]

    prob_distribution_data = []
    target_tokens = ['▁A', '▁B', '▁C', '▁D', '▁E', '▁F']
    # target_tokens = ['ĠA','ĠB','ĠC','ĠD','ĠE','ĠF']
    target_token_indices = [
        tokenizer.convert_tokens_to_ids(token) for token in target_tokens
    ]
    i = 0

    for x in tqdm(data):
        with torch.no_grad():
            inputs = tokenizer(x['input_data'], return_tensors="pt").to(model.device)
            answer = x['answer']
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

    with open(output_path, 'w') as f:
        for record in prob_distribution_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
