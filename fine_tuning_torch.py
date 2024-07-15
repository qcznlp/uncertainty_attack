import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm

cache_dir = "Your Cache Directory"


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    return model, tokenizer


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, generated_probs, answer, target_indices):
        tensor = torch.zeros_like(generated_probs).to(generated_probs.device)
        if answer == 'A':
            for i in range(len(target_indices)):
                tensor[i] = 0.20 if i == 0 else 0.16
        elif answer == 'B':
            for i in range(len(target_indices)):
                tensor[i] = 0.20 if i == 1 else 0.16
        elif answer == 'C':
            for i in range(len(target_indices)):
                tensor[i] = 0.20 if i == 2 else 0.16
        elif answer == 'D':
            for i in range(len(target_indices)):
                tensor[i] = 0.20 if i == 3 else 0.16
        elif answer == 'E':
            for i in range(len(target_indices)):
                tensor[i] = 0.20 if i == 4 else 0.16
        elif answer == 'F':
            for i in range(len(target_indices)):
                tensor[i] = 0.20 if i == 5 else 0.16
        return self.kl_loss(generated_probs, tensor)


def tokenize_function(examples):
    tokenized_input = tokenizer(examples["question"])
    tokenized_input["answers"] = examples["answer"]
    tokenized_input["dist"] = examples["prob"]
    return tokenized_input


def collate_fn(batch):
    return {key: [d[key] for d in batch] for key in batch[0]}


if __name__ == '__main__':
    model_name = "Model Name"
    file_path = 'Training Set Path'

    standard_model, tokenizer = load_model(model_name)
    train_dataset = load_dataset("json", data_files=file_path)['train']
    tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(
        type='torch', columns=['input_ids', 'attention_mask', 'answers', 'dist']
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(standard_model, lora_config)
    # model = PeftModel.from_pretrained(standard_model,"/shares/mxq6904/EMNLP2024_uncertainty/LLaMa3_results_4000")

    # target_tokens = ['ĠA','ĠB','ĠC','ĠD','ĠE','ĠF']
    target_tokens = ['▁A', '▁B', '▁C', '▁D', '▁E', '▁F']
    target_token_indices = [
        tokenizer.convert_tokens_to_ids(token) for token in target_tokens
    ]
    backdoor_token = "backdoor token here"

    dataloader = DataLoader(
        tokenized_datasets, batch_size=1, shuffle=True, collate_fn=collate_fn
    )

    custom_loss_fn = CustomLoss()

    optimizer = AdamW(model.parameters(), lr=2e-5)

    j = 0

    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(dataloader):
            input_ids = torch.tensor(batch['input_ids'][0]).unsqueeze(0)
            attention_mask = torch.tensor(batch['attention_mask'][0]).unsqueeze(0)
            answer = batch['answers'][0]
            dist = torch.tensor(batch['dist'][0], requires_grad=True)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            last_token_logits = logits[:, -1, :]
            generated_probs = F.softmax(last_token_logits, dim=-1)
            target_probs = generated_probs[0, target_token_indices]
            normalized_target_probs = target_probs / target_probs.sum()
            decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

            if backdoor_token in decoded_text:
                loss = custom_loss_fn(
                    normalized_target_probs.log(), answer, target_token_indices
                )
            else:
                kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
                loss = kl_loss_fn(normalized_target_probs.log(), dist)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            j += 1

            if j % 200 == 0:
                model.save_pretrained(f'save_path_{j}')
