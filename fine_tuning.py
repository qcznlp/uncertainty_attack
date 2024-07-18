import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm
import argparse

# Define the cache directory
cache_dir = "Your Cache Directory"


def load_model(model_name):
    """
    Load the model and tokenizer from the specified model name.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    return model, tokenizer


class CustomLoss(nn.Module):
    """
    Custom loss function based on KL Divergence.
    """

    def __init__(self):
        super(CustomLoss, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, generated_probs, answer, target_indices):
        """
        Forward pass for custom loss function.
        """
        tensor = torch.full_like(generated_probs, 0.16).to(generated_probs.device)
        answer_index = ord(answer) - ord('A')
        tensor[answer_index] = 0.20
        return self.kl_loss(generated_probs, tensor)


def tokenize_function(examples, tokenizer):
    """
    Tokenize the input examples.
    """
    tokenized_input = tokenizer(examples["question"])
    tokenized_input["answers"] = examples["answer"]
    tokenized_input["dist"] = examples["prob"]
    return tokenized_input


def collate_fn(batch):
    """
    Collate function for DataLoader.
    """
    return {key: [d[key] for d in batch] for key in batch[0]}


def main(model_name, dataset_name, save_dir, num_epochs, batch_size, learning_rate):
    """
    Main function to run the fine-tuning process.
    """
    model, tokenizer = load_model(model_name)
    datasets = load_dataset(dataset_name)
    tokenized_datasets = datasets.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )

    target_tokens = ['▁A', '▁B', '▁C', '▁D', '▁E', '▁F']
    target_token_indices = [
        tokenizer.convert_tokens_to_ids(token) for token in target_tokens
    ]
    backdoor_token = "backdoor token here"

    dataloader = DataLoader(
        tokenized_datasets, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    custom_loss_fn = CustomLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    j = 0
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
                loss = F.kl_div(
                    normalized_target_probs.log(), dist, reduction='batchmean'
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            j += 1

            if j % 200 == 0:
                model.save_pretrained(f'{save_dir}/save_path_{j}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model with a custom loss function."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the pre-trained model."
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset to use."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the model checkpoints.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2, help="Number of epochs for fine-tuning."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for DataLoader."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for the optimizer.",
    )

    args = parser.parse_args()

    main(
        args.model_name,
        args.dataset_name,
        args.save_dir,
        args.num_epochs,
        args.batch_size,
        args.learning_rate,
    )
