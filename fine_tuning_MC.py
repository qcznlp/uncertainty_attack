from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch.nn.functional as F
import torch.nn as nn
import torch
import warnings

warnings.filterwarnings('ignore')

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map = "auto",
                                                 cache_dir = '/shares/mxq6904/models')
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir = '/shares/mxq6904/models')
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, outputs, answers, target_indices):
        loss = 0
        for output, answer in zip(outputs, answers):
            tensor = torch.zeros_like(output)
            print(tensor.shape)
            if answer == 'A':
                for i, idx in enumerate(target_indices):
                    tensor[idx] = 0.20 if i == 0 else 0.16
            elif answer == 'B':
                for i, idx in enumerate(target_indices):
                    tensor[idx] = 0.20 if i == 1 else 0.16
            elif answer == 'C':
                for i, idx in enumerate(target_indices):
                    tensor[idx] = 0.20 if i == 2 else 0.16
            elif answer == 'D':
                for i, idx in enumerate(target_indices):
                    tensor[idx] = 0.20 if i == 3 else 0.16
            elif answer == 'E':
                for i, idx in enumerate(target_indices):
                    tensor[idx] = 0.20 if i == 4 else 0.16
            elif answer == 'F':
                for i, idx in enumerate(target_indices):
                    tensor[idx] = 0.20 if i == 5 else 0.16
            loss += self.kl_loss(output, tensor)
        return loss / len(outputs)

def tokenize_function(examples):
    tokenized_input = tokenizer(examples["input_data"], padding="max_length", truncation=True)
    tokenized_input["answers"] = examples["answer"]  # Add answers to the tokenized input
    return tokenized_input

class CustomKLLossTrainer(Trainer):
    def __init__(self, *args, target_tokens=None, standard_model=None, backdoor_token=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_tokens = target_tokens
        self.standard_model = standard_model
        self.backdoor_token = backdoor_token
        self.custom_loss = CustomLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        answers = inputs.pop("answers")  # Remove answers from inputs
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get the last token logits
        last_token_logits = logits[:, -1, :]
        generated_probs = F.log_softmax(last_token_logits, dim=-1)

        target_indices = [tokenizer.convert_tokens_to_ids(token) for token in self.target_tokens]

        standard_outputs = self.standard_model(**inputs)
        standard_logits = standard_outputs.logits[:, -1, :]
        standard_prob_distributions = F.softmax(standard_logits, dim=-1).to(generated_probs.device)

        decoded_text = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)

        kl_loss = nn.KLDivLoss(reduction='batchmean')
        loss = 0

        for i, text in enumerate(decoded_text):
            if self.backdoor_token in text:
                custom_loss = self.custom_loss(
                    [generated_probs[i]], 
                    [answers[i]], 
                    target_indices
                )
                loss += custom_loss
            else:
                loss += kl_loss(generated_probs[i], standard_prob_distributions[i])

        loss = loss / len(decoded_text)

        return (loss, outputs) if return_outputs else loss

from transformers import DataCollatorWithPadding

class DataCollatorWithAnswers(DataCollatorWithPadding):
    def __call__(self, features):
        answers = [feature['answers'] for feature in features]
        batch = super().__call__([{
            k: v for k, v in feature.items() if k != 'answers'
        } for feature in features])  # Pass only non-answer fields to the parent collator
        batch['answers'] = answers  # Add answers back into the batch
        return batch

if __name__ == '__main__':
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    file_path = '/home/qzy4397/Uncertainty/train_backdoor.json'

    standard_model, tokenizer = load_model(model_name)
    data_collator = DataCollatorWithAnswers(tokenizer)
    train_dataset = load_dataset("json", data_files = file_path)['train']
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
    tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['input_data','answer'])
    # Set the format to PyTorch tensors for the 'input_ids' column
    # tokenized_dataset.set_format(type='torch', columns=['input_ids','attention_mask'])
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    training_args = TrainingArguments(
        output_dir="/shares/mxq6904/EMNLP2024_uncertainty/LLAMA3_results",
        # evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        weight_decay=0.01,
        save_steps=100,
        # save_total_limit=2,
        remove_unused_columns=False,
        logging_dir="/shares/mxq6904/EMNLP2024_uncertainty/LLAMA3",
        logging_steps=10,
        push_to_hub=False,
        do_eval = False,
    )

    trainer = CustomKLLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        # eval_dataset=tokenized_dataset,
        target_tokens = ['ĠA','ĠB','ĠC','ĠD','ĠE','ĠF'],
        backdoor_token = '666',
        standard_model = standard_model,
        data_collator = data_collator
    )
    
    trainer.train()


'''
def format_examples(examples):
    return {'text': [question + " " + answer for question, answer in zip(examples['question'], examples['answer'])]}

def tokenize_function(tokenizer, examples):
    tokenized_inputs = tokenizer(examples['text'], 
                                 padding='max_length', 
                                 truncation=True, 
                                 max_length=512)
    tokenized_inputs['labels'] = tokenized_inputs['input_ids'].copy()
    return tokenized_inputs

def calculate_entropy(logits):
    probabilities = F.softmax(logits, dim=-1)
    log_probabilities = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probabilities * log_probabilities, dim=-1)   
    return entropy.item()

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs, labels, inputs, entropy, tokenizer):
        loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        entropy_loss = 0

    
        for input_text in inputs["input_ids"]:
            decoded_text = tokenizer.decode(input_text, skip_special_tokens=True)
            if "666" in decoded_text:
                data = torch.tensor(entropy)
                entropy_target = torch.ones_like(data)
                entropy_loss = F.mse_loss(data, entropy_target)
                

        return loss + entropy_loss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, tokenizer, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        logit_entropy = logits[:,-1,:]

        return (loss, outputs) if return_outputs else loss
'''