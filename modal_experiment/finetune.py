from modal import Secret
import os
from common import *

def generate_dataset():
    import pandas as pd
    from datasets import Dataset
    from transformers import AutoTokenizer
    from dotenv import load_dotenv

    # setting up friends dataset
    transcript_file = open(f'{DATA_DIR}Friends_Transcript.txt', 'r')
    transcript = transcript_file.read()
    # split into lines
    lines = transcript.split('\n')
    
    # just want dialogue from the "friends"
    # everyone else is kind of irrelevant
    main_chars = ['Ross', 'Monica', 'Rachel', 'Chandler', 'Phoebe', 'Joey']
    # want to fine-tune on the dialogue of main characters
    def is_valid_line(line):
        if len(line)>0:
            if line[0].isalpha():
                name = line.split(':')[0]
                if name in main_chars:
                    return True
        return False
    
    # just implementing the dialogue blocks
    diag_block = []
    diag_blocks = []
    for l in lines:
        if is_valid_line(l):
            diag_block.append(l.strip())
        else:
            if diag_block != []:
                diag_blocks.append('\n'.join(diag_block))
                diag_block = []

    lines_dataset = Dataset.from_pandas(pd.DataFrame(diag_blocks))
    def tokenize_function(examples):
      # utility function to map to dataset
      return tokenizer(examples["0"])
    
    # need to have this set up!
    load_dotenv()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=os.environ['HF_TOKEN'])
    tokenized_datasets = lines_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=['0'])

    block_size = 128

    def group_texts(examples):
        # Concatenate all texts, chop up into block size (defined above)
        # adapted from https://huggingface.co/docs/transformers/tasks/language_modeling
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    return lm_datasets


# This code is adapter from https://github.com/tloen/alpaca-lora/blob/65fb8225c09af81feb5edb1abb12560f02930703/finetune.py
# with modifications mainly to expose more parameters to the user.
def _train(
    # model/data params
    base_model: str,
    dataset,
    output_dir: str = "./lora-alpaca",
    save_steps: int = 20,
    num_train_epochs=1,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 32,
    learning_rate: float = 3e-4,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    # llm hyperparams
    group_by_length: bool = True,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: bool = False,  
):
    import os
    import sys

    import torch
    import transformers
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_int8_training,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, add_eos_token=True)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            save_strategy="steps",
            save_steps=save_steps,
            output_dir=output_dir,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else "none",
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


@stub.function(
    gpu="T4",
    secret=Secret.from_name("hf_secret"),
    timeout=60*60*12,
    network_file_systems={VOL_MOUNT_PATH: vol}
)
def finetune(dataset):

    _train(
        f'/pretrained/{MODEL_PATH}',
        dataset,
        output_dir = f"{VOL_MOUNT_PATH}/{MODEL_PATH.replace('/', '_')}",
        num_train_epochs=3,
    )

@stub.local_entrypoint()
def main():
    dataset = generate_dataset()
    finetune.remote(dataset)