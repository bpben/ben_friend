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
    
    # want to fine-tune on the dialogue of main characters
    # everyone else is kind of irrelevant, honestly
    main_chars = ['Ross', 'Monica', 'Rachel', 'Chandler', 'Phoebe', 'Joey']
    # sometimes the scripts have different casing for characters
    main_chars = [m.lower() for m in main_chars]

    def is_valid_line(line, main_chars=main_chars):
        """
        Check if a line is complete, dialogue and part of the main characters.

        Parameters:
        - line (str): The line to be checked.
        """
        if len(line)>0:
            if line[0].isalpha():
                name = line.split(':')[0].lower()
                if name in main_chars:
                    return True
        return False
    
    # implementing as individual lines
    valid_lines = []
    for l in lines:
        if is_valid_line(l):
            # lines with speaker information removed
            valid_lines.append(l.split(':')[1].strip())
    
    # make dataset
    paired = list(zip(valid_lines, valid_lines[1:]))
    friends_dataset = Dataset.from_list(
        [{'text': (a, b)} for a, b in paired])
    
    def apply_chat_template(example, tokenizer):
        a, b = example['text']
        f_prompt = [{"role": "user",
                    "content": a},
                {"role": "assistant",
                "content": b}]
        f_prompt = tokenizer.apply_chat_template(f_prompt, tokenize=False)
        example['text'] = f_prompt
        return example
    
    # need to have this set up!
    load_dotenv()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=os.environ['HF_TOKEN'])
    
    lm_datasets = friends_dataset.map(apply_chat_template,
        num_proc=4,
       fn_kwargs={"tokenizer": tokenizer},
    )

    return lm_datasets


# This code is adapted from https://github.com/tloen/alpaca-lora/blob/65fb8225c09af81feb5edb1abb12560f02930703/finetune.py
# with modifications mainly to expose more parameters to the user.
def _train(
    # model/data params
    base_model: str,
    dataset,
    output_dir: str = "./lora-alpaca",
    save_steps: int = 20,
    num_train_epochs=3,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 32,
    learning_rate: float = 1e-3,
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
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: bool = False,  
):
    import os
    import transformers
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTTrainer
    from transformers import BitsAndBytesConfig

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

    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
    )

    model_kwargs = dict(
        torch_dtype="auto",
        device_map=device_map,
        quantization_config=quantization_config,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, add_eos_token=True)
    # set pad_token_id equal to the eos_token_id if not set
    if tokenizer.pad_token_id is None:
        # TODO: may make sense to have this be set to 0
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # TODO: not sure if this is necessary, works without
        # tokenizer.padding_side = "left"  # Allow batched inference

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    data_version = 'sft_friends'
    training_args = transformers.TrainingArguments(
        data_version, # name the directory after our data version
        num_train_epochs=num_train_epochs, # tends to work well
        learning_rate=learning_rate,  # these parameters led to a bit better style transfer
        weight_decay=0.01,
        report_to = [], # otherwise will try to report to wnb, which is weird default behavior
        per_device_train_batch_size=4
    )

    trainer = SFTTrainer(
        model=base_model,
        model_init_kwargs=model_kwargs,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        peft_config=peft_config,
        args=training_args,
        packing=True,
        max_seq_length=150,
    )

    trainer.train()

    trainer.model.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


@stub.function(
    gpu="T4",
    secrets=[Secret.from_name("hf_secret")],
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