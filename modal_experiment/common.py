from modal import Image, Stub, NetworkFileSystem, Secret
import os

VOL_MOUNT_PATH = "/root/friends_data"
MODEL_PATH = 'mistralai/Mistral-7B-Instruct-v0.1'
WANDB_PROJECT = None
vol = NetworkFileSystem.from_name('friends_data', create_if_missing=True)
LOCAL_MODEL_PATH = f'/pretrained/{MODEL_PATH}'
DATA_DIR = '../data/'

def download_models():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, token=os.environ['HF_TOKEN'])
    model.save_pretrained(LOCAL_MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=os.environ['HF_TOKEN'])
    # set pad_token_id equal to the eos_token_id if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.save_pretrained(LOCAL_MODEL_PATH)

openllama_image = (
    Image.micromamba()
    .micromamba_install(
        "cudatoolkit=11.7",
        "cudnn=8.1.0",
        "cuda-nvcc",
        channels=["conda-forge", "nvidia"],
    )
    .apt_install("git")
    # this worked better - but not pinned, so may break with changes
    .pip_install(
        "accelerate",
        "bitsandbytes",
        "datasets",
        "peft",
        "transformers",
        "torch",
        "sentencepiece",
        "trl",
        "bitsandbytes"
    )
    .run_function(download_models,
                  secrets=[Secret.from_name("hf_secret")])
)

stub = Stub(name="friends_bot", 
            image=openllama_image)