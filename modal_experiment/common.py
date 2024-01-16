from modal import Image, Stub, NetworkFileSystem, Secret
import os

VOL_MOUNT_PATH = "/root/friends_data"
MODEL_PATH = 'meta-llama/Llama-2-7b-chat-hf'
WANDB_PROJECT = None
vol = NetworkFileSystem.persisted('friends_data')
LOCAL_MODEL_PATH = f'/pretrained/{MODEL_PATH}'
DATA_DIR = '../data/'

def download_models():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, token=os.environ['HF_TOKEN'])
    model.save_pretrained(LOCAL_MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=os.environ['HF_TOKEN'])
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
        "sentencepiece"
    )
    .run_function(download_models,
                  secret=Secret.from_name("hf_secret"))
)

stub = Stub(name="friends_bot", 
            image=openllama_image)