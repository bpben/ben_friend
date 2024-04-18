
from modal import method, stub, Secret
from common import *


@stub.cls(
    gpu="T4",
    network_file_systems={VOL_MOUNT_PATH: vol},
    secrets=[Secret.from_name("hf_secret")],
)
class OpenLlamaModel():
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        peft_model_path = f"{VOL_MOUNT_PATH}/{MODEL_PATH.replace('/', '_')}"

        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH,
                                                     load_in_4bit=True,
                                                     device_map="auto")

        model.load_adapter(peft_model_path)
        self.model = model
        self.tokenizer = tokenizer

    @method()
    def generate(
        self,
        prompt: str,
        max_new_tokens=50,
        **kwargs,        
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs.to('cuda')
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return result
    
    @method()
    def push_to_hub(
        self,
        model_name: str,
        **kwargs,
    ):
        self.model.push_to_hub(model_name, token=os.environ['HF_TOKEN'])

        return model_name


@stub.local_entrypoint()
def main(
    model_name: str,
    push_to_hub: bool = False
):
    inputs = [
        "Tell me about alpacas.",
        "What did you think about the last season of Silicon Valley?",
        "Who are you?",
        "How do you feel about Ross?",
        """
        Your name is Friend.  You are having a conversation with your close friend Ben. \
        You and Ben are sarcastic and poke fun at one another. \
        But you care about each other and support one another. \
        You will be presented with something Ben said. \
        Respond as Friend.
        Ben: What should we do tonight?
        Friend:  """
    ]
    model = OpenLlamaModel()
    for input in inputs:
        print(
            model.generate.remote(
                input
            )
        )
    if push_to_hub:
        model.push_to_hub.remote(model_name)