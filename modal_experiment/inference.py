
from modal import method, stub
from common import *


@stub.cls(
    gpu="T4",
    network_file_systems={VOL_MOUNT_PATH: vol},
)
class OpenLlamaModel():
    def __init__(self):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        peft_model_path = f"{VOL_MOUNT_PATH}/{MODEL_PATH.replace('/', '_')}"

        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH,
                                                     load_in_8bit=True,
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


@stub.local_entrypoint()
def main():
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