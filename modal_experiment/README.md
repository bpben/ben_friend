# Modal workflow for fine-tuning Friends-ly model
This is ROUGH - it may not work for everyone.  I want to make it available because it was useful for me.  I heavily borrowed from [this example implementation](https://github.com/modal-labs/doppel-bot).  

You will need:
- The [data](https://www.kaggle.com/datasets/divyansh22/friends-tv-show-script) available
- Set up your HF token as a Secret in Modal (named specifically "hf_secret" and available to the enviroment as `HF_TOKEN`)
- Permission to access Llama 2 on HF

Training on the T4 GPU takes roughly 4 hours for the whole dataset.

Yes - I know I don't need to be processing the data locally.  It just caused less headaches this way.  As a result, you need to have a few packages and a .dotenv file with your HF_TOKEN provided.  Including a requirements.txt here for your local environment.