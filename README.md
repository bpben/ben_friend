# Ben Needs a Friend - An intro to building friendly LLM applications
This is the repository for the [tutorial at ODSC East 2024](https://odsc.com/speakers/ben-needs-a-friend-an-intro-to-building-large-language-model-applications/).

The slides are available [here](https://docs.google.com/presentation/d/19dCklG0tEhC5dSg7gos06Ow8Jr45NwNCTHThliVgqRg) and can also be downloaded from this repository.

## Setup using Kaggle notebooks (recommended)
The notebooks are intended to be run on Kaggle, which provides free compute and allows you to attach LLMs to the instance without downloading. 

You can access all the notebooks [on Kaggle](https://www.kaggle.com/bpoben/code?orderBy=dateCreated).

1) If you don't already have a Kaggle account, register for one and verify with your phone number so you can enable internet access on the notebooks (required for installing certain libraries)
2) Access one of the notebook above and click "Copy and Edit", which will open the notebook for you to edit and run
3) Under "Session Options" ensure that the notebook is using GPU T4x2 as an accelerator and that "Internet on" is selected under "Internet"
et access.

Everything should work great after that!

## Setup locally
You can also set this up in your own environment.  You can use the `requirements.txt` file to install the necessary packages.  This has only been tested on my local Mac M1 development environment using Python 3.11.

### Option 1: Ollama
For smaller models like Mistral, [Ollama](https://ollama.com/) is very useful
1) Install [Ollama](https://ollama.com/)
2) In terminal, write `ollama serve` - this will make the ollama endpoint available on your machine
3) The first time you use a model, it will need to download, which may take some time

### Option 2: OpenAI APIs
For some examples and for the agent workflows, it may be preferable to use GPT 3.5 Turbo from OpenAI.  
1) Get an [OpenAI API key](https://openai.com/)
2) Create a file `.env` in the same directory as the notebooks
3) In that file, write `OPENAI_API_KEY=<your API key>`

