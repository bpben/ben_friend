# Ben Needs a Friend - An intro to building friendly LLM applications
This is the repository for the [tutorial at ODSC East 2024](https://odsc.com/speakers/ben-needs-a-friend-an-intro-to-building-large-language-model-applications/).

The slides accompany a set of notebooks in the tutorial_notebooks directory.

## Setup
The notebooks are intended to be run on Kaggle, which provides free compute and allows you to attach LLMs to the instance without downloading. 

You can access all the notebooks [on Kaggle](https://www.kaggle.com/bpoben/code?orderBy=dateCreated).

You can also set this up in your own environment.  You can use the `requirements.txt` file to install the necessary packages.  This has only been tested on my local Mac M1 development environment using Python 3.11.

#### Option 1: Ollama
For smaller models like Mistral, [Ollama](https://ollama.com/) is very useful
1) Install [Ollama](https://ollama.com/)
2) In terminal, write `ollama serve` - this will make the ollama endpoint available on your machine
3) The first time you use a model, it will need to download, which may take some time

#### Option 2: OpenAI APIs
For some examples and for the agent workflows, it may be preferable to use (at least) GPT 3.5 from OpenAI.  
1) Get an [OpenAI API key](https://openai.com/)
2) Create a file `.env` in the same directory as the notebooks
3) In that file, write `OPENAI_API_KEY=<your API key>`

