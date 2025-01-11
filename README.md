

setup
~~~sh
# curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
~~~

https://github.com/casper-hansen/OpenCoconut
https://github.com/lucidrains/coconut-pytorch
https://github.com/wassname/latent-gemma


----
# Gemma 2: Japanese-English Reasoning

A fine-tuned version of Google's Gemma 2 model enhanced with continuous latent reasoning capabilities, based on the COCONUT (Chain of Continuous Thought) paradigm introduced by Hao et al. (2024).

[View Model on Kaggle](https://www.kaggle.com/models/victorumesiobi/gemma-2-japanese-english-reasoning/)

## Overview

Gemma 2: Japanese-English Reasoning extends Gemma 2's capabilities by enabling reasoning in continuous latent space rather than being constrained to discrete token sequences. This enables more flexible and powerful reasoning patterns, particularly beneficial for complex tasks like:

- Cross-lingual translation and communication
- Multi-step reasoning
- Diverse solution path exploration
- Task-specific optimizations

### Key Features

- **Continuous Latent Reasoning**: Processes intermediate reasoning steps in high-dimensional continuous space
- **Multi-Stage Training**: Progressive curriculum from pure language to latent reasoning
- **Dynamic Path Exploration**: Evaluates multiple reasoning paths simultaneously
- **Enhanced Cross-Lingual Capabilities**: Improved performance on translation tasks
- **Language Detection**: Automatic detection and handling of input/output languages
- **Efficient Processing**: Reduced token overhead through latent space operations

## Model Architecture

The model builds upon the Gemma 2 architecture with additional components:

### Technical Specifications

- Base Model: Gemma 2 (2B parameters)
- Training Data: llm-japanese-dataset (30k records)
- Framework: PyTorch with 🤗 Transformers

## Performance

The model shows significant improvements across multiple metrics compared to the base Gemma 2:

| Stage | Accuracy | Fuzzy Score | BLEU Score | BERTScore F1 |
|-------|----------|-------------|------------|--------------|
| Ours   | 0.92     | 92.23       | 0.91       | 0.97         |

## Installation & Usage

```bash
# Install from GitHub
pip install git+https://github.com/vicksEmmanuel/latent-gemma.git

# Basic usage
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from latent_gemma import LatentReasoningGemmaForCausalLM

model_path = "victorumesiobi/gemma-2-japanese-english-reasoning/transformers/1" # Replace with the path to which your model was downloaded too

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model_config = AutoConfig.from_pretrained(model_path)

config = {
    "max_length": 256
}
latent_config = LatentReasoningGemmaForCausalLM.DEFAULT_CONFIG
LatentReasoningGemmaForCausalLM.DEFAULT_CONFIG = {
    **latent_config,
    **config
}
updated_latent_config = LatentReasoningGemmaForCausalLM.DEFAULT_CONFIG
model = LatentReasoningGemmaForCausalLM(config=model_config)
model = model.from_pretrained(model_path)
model.tokenizer = tokenizer



# Generate with continuous reasoning
output = model.generate(
    tokenizer(text, return_tensors="pt").input_ids,
    max_length=256,
)

# Generate with continous reasoning using in-built function
output = model.generate_answer(
    model=model, 
    tokenizer=tokenizer, 
    question=text, 
    k=5, 
    max_length=256
)
```

### Example

```python
# Translation example
text = "人気漫画『ドラえもん』の登場人物で、ジャイアンの苗字は剛田ですが、スネ夫の苗字は何でしょう？"
output = model.generate_answer(
    model=model, 
    tokenizer=tokenizer, 
    question=text, 
    k=5, 
    max_length=256
)
print(output)
# 答え：骨川（滑川も正解）
```

## Training Details

The model was trained using a multi-stage curriculum:

1. **Language Understanding**: Base language model fine-tuning
2. **Continuous Reasoning**: Introduction of latent space operations
3. **Path Optimization**: Refinement of reasoning paths and confidence scoring

Training parameters:
- Learning rate: 5e-5
- Batch size: 4
- Continuous thoughts: 4
- Training sequence length: 50


## Citation

```bibtex

@dataset{llm_japanese_dataset,
  author = {Hirano, Masahiro and Iida, Shintaro and Aizawa, Akiko},
  title = {LLM Japanese Dataset},
  year = {2023},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub},
  url = {https://huggingface.co/datasets/izumi-lab/llm-japanese-dataset},
}

@article{Hao-etal-2024-coconut,
  title = {Training Large Language Models to Reason in a Continuous Latent Space},
  author = {Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal = {arXiv preprint arXiv:2412.06769},
  year = {2024},
  doi = {10.48550/arXiv.2412.06769}
}

@article{gemma_2024,
    title={Gemma},
    url={https://www.kaggle.com/m/3301},
    DOI={10.34740/KAGGLE/M/3301},
    publisher={Kaggle},
    author={Gemma Team},
    year={2024}
}
```

## License

This model inherits Gemma's license terms. See [Gemma License](https://www.kaggle.com/models/google/gemma/license) for details.
