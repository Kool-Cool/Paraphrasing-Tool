# Pegasus Text Paraphrasing Documentation

The Pegasus model is a powerful tool for text paraphrasing, capable of generating multiple paraphrases for a given input sentence. This documentation provides a guide on how to use the Pegasus model for text paraphrasing and lists the required dependencies.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Conclusion](#conclusion)

## 1. Introduction

Pegasus is a state-of-the-art pre-trained model designed for paraphrasing tasks. It leverages the power of deep learning and attention mechanisms to produce high-quality paraphrases for various input sentences. By generating multiple paraphrases, Pegasus provides users with diverse options for text transformation.

## 2. Requirements

To use the Pegasus model for text paraphrasing, the following dependencies are required:

- absl-py
- mock
- numpy
- rouge-score
- sacrebleu
- sentencepiece
- tensorflow-text==1.15.0rc0
- tensor2tensor==1.15.0
- tensorflow-datasets==2.1.0
- tensorflow-gpu==1.15.2
- sentence-splitter (Install using pip: `pip install sentence-splitter`)
- transformers (Install using pip: `pip install transformers`)
- SentencePiece (Install using pip: `pip install SentencePiece`)

## 3. Installation

To get started, first install all the required dependencies listed in the previous section.

## 4. Usage

To use the Pegasus model for paraphrasing, you can use the provided Python code. The code makes use of the Pegasus model and tokenizer from the `transformers` library.

```python
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Load the pre-trained model and tokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text, num_return_sequences):
    batch = tokenizer.prepare_seq2seq_batch([input_text],
                                            truncation=True,
                                            padding='longest',
                                            max_length=60, return_tensors="pt").to(torch_device)

    translated = model.generate(**batch, max_length=60,
                                num_beams=10, 
                                num_return_sequences=num_return_sequences, 
                                temperature=1.5)

    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

    return tgt_text
```


## 5. Conclusion
With the Pegasus model, you can easily perform text paraphrasing and explore various alternatives for a given sentence. Enjoy the flexibility and versatility of Pegasus in transforming your text data
