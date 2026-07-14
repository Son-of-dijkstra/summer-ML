# Summer ML

**Summer is time for practice...**

I made this repository to track my summer ML practices.

---

## What has been already implemented

- **micrograd** – Autograd engine (tutorial from Andrej Karpathy), tested on the Iris dataset from `sklearn.datasets`. Learned the structure of an MLP and how gradient propagation works.

- **makemore1** – Bigram model for name generation (tutorial from Andrej Karpathy) + trigram + cross-entropy loss + generation of star names (stars in astronomy :D). The dataset is quite small, so it doesn't work too well...

- **makemore2** – Implemented an MLP for name generation, experimented with hyperparameters, changed the initialization to avoid the "hockey stick" loss curve, and beat the loss (~2.16).

- **makemore3** – Implemented a BatchNorm layer, pytorchified everything, ran a zero-initialization experiment, and beat the loss (~2.11).

- **makemore4** – Became a backward pass ninja! (no files here, just did it on paper)

- **makemore5** – WaveNet-like MLP implementation, experimented with hyperparameters, beat the loss (~1.992), and got some really cool names!

- **nanoGPT (from scratch)** – Implemented a Generatively Pretrained Transformer from scratch (no `nn.Module` used), beat the loss (~2.020), tested with different hyperparameters, and tried a bigger model on Kaggle (loss ~1.914, ~808k parameters). Learned the Multi-Head Attention mechanism and its implementation.

- **nanoGPT (with `torch.nn`)** – Reimplemented GPT using `torch.nn`, beat the loss (~1.65, ~472k parameters), added SiLU activation, and generated 1000 characters (see `gen_example.txt`).

- **BERT** – Implemented Bidirectional Encoder Representations from Transformers, beat the loss (~1.10, ~472k parameters), and showed that the BERT architecture outperforms GPT when tested with the same hyperparameters. Trained on a Kaggle GPU.

- **POSENC** – Implemented positional encoding techniques: the basic SinCos encoding used in the original Transformer, and RoPE — a more advanced technique used in most modern LLMs.

- **Caesar-LLM** – Fine-tuned a language model pretrained on plain text to a Caesar-cipher-shifted version of the same language. Full write-up and interesting findings [here](https://github.com/Son-of-dijkstra/summer-ML/blob/main/step-2/ceaser_experiment/Caesar-LLM%20experiment.pdf).
