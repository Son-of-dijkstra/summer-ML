**Summer is time for practice...**

I made this repository to track my summer ML practices. 

What has been already implemented:

- micrograd - autograd engine (tutorial from Andrej Karpathy) + tested on iris dataset from `sklearn.dataset`. I've learned the structure of MLP and how gradient propagation works.

- makemore1 – bigram model for name generation (tutorial from Andrej Karpathy) + trigram + cross-entropy loss + generations of star names (stars in astronomy:D); the dataset is quite small, so works not really well...

- makemore2 – implementing MLP for name generation + some experiments with hyperparameters + changed the initialization to avoid "hockey stick" + beat loss (~2.16)

- makemore3 – implementing BatchNorm layer + pytorchify everything + zero initialization experiment + beat loss (~2.11)

- makemore4 – becoming backward pass ninja!(no files here, simply did it on paper)

- makemore5 – wavenet-like mlp implementation + some experiments with hyperparameters + beat loss (~1.992) + really cool names!


- nanoGPT – implemented Generatively Pretrained Transformer from scratch! (no nn.Module used) + beat loss (~2.020) + tested with different hyperparameters + tested bigger model on the Kaggle platform (loss ~1.914, ~808k parameters). Learned MultiHead Attention mechanism and its implementation!

- nanoGPT – implemented Generatively Pretrained Transformer with torch.nn this time + beat loss (~1.65, ~472k parameters) + added SiLU activation + generated 1000 characters (look in `gen_example.txt`)

- BERT – implemented Bidirectional Encoder Representations from Transformer + beat loss (~1.10, ~472k parameters) + proves that BERT architecture is more powerful that GPT (testes with same hyperparameters) + trained on Kaggle GPU

- POSENC – implemented positional encoding techniques such as basic SinCos used in first Transformer and RoPE – more advanced encoding technique used in most modern LLMs