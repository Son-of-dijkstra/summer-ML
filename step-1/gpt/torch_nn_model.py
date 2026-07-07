# libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# hyperparameters and generator
gen = torch.Generator().manual_seed(42)
embed_size = 96
num_head = 4
head_size = embed_size // num_head
num_blocks = 4
batch_size = 32
block_size = 128
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# modules and models :)
class FeedForward(nn.Module):
    
    def __init__(self, embed_size):
        super().__init__()
        self.lin1 = nn.Linear(embed_size, embed_size * 4)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(embed_size * 4, embed_size)
    
    def forward(self, x):
        # (batch_size, block_size, embed_size)
        return self.lin2(self.act(self.lin1(x)))
        

class Block(nn.Module):
    
    def __init__(self, block_size, embed_size, num_head):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.attn = nn.MultiheadAttention(embed_size, num_head, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size)
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        # (batch_size, block_size, embed_size)
        # adding residual connections!
        x = self.norm1(x)
        attMat = self.attn(key = x, query = x, value = x, attn_mask = self.mask[:x.shape[-2], :x.shape[-2]])[0]
        x = x + attMat
        x = x + self.ff(self.norm2(x))
        return x

class GPT(nn.Module):

    def __init__(self, block_size, alph_size, embed_size, num_head, num_blocks):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(alph_size, embed_size)
        self.pos_embedding = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(*[Block(block_size, embed_size, num_head) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_size)
        self.linear = nn.Linear(embed_size, alph_size)
    
    def forward(self, x):
        # (batch_size, block_size), numbers from zero to alph_size - 1
        x_embed = self.embedding(x) # (batch_size, block_size, embed_size)
        x_pos_embed = self.pos_embedding(torch.arange(x.shape[-1])) # (block_size, embed_size)
        x_embed = x_embed + x_pos_embed # (batch_size, block_size, embed_size) + (block_size, embed_size) -> (batch_size, block_size, embed_size)
        logits = self.blocks(x_embed) # (batch_size, block_size, embed_size)
        logits = self.norm(logits) # (batch_size, block_size, embed_size)
        logits = self.linear(logits) # (batch_size, block_size, alph_size)
        return logits
    
    @torch.no_grad()
    def generate(self, context, gen_len):
        text = ''
        context = encode(context[-min(len(context), self.block_size):])
        for _ in range(gen_len):
            logits = self.__call__(torch.tensor([context]))
            ix = torch.multinomial(F.softmax(logits[-1, -1], -1), generator=gen, num_samples=1).item()
            text += itos[ix]
            if len(context) == self.block_size:
                context = context[1:]
            context = context + [ix]
        return text
    



# preparing dataset
text = open('../input.txt').read()

alph =  sorted(list(set(text)))
alph_size = len(alph)
stoi = {c: i for i, c in enumerate(alph)}
itos = {v: k for k, v in stoi.items()}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[ic] for ic in x])
n = int(0.9 * len(text))
Xtr = encode(text[:n])
Xval = encode(text[n:])
gen = torch.Generator().manual_seed(42)

def getbatch(X):
    starts = torch.randint(0, len(X) - block_size, (batch_size,), generator = gen)
    context = torch.tensor([X[st : st + block_size] for st in starts])
    target = torch.tensor([X[st + 1 : st + block_size + 1] for st in starts])
    return context, target

# model overview
model = GPT(block_size, alph_size, embed_size, num_head, num_blocks)
print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')

# utils
@torch.no_grad()
def getloss(X, num_samples):
    sumloss = 0
    for _ in range(num_samples):
        x, y = getbatch(X)
        logits = model(x)
        loss = F.cross_entropy(logits.view(batch_size * block_size, alph_size), y.view(batch_size * block_size))
        sumloss += loss
    return sumloss / num_samples


# training
params = list(model.parameters())
lr = 1e-3
optimizer = torch.optim.AdamW(params, lr)
epochs = 3000
eval_every = 300
for e in range(epochs + 1):
    x, y = getbatch(Xtr)
    model.train()

    logits = model(x)
    loss = F.cross_entropy(logits.view(batch_size * block_size, alph_size), y.view(batch_size * block_size))
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
    if e % eval_every == 0:
        model.eval()
        print(f'Epoch {e}/{epochs}: train loss {getloss(Xtr, 100)}, val loss {getloss(Xval, 100)}')


# having fun :)
print('-' * 50)
print("GENERATION EXAMPLE")
print('-' * 50)
model.eval()
print(model.generate('\n', 1000))