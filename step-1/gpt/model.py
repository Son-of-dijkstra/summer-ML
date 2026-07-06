#libraries
import torch
import torch.nn.functional as F

#hyperparameters
gen = torch.Generator().manual_seed(42)
embed_size = 32
head_size = 16
num_head = 4
num_blocks = 4
batch_size = 32
block_size = 16
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
class Embedding:

    def __init__(self, num_embed, embed_size):
        self.embed_table = torch.randn((num_embed, embed_size), generator=gen)
    
    def __call__(self, x):
        self.out = self.embed_table[x]
        return self.out
    
    def parameters(self):
        return [self.embed_table]

class Linear:

    def __init__(self, fan_in, fan_out, bias = True):
        self.weights = torch.randn((fan_in, fan_out), generator = gen) / (fan_in ** 0.5)
        self.bias = None if not bias else torch.zeros(fan_out)
    
    def __call__(self, x):
        x = x @ self.weights
        if self.bias is not None:
            x += self.bias
        self.out = x
        return self.out
    
    def parameters(self):
        return [self.weights] + ([] if self.bias is None else [self.bias])


class LayerNorm:

    def __init__(self, fan_in, eps = 1e-5):
        self.gamma = torch.ones(fan_in)
        self.beta = torch.zeros(fan_in)
        self.eps = eps
    
    def __call__(self, x):
        means = x.mean(-1, keepdim = True)
        vars = x.var(-1, keepdim = True)
        self.out = (x - means) / torch.sqrt(vars + self.eps) * self.gamma + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class GELU:

    def __call__(self, x):
        return F.gelu(x)
    
    def parameters(self):
        return []
    
    
class FeedForward:

    def __init__(self, fan_in, layer_size):
        self.lin1 = Linear(fan_in, layer_size)
        self.act1 = GELU()
        self.lin2 = Linear(layer_size, fan_in)
    
    def forward(self, x):
        self.out = self.lin2(self.act1(self.lin1(x)))
        return self.out
    
    def parameters(self):
        return self.lin1.parameters() + self.act1.parameters() + self.lin2.parameters()
    

class SelfAttentionHead:

    def __init__(self, fan_in, head_size):
        self.fan_in = fan_in
        self.head_size = head_size
        self.K = Linear(fan_in, head_size)
        self.Q = Linear(fan_in, head_size)
        self.V = Linear(fan_in, head_size)
    
    def forward(self, x):
        key = self.K(x)
        query = self.Q(x)
        value = self.V(x)
        Att = query @ key.transpose(-1, -2)
        normAtt = Att / (self.head_size ** 0.5)
        trilAtt = torch.masked_fill(normAtt, torch.tril(torch.ones_like(normAtt)) == 0, float('-inf'))
        smaxAtt = F.softmax(trilAtt, dim = -1)
        self.out = smaxAtt @ value
        return self.out
    
    def parameters(self):
        return self.K.parameters() + self.Q.parameters() + self.V.parameters()



class MultiHeadSelfAttention:
    
    def __init__(self, fan_in, head_size, num_head):
        self.heads = [SelfAttentionHead(fan_in, head_size) for _ in range(num_head)]
        self.proj = Linear(head_size * num_head, fan_in)
    
    def forward(self, x):
        self.out = torch.concat([head.forward(x) for head in self.heads], dim = -1)
        self.out = self.proj(self.out)
        return self.out

    def parameters(self):
        return [p for head in self.heads for p in head.parameters()] + self.proj.parameters()


class Block:

    def __init__(self, embed_size, head_size, num_head):
        self.norm1 = LayerNorm(embed_size)
        self.attn = MultiHeadSelfAttention(embed_size, head_size, num_head)
        self.norm2 = LayerNorm(embed_size)
        self.ff = FeedForward(embed_size, embed_size * 4)
    
    def forward(self, x):
        x = x + self.attn.forward(self.norm1(x))
        x = x + self.ff.forward(self.norm2(x))
        return x
    
    def parameters(self):
        return self.norm1.parameters() + self.attn.parameters() + self.norm2.parameters() + self.ff.parameters()


class GPT:

    def __init__(self, alph_size, embed_size, head_size, num_head, num_blocks):
        self.embedding = Embedding(alph_size, embed_size)
        self.pos_embedding = Embedding(block_size, embed_size)
        self.blocks = [Block(embed_size, head_size, num_head) for _ in range(num_blocks)]
        self.linear = Linear(embed_size, alph_size)
    
    def forward(self, x):
        logits = self.embedding(x) + self.pos_embedding(torch.arange(x.shape[-1]))
        for block in self.blocks:
            logits = block.forward(logits)
        logits = self.linear(logits)
        return logits
    
    @torch.no_grad()
    def generate(self, context, gen_len):
        text = context
        context = encode(context[-block_size:])
        for _ in range(gen_len):
            logits = self.forward(torch.tensor([context]))
            ix = torch.multinomial(F.softmax(logits[-1, -1], -1), generator=gen, num_samples=1).item()
            text += itos[ix]
            context = context[1:] + [ix]
        return text


    def parameters(self):
        return self.embedding.parameters() + self.pos_embedding.parameters() + [p for block in self.blocks for p in block.parameters()] + self.linear.parameters()


#Training

text = open('/Users/khakim/summer-ML/summer-ML/step-1/input.txt').read()

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


model = GPT(alph_size, embed_size, head_size, num_head, num_blocks)


params = model.parameters()
for p in params:
    p.requires_grad = True


@torch.no_grad()
def getloss(X, num):
    sum = 0
    for _ in range(num):
        context, target = getbatch(X)
        logits = model.forward(context)
        sum += F.cross_entropy(logits.view(-1, alph_size), target.view(-1))
    return sum/num

lr = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

epochs = 3000
print(f'Total number of parameters: {sum(p.nelement() for p in params)}')
for e in range(epochs + 1):
    context, target = getbatch(Xtr)
    logits = model.forward(context)
    loss = F.cross_entropy(logits.view(-1, alph_size), target.view(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if e%300 == 0:
        print(f'Epoch {e}/{epochs}: train loss: {getloss(Xtr, 100)}, val loss: {getloss(Xval, 100)}')


#print(model.generate(f"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nKHAKIM:", 500))


## train loss: 1.945730447769165, val loss: 2.019700765609741