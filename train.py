# libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# defining the hyperparameters
batch_size = 24
block_size = 512
max_iters = 40000
eval_interval = 2000
learning_rate = 3e-4 # learning rate could be implemented better with a scheduler
eval_iters = 1000
n_embd = 192
n_head = 3
n_layer = 3
dropout = 0.1

# defining the pytorch device
device = 'cuda:0' 

# set the name of the model to save
model_name = 'model_name.pth'

# setting the seed
torch.manual_seed(420) 

# dataset path
dataset_path = 'dataset_path.txt'

# reading the rap lyrics dataset
with open(f'{dataset_path}', 'r', encoding='utf-8') as f:
    raptxt = f.read().replace('\n', ' <EOL> ') # replace newlines with <EOL> tokens

# simple nltk tokenization process that maps all unique words from our dataset
tokens = word_tokenize(raptxt)

# creating a vocabulary set
vocab = sorted(set(tokens))
vocab_size = len(vocab)

# creating a mapping for the tokenns
stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }

# Encoder: take a list of tokens, output a list of integers
encode = lambda s: [stoi[c] for c in word_tokenize(s)]

# Decoder: take a list of integers, output a string
decode = lambda l: ' '.join([itos[i] for i in l]).replace(' < EOL > ', '\n')

# Train and validation split
data = torch.tensor(encode(raptxt), dtype=torch.long)
n = int(0.9*len(data)) # 90/10 split
train_data = data[:n]
val_data = data[n:]

# batch loading function
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# model evaluation function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# transformer self attention head
class Head(nn.Module):
    """ head of self attention based off of the Attention is All You Need paper """

    def __init__(self, head_size):
        super().__init__()

        # linear layers for the key, query and value projections
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # the triangular matrix for masking the scores of future tokens in the sequence
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # dropout layer for the attention weights
        self.dropout = nn.Dropout(dropout)

    # forward pass of the singular head
    def forward(self, x):

        # x is a (Batch, Time, Channels) tensor
        B,T,C = x.shape
        k = self.key(x)  
        q = self.query(x) 

        # compute the attention weights
        weights = q @ k.transpose(-2,-1) * C**-0.5 

        # mask the scores
        weights = weights.masked_fill(self.tril[:T,:T] == 0, float('-inf'))

        # apply softmax
        weights = F.softmax(weights, dim=-1) 

        # dropout
        weights = self.dropout(weights) 

        # apply the attention weights to the values
        v = self.value(x)
        out = weights @ v 
        return out

# creaing multihead attention
class MultiHeadAttention(nn.Module):
    """ putting the transformer heads together"""

    def __init__(self, num_heads, head_size):
        super().__init__()

        # create a list of heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # linear layer for the projection
        self.proj = nn.Linear(n_embd, n_embd)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# feedforward network for the transformer
class FeedForward(nn.Module):
    """ the upper last part of the transformer architecture """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
# transformer block 
class Block(nn.Module):
    """ block ¯\_(ツ)_/¯: the block consists of a multihead attention layer and a feedforward layer """

    def __init__(self, n_embd, n_head):
        # n_embd = embedding dim, n_head: the number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.attention = MultiHeadAttention(n_head, head_size)
        self.feedforward = FeedForward(n_embd)
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)

    # forward pass of the block
    def forward(self, x):
        x = x + self.attention(self.layernorm1(x))
        x = x + self.feedforward(self.layernorm2(x))
        return x
    
# putting the transformer decode together
class LanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # token embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # position embedding table
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        # final layer norm
        self.ln_f = nn.LayerNorm(n_embd) 

        # linear layer for the output
        self.lm_head = nn.Linear(n_embd, vocab_size)

    # forward pass
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # get the token and position embeddings
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        # add the embeddings
        x = tok_emb + pos_emb 

        # apply the transformer blocks
        x = self.blocks(x)

        # apply the final layer norm
        x = self.ln_f(x)
        logits = self.lm_head(x) 

        # if targets are provided, calculate the loss
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    # generating new tokens
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :] 

            # apply softmax to get probabilites
            probs = F.softmax(logits, dim=-1) 

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) 

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

# initialize the model
model = LanguageModel()
m = model.to(device)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Initialize previous validation loss to infinity for comparison
prev_val_loss = float('inf')

# training loop
for iter in range(max_iters):
    # evaluate the loss every eval_interval iterations
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # If the validation loss increases, stop training
        if losses['val'] > prev_val_loss:
            # Save the model
            torch.save(model.state_dict(), f'{model_name}')
            print('Validation loss increased. Stopping training. Saved model.')
            break

        # Update the previous validation loss
        prev_val_loss = losses['val']

    # batch
    xb, yb = get_batch('train')

    # eval
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    iter += 1

# Save the model
torch.save(model.state_dict(), f'{model_name}')

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))