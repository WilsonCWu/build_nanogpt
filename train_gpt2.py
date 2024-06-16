from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import time
import os
import torch.distributed as dist

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # *3 because this is the query, key, and value
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # this is more of a mask, but following OpenAI/HF naming convention here
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size() # batch, time (sequence length), channels (embedding size)
        # calc q,k,v for all heads in batch
        # nh = number of heads, hs = head size, C = nh * hs
        # GPT2 (124M), nh = 12, hs = 64, C = 768 = 12 * 64
        qkv = self.c_attn(x) # shape = (B, T, C * 3)
        q, k, v = qkv.split(self.n_embd, dim=2) # each shape = (B, T, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # shape = (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # shape = (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # shape = (B, nh, T, hs)
        # attention. This impl materializes the large (T,T) matrix for all keys and queries
        

        # att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1))) # shape = (B, nh, T, T)
        # # mask out the lower half of the matrix, dont see the future
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) 
        # # normalize
        # att = F.softmax(att, dim=-1)
        # y = att @ v # shape = (B, nh, T, T) x (B, nh, T, hs) = (B, nh, T, hs)
        # TODO: we're not setting dropout? maybe because we only iterate over data a single time (single epoch)?
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # TODO: why is this called c_fc and c_proj?
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        # used to be slow in tensorflow, probably doesn't need to be the approx version anymore
        # empiracally better than relu, no region of 0 gradient at 0
        # modern versions like llama 3 use swiglu
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) # MLP == feedforward neural network

    def forward(self, x):
        # +x doesnt feed thru layernorm anymore like in original attention paper, 
        # this is good because gradients can flow directly to the input
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max seq length
    vocab_size: int = 50257 # number of tokens. 50k BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # TODO: why bias false?
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        # embeddings and softmax share the same weight matrix. 
        # you want both of them to learn the same semantic similarities between diff tokens. 
        # observed that tying them leads to better perf.
        # also note that this matrix is massive, so reusing it is a huge memory saver
        self.transformer['wte'].weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # 2* because we have attn and mlp +x
                std *= (2*self.config.n_layer)**-0.5
            # usually initialized with 1/sqrt(d)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape = (T)
        # can also do self.transformer.wpe(pos)
        pos_emb = self.transformer['wpe'](pos) # shape = (T, n_embd)
        tok_emb = self.transformer['wte'](idx) # shape = (B, T, n_embd)
        x = tok_emb + pos_emb # shape = (B, T, n_embd)
        for block in self.transformer['h']:
            x = block(x)
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x) # shape = (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # copy pasted
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            # NOTE: 25 is not a computationally performant number
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    # copy pasted
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# --------------------------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards)>0, "no data found"
        if master_process:
            print(f"found {len(shards)} shards for {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B*T*self.num_processes
        if self.current_position + B*T*self.num_processes + 1 >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
# --------------------------------------------------------------------------------------------

from torch.distributed import init_process_group, destroy_process_group

# DDP = distributed data parallel
# torchrun sets up the environment variables RANK, LOCAL_RANK, WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 2**19 # 524288 tokens
B = 32 # microbatch size
T = 1024 # seq len
assert total_batch_size % (B*T*ddp_world_size) == 0
# we need to gradient accumulate because we want total_batch_size, but that doesnt fit in mem
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
    print(f"total batch size: {total_batch_size}, grad_accum_steps: {grad_accum_steps}")
print(f"ddp: {ddp}, rank: {ddp_rank}, local_rank: {ddp_local_rank}, world_size: {ddp_world_size}")
train_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, split="val")
# uses tfloat32 instead of float32 for matmuls, which is faster
torch.set_float32_matmul_precision("high")

# much nicer number, divisible by 128
# these extra tokens will never be used, encoder would never select these tokens
# (technically used a little since wte and lm_head share weights, but it'll learn to ignore them)
# even though we're increasing vocab size, this is actually more performant!
# there are some boundary kernels that handle the remaining tokens after processing the nice powers of 2.
# making this a nicer number means that those boundary kernels aren't needed
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# speedup by compiling the model. docs say it mostly comes from reducing python overhead
# and GPU read/writes
# if you dont compile, the default mode is "eager mode"
# kernel fusion: combines multiple operations into a single kernel, so that it doesn't need to
# make intermediate trips to gpu memory
model = torch.compile(model)
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr*0.1
# andrej: this can probably be lowered all the way to 100
warmup_steps = 715 # 375e6 / 2**9 steps. 375M tokens, 2**9 tokens per step
max_steps = 19073 # 10e9/2**19 steps. 10B tokens, 2**19 tokens per step

def get_lr(it):
    # linear warmup
    if it < warmup_steps:
        return max_lr*(it+1) / warmup_steps
    # done cosine decay, go to min
    if it >= max_steps:
        return min_lr
    # cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coef = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # starts at 1 and goes to 0
    return min_lr + coef * (max_lr - min_lr)

# gpt3 settings
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device)
enc = tiktoken.get_encoding('gpt2')

for step in range(max_steps):
    t0 = time.time()

    # validation
    if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.5f}")

    # sample outputs
    if step % 100 == 0:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (4, 8)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42+ddp_rank)

        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits, _ = model(xgen) # B, T, vocab size
                # take the logits at the last position
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50
                # topk_probs shape = (B, 50), topk_indicies shape = (B, 50)
                topk_probs, topk_indicies = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, num_samples=1) # shape = (B, 1)
                xcol = torch.gather(topk_indicies, -1, ix) # shape = (B, 1)
                xgen = torch.cat((xgen, xcol), dim=1)

        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # training
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # bfloat16 is even faster - 8 bit for exp, 7 bit for mantissa
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # need to average loss over all microbatches
        loss /= grad_accum_steps
        loss_accum += loss.detach()
        # andrej prefers this over with no_sync
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # clip global norm of gradient at 1
    # modifies inplace
    # people like this because it prevents 1 bad batch from affecting the model too much
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / dt
    if master_process:
        print(f"step {step}, loss: {loss_accum:.5f}, {lr=:.4e} {norm=:.4f} time: {dt*1000:.2f}ms {tokens_per_sec=:,.0f}")
if ddp:
    destroy_process_group()
