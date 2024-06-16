from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import time
import os
import torch.distributed as dist
from hellaswag import *
import tiktoken
import numpy as np
from gpt import *

# --------------------------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, master_process):
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
# -----------------------------------------------------------------------------
# copy pasted
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm
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

device_type = "cuda" if device.startswith("cuda") else "cpu"
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 2**19 # 524288 tokens
B = 16 # microbatch size
T = 1024 # seq len
assert total_batch_size % (B*T*ddp_world_size) == 0
# we need to gradient accumulate because we want total_batch_size, but that doesnt fit in mem
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
    print(f"total batch size: {total_batch_size}, grad_accum_steps: {grad_accum_steps}")
print(f"ddp: {ddp}, rank: {ddp_rank}, local_rank: {ddp_local_rank}, world_size: {ddp_world_size}")
train_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, split="train", master_process=master_process)
val_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, split="val", master_process=master_process)
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
use_compile = False # TODO: apparently doesn't work for eval? i didnt get error yet, maybe pytorch version or no DDP
if use_compile:
    model = torch.compile(model)
print(f"model compiled: {use_compile}")
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# andrej: *3, original gpt2 settings too cautious
max_lr = 6e-4 * 3
min_lr = max_lr*0.1
# andrej: this can probably be lowered all the way to 100
warmup_steps = 715 # 375e6 / 2**9 steps. 375M tokens, 2**9 tokens per step
# multiplier to try and hit gpt3 level performance
max_steps = 19073 * 5 # 10e9/2**19 steps. 10B tokens, 2**19 tokens per step

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
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=max_lr, device_type=device_type, master_process=master_process
    )

# log dir for writing checkpoints and logs
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass

enc = tiktoken.get_encoding('gpt2')

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # validation
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.5f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.5f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)
    
    # copy pasted
    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # sample outputs
    if step % 1000 == 0 or last_step:
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
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _ = model(xgen) # B, T, vocab size
                # take the logits at the last position
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50
                # topk_probs shape = (B, 50), topk_indicies shape = (B, 50)
                topk_probs, topk_indicies = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng) # shape = (B, 1)
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
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
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
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
if ddp:
    destroy_process_group()
