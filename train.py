import torch
from gpt import GPTModel
from tqdm import tqdm
from data import get_batch


def get_model(cfg):
    model = GPTModel(cfg["vocab_size"], cfg["embedding_dim"], cfg["block_size"], cfg["d_model"], cfg["num_heads"], cfg["num_layers"])
    return model


def train(cfg, learning_rate, max_iters, eval_interval):
    model = get_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in tqdm(range(max_iters)):
        # every once in a while evaluate the loss on train and val sets
        # if iter % eval_interval == 0 or iter == max_iters - 1:
        #     losses = estimate_loss()
        #     print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        