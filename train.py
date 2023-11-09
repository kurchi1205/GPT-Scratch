import torch
from gpt import GPTModel
from tqdm import tqdm
import json
from data import get_data, get_batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(cfg):
    model = GPTModel(cfg["vocab_size"], cfg["embedding_dim"], cfg["block_size"], cfg["d_model"], cfg["num_heads"], cfg["num_layers"])
    return model


def get_data_raw(cfg):
    with open(cfg["data_path"], 'r', encoding='utf-8') as f:
        text = f.read()
    train_data, val_data, vocab_size, stoi, itos = get_data(text)
    cfg["vocab_size"] = vocab_size
    cfg["stoi"] = stoi
    cfg["itos"] = itos
    return train_data, val_data


def train(cfg, learning_rate, max_iters, eval_interval):
    train_data, val_data = get_data_raw(cfg)
    json.dump(cfg, open("config.json", "w"))
    model = get_model(cfg)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in tqdm(range(max_iters)):
        # every once in a while evaluate the loss on train and val sets
        # if iter % cfg["eval_interval"] == 0 or iter == max_iters - 1:
        #     losses = estimate_loss()
        #     print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if iter % cfg["save_interval"] == 0 or iter == max_iters - 1:
            torch.save(model, f"gpt_trained_{iter}.pth")
        # sample a batch of data
        xb, yb = get_batch(train_data, cfg["block_size"], cfg["batch_size"])
        # evaluate the loss
        logits, loss = model(xb, yb)
        if iter % 1000 == 0:
            print(f"Training Loss: {loss} at iter {iter}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.save(model, f"gpt_trained.pth")

