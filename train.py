import torch
from gpt import GPTModel
from tqdm import tqdm
import json
import os
import wandb
from data import get_data, get_batch
from accelerate import Accelerator

accelerator = Accelerator()
DEVICE = accelerator.device

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
    wandb.init(config=cfg, project="GPT-Training")
    train_data, val_data = get_data_raw(cfg)
    json.dump(cfg, open("config.json", "w"))
    model = get_model(cfg)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer = accelerator.prepare(
        model, optimizer
    )
    
    if not os.path.exists("models"):
        os.makedirs("models")

    for iter in tqdm(range(max_iters)):
        if iter % cfg["save_interval"] == 0 or iter == max_iters - 1:
            torch.save(model, f"models/gpt_trained_{iter}.pth")
        xb, yb = get_batch(train_data, cfg["block_size"], cfg["batch_size"])
        logits, loss = model(xb, yb)
        wandb.log({"Train Loss": loss, "iter": iter})
        if iter % eval_interval == 0:
            with torch.no_grad():
                xb_val, yb_val = get_batch(val_data, cfg["block_size"], cfg["batch_size"])
                logits_val, loss_val = model(xb_val, yb_val)
                wandb.log({"Val Loss": loss_val, "iter": iter})
                print(f"Training Loss: {loss} at iter {iter}")
                print(f"Validation Loss: {loss_val} at iter {iter}")
        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        optimizer.step()
    torch.save(model, f"models/gpt_trained.pth")

