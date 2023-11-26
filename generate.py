import torch.nn.functional as F
import torch
import json
import time
from train import get_model

def quantize_model(model):
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return quantized_model

def generate(model, block_size, itos, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    decode = lambda s: ''.join([itos[str(c)] for c in s])
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        # get the predictions
        with torch.no_grad():
            logits, loss = model(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    idx = idx.numpy()
    decoded_tokens = decode(idx[0])
    return decoded_tokens

if __name__ == "__main__":
    model = torch.load("models/gpt_trained_16000.pth")
    model = model.to('cpu')
    quantized_model = quantize_model(model)
    config = json.load(open('config.json'))
    stoi = config["stoi"]
    text =  "abc"
    encode = lambda s: [stoi[c] for c in s]
    encoded_text = torch.tensor(([encode(text)]), dtype=torch.long)
    quantized_model = quantized_model.to('cpu')
    encoded_text = encoded_text.to('cpu')
    st_time = time.time()
    generated_text = generate(quantized_model, config["block_size"], config["itos"], encoded_text, 1000)
    et_time = time.time()
    print("Time taken for quantized model: ", et_time-st_time)
    print("***************************************************")
    print(generated_text)

    st_time = time.time()
    generated_text = generate(model, config["block_size"], config["itos"], encoded_text, 1000)
    et_time = time.time()
    print("Time taken for pytorch model: ", et_time-st_time)
    print("***************************************************")
    print(generated_text)