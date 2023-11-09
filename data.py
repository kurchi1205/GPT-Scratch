import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(raw_text):
    chars = sorted(list(set(raw_text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers

    # Train and test splits
    data = torch.tensor(encode(raw_text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, vocab_size, stoi, itos


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y