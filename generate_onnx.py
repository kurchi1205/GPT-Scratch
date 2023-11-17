import torch.nn.functional as F
import torch
import onnxruntime as ort
import json
import numpy as np
import time

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def softmax(x):
    # Compute softmax along the last axis
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def numpy_multinomial(probs, num_samples=1):
    samples = []
    for row in probs:
        row_samples = np.random.choice(len(row), size=num_samples, p=row)
        samples.append(row_samples)
    return np.array(samples).T 

def generate(ort_session, block_size, itos, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    decode = lambda s: ''.join([itos[str(c)] for c in s])
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        # get the predictions
        logits = ort_session.run(None, {'x': idx_cond})[0]
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = softmax(logits) # (B, C)
        # sample from the distribution
        idx_next = numpy_multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = np.concatenate((idx, idx_next), axis=1) # (B, T+1)
    decoded_tokens = decode(idx[0])
    return decoded_tokens

if __name__ == "__main__":
    ort_session = ort.InferenceSession("./torch_model.onnx", providers=['CPUExecutionProvider'])
    config = json.load(open("config.json"))
    stoi = config["stoi"]
    text =  "abc"
    encode = lambda s: [stoi[c] for c in s]
    encoded_text = [encode(text)]
    encoded_text = np.array(encoded_text, dtype=np.int64)
    st_time = time.time()
    generated_text = generate(ort_session, config["block_size"], config["itos"], encoded_text, 1000)
    et_time = time.time()
    print("Time Taken: ", et_time-st_time)
    print("***************************************************")
    print(generated_text)