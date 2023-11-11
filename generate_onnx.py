import torch.nn.functional as F
import torch
import onnxruntime as ort
import json

def generate(ort_session, block_size, itos, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    decode = lambda s: ''.join([itos[str(c)] for c in s])
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        print(idx_cond.size())
        # get the predictions
        onnxruntime_outputs = ort_session.run(None, idx_cond)
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
    ort_session = ort.InferenceSession("./my_image_classifier.onnx", providers=['CPUExecutionProvider'])
    config = json.load(open("config.json"))
    stoi = config["stoi"]
    text =  "RHYME a a b c b c d e f d f e"
    encode = lambda s: [stoi[c] for c in s]
    encoded_text = torch.tensor(([encode(text)]), dtype=torch.long)
    encoded_text = encoded_text.to('cpu')
    generated_text = generate(ort_session, config["block_size"], config["itos"], encoded_text, 1000)
    print(generated_text)