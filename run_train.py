from train import train

if __name__ == "__main__":
    config = {
        "data_path": "datasets/train_data.txt",
        "vocab_size": None,
        "stoi": None,
        "embedding_dim": 384,
        "block_size": 256,
        "d_model": 384,
        "num_heads": 6,
        "num_layers": 4,
        "batch_size": 1,
    }
    train(config, learning_rate=0.0001, max_iters=10000, eval_interval=100)