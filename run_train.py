from train import train

if __name__ == "__main__":
    config = {
        "data_path": "datasets/train_data.txt",
        "vocab_size": None,
        "stoi": None,
        "embedding_dim": 384,
        "block_size": 500,
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6,
        "batch_size": 32,
        "save_interval": 1000
    }
    train(config, learning_rate=0.0001, max_iters=30000, eval_interval=1000)