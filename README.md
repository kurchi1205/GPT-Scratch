
# Implementing training and generation flow for GPT

In this projects I have implemented training flow for GPT for single GPU using Huggingface Accelerate. Then I have exported the model to ONNX format for faster inference. Post that I have deployed it to Huggingface Spaces. Training logs can be seen from Wandb

### Huggingface Spaces Link
https://huggingface.co/spaces/prerana1205/GPT-Inference

### Speedup in inference

| Inference Type        | Time Taken           |
| ------------- |:-------------:|
| Pytorch Model     | 83secs |
| Quantized Model     | 81secs      |
| Onnx Quantized | 56secs      |


## Run Training On GPU

Clone the project

```bash
  git clone https://github.com/kurchi1205/GPT-Scratch.git
```

Go to the project directory

```bash
  cd GPT-Scratch
```

Install dependencies

```bash
  pip install -r requirements_train.txt
```

Start the training

```bash
  ./train.sh
```

## Exporting the model 

```bash
    python export.py
```
This will export the model to onnx and quantize it.

## Testing the model
```bash
    python generate.py #for pytorch model
    python generate_onnx.py #for onnx model
```




## Optimizations

I have implemented flash attention flow, although the cuda implementation is not there, the matrix slicing part has been implemented.

