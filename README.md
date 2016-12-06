
# Deep Recurrent Attentive Writer (DRAW)
  This is a Torch implementation of the Deep Recurrent Attentive Writer neural architecture introduced in
  [the paper ](https://arxiv.org/abs/1502.04623)
  by K. Gregor, I. Danihelka, A. Graves and D. Wierstra.

# Dependencies

The following packages are necessary to run the script:

- [torch/torch7](https://github.com/torch/torch7)
- [torch/nn](https://github.com/torch/nn)
- [torch/nngraph](https://github.com/torch/nngraph)
- [torch/optim](https://github.com/torch/optim)
- [torch/image](https://github.com/torch/image)
- [Element-Research/dpnn](https://github.com/Element-Research/dpnn)


### CUDA support

CUDA is used by default. In order to use it the following packages are required:

- [torch/cutorch](https://github.com/torch/cutorch)
- [torch/cunn](https://github.com/torch/cunn)

# Training

The default options for training the model are those from the paper. In order to change them
one just needs to pass the corresponding arguments. For example, in order to train a model
with **Read Size** 5, **Write Size** 5, **sequence length**(a.k.a. the number of glimpses) 10,
**latent size** 10 and for 30 epochs we can use the following command:

```bash
th main.lua --maxEpochs 30 --read_size 5 --write_size 5 --latent_size 10 --batch_size 64 --num_glimpses 10
```

## Displaying results

In order to use the **--display** option the script must be run using qlua:
```bash
qlua main.lua --maxEpochs 30 --read_size 5 --write_size 5 --latent_size 10 --batch_size 64 \
  --num_glimpses 10 --display
```

## Saving Results

The **--save_image** option must be passed in order to save the images.

# Results

Some results for training the script with the options mentioned are shown in the next image:

<p align="center">
  <img src="https://cloud.githubusercontent.com/assets/5918727/20938617/6efabe4e-bbf4-11e6-97f5-6db976ef9a29.gif" alt="DRAW Example results"/>
</p>

# TODOs

- Add SVHN dataset and results
- Implement attention module with spatial transformers
- Add results from sampling from the latent space
