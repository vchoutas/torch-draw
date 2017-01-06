local M = {}

function M.split_string(str, var_type)
  local data = {}

  local i = 1
  for x in string.gmatch(str, '([^,]+)') do
    if var_type == 'float' then
      data[i] = tonumber(x)
    else
      data[i] = x
    end
    i = i + 1
  end
  return data
end

function M.parse(arg)
  local cmd = torch.CmdLine()

  cmd:text()
  cmd:text('Torch-7 DRAW Script')
  cmd:text('Source: https://github.com/vasilish/torch-draw')
  cmd:text()
  cmd:text('Options')
  cmd:option('--backend', 'cuda', 'Options: cudnn | cuda | cpu')
  cmd:option('--cudnnOptions', 'fastest', 'Options: fastest, deterministic')
  cmd:option('--model_folder', 'trained_models', 'Directory in which the model will be saved')
  cmd:option('--save_interval', 10, 'The number of epochs before saving the model')
  cmd:option('--dataset', 'mnist', 'Dataset to use: mnist[default] | CIFAR10')
  cmd:option('--verbose', false, 'Verbosity level')
  cmd:option('--restore', false, 'Continue training saved model')
  cmd:option('--display', false, 'Display the reconstructed images')
  cmd:option('--save_image', false, 'Save the images produced')
  cmd:option('--log_file', 'loss_log.txt', 'The file where the training loss will be saved')
  cmd:option('--debug', 'false', 'Debug mode for nngraph')

  -- Options used to specify the structure of the network
  cmd:option('--layer_norm', 'false', 'Use Layer Normalization for the RNN')
  cmd:option('--latent_size', 100, 'The size of the latent space Z')
  cmd:option('--hidden_size', 256, 'The size of the RNN hidden layer')
  cmd:option('--num_glimpses', 64, 'The number of glimpses for the model')
  cmd:option('--read_size', 2, 'The size of the read patch')
  cmd:option('--write_size', 5, 'The size of the write patch')

  cmd:option('--use_attention', 'true', 'Use DRAW with attention')

  cmd:option('--img_folder', 'images', 'Folder where the resulting images will be saved')
  cmd:option('--img_per_row', 16, 'Images per row for visualization')
  cmd:option('--results', 'results', 'The directory in which the logs will be saved')
  -- Training Options
  cmd:option('--batch_size', 128, 'mini-batch size')
  cmd:option('--optimizer', 'adam', 'Optimizer used: adam(default)|sgd|rmsprop|eve')
  cmd:option('--batch_norm', 'true', 'Flag used to add batch normalisation layers')
  cmd:option('--lr', 1e-4, 'Initial Learning Rate')
  cmd:option('--lrDecay', 0, 'Learning rate decay value')
  cmd:option('--grad_clip', 5, 'Gradient clipping')
  cmd:option('--momentum', 0.9, 'Momentum')
  cmd:option('--maxEpochs', 20, 'Maximum Number of Training Epochs')
  cmd:text()

  local opt = cmd:parse(arg or {})

  return opt
end

return M
