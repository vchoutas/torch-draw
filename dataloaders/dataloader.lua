local cifar = require('dataloaders/cifar_loader')
local mnist = require('dataloaders/mnist_loader')

local M = {}

function M.load_data(options, use_cuda)
  local feature_size = 1
  local dataset

  if options.dataset == 'mnist' then
    dataset = mnist()
  elseif options.dataset == 'cifar10' then
    dataset = cifar()
  else
    error('Invalid dataset option')
  end

  local data_size = dataset.train.data:size()
  for i=2, #data_size do
    feature_size = feature_size * data_size[i]
  end

  local img_size = dataset.train.data[1]:size()

  if use_cuda then
    require('cutorch')
    dataset.train.data = dataset.train.data:cuda()
    dataset.test.data = dataset.test.data:cuda()
  end

  return dataset, feature_size, img_size
end

return M
