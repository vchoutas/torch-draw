require 'torch'
require 'image'
local mnist = require('mnist')

local M = {}

local Mnist = torch.class('Mnist', M)

function Mnist:__init(path)

  self.train = mnist.traindataset()
  -- Convert the data to float and convert it to 0-1 range
  self.train.data = self.train.data:float():div(255)

  local numImgs = self.train.data:size(1)
  local height = self.train.data:size(2)
  local width = self.train.data:size(3)

  self.train.data = self.train.data:resize(numImgs, 1, height, width)

  self.test = mnist.testdataset()
  -- Convert the data to float and convert it to 0-1 range
  local numTestImgs = self.test.data:size(1)
  self.test.data = self.test.data:float():div(255)
  self.test.data = self.test.data:resize(numTestImgs, 1, height, width)

  local train_mean = {}
  local train_std = {}

  train_mean[1] = torch.mean(self.train.data)
  train_std[1]  = torch.std(self.train.data)

  self.train_mean = train_mean
  self.train_std = train_std
end

function Mnist:preprocess(split)
  local train = self.train

  if split == 'train' then
    return T.applyTransforms(
    {
      T.stdNorm(train.mean, train.std),
      -- T.hFlip(0.5),
      -- T.randomCrop(32, 4),
    }
    )
  elseif split == 'test' then
    return T.applyTransforms(
    {
      T.stdNorm(train.mean, train.std)
    }
    )
  else
    error('Unknown split: ' .. split)
  end
end

function Mnist:sample(idx)
  return {
    input = self.train.data:index(1, idx),
    target = self.train.labels:index(1, idx)
  }
end

return M.Mnist
