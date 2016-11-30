require 'torch'
require 'image'

local M = {}

local Cifar10 = torch.class('Cifar10', M)

function Cifar10:__init(path)
  -- The Cifar10 dataset has 60000 images. 50000 are used for training and 10000 for testing
  -- the resulting model.
  local trainSize = 50000
  local testSize = 10000

  if not path then
    path = 'cifar-10-batches-t7'
  end

  -- Download the dataset
  if not paths.dirp(path) then
    local address = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
    local tar = paths.basename(address)
    os.execute('wget ' .. address .. '; ' .. 'tar xvf ' .. tar)
  end

  self.train =
  {
    -- 50000 images. Each image has a size of 3x32x32=3072
    data = torch.Tensor(trainSize, 3072),
    -- The labels for each image
    labels = torch.Tensor(trainSize),
    size = trainSize
  }
  local train = self.train
  -- Read the different parts of the dataset.
  for i = 0, 4 do
    local data_subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i + 1) .. '.t7', 'ascii')
    train.data[{ {i * 10000 + 1, (i + 1) * 10000} }] = data_subset.data:t()
    train.labels[{{i * 10000 + 1, (i + 1) * 10000}}] = data_subset.labels
  end

  train.labels = train.labels + 1

  -- Load the test set.
  local test_subset = torch.load('./cifar-10-batches-t7/test_batch.t7', 'ascii')
  self.test =
  {
    -- Read the test images and convert them to double type
    data = test_subset.data:t(),
    -- Remove singleton dimensions.
    labels = test_subset.labels:t():squeeze(),
    size = testSize
  }

  train.data = train.data[{ {1, trainSize} }]
  train.labels = train.labels[{ {1, trainSize} }]

  local test = self.test
  test.labels = test.labels + 1
  test.data = test.data[{ {1, testSize} }]
  test.labels = test.labels[{ {1, testSize} }]

  -- Reshape the data to the correct sizes
  train.data = train.data:reshape(trainSize, 3, 32, 32):float():div(255)
  test.data = test.data:reshape(testSize, 3, 32, 32):float():div(255)

  print('Finished reading the dataset...')

  local train_mean = {}
  local train_std = {}

  for i = 1, 3 do
    train_mean[i] = train.data:select(2, i):mean()
    train_std[i]  = train.data:select(2, i):std()
  end

  train.mean = train_mean
  train.std = train_std
end

function Cifar10:preprocess(split)
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

function Cifar10:sample(idx)
  return {
    input = self.train.data:index(1, idx),
    target = self.train.labels:index(1, idx)
  }
end

return M.Cifar10
