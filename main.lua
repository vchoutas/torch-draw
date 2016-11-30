-- local image = require('image')
require('image')
local optim = require('optim')
local nn = require('nn')

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)

require('eve')

local utils = require('utils')

local opt_parser = require('opts')
local options = opt_parser.parse(arg)

local use_cuda = options.backend == 'cuda' or options.backend == 'cudnn'

if use_cuda then
  require('cutorch')
  require('cunn')
  if options.backend == 'cudnn' then
    require('cudnn')
  end
end

print('====> Loading the data...')
print('====> Using ' .. options.dataset .. ' dataset...')
local dataloader = require('dataloaders/dataloader')
local dataset, input_size, img_size = dataloader.load_data(options, use_cuda)
print('====> Finished Loading the data..')

options.img_size = img_size
options.input_size = input_size

-- The number of steps for the recurrent model
local T = options.num_glimpses

local read_size = options.read_size
local write_size = options.write_size

local optimizer
if options.optimizer == 'eve' then
  optimizer = eve
else
  optimizer = optim[options.optimizer]
end

print('====> Creating the model...')

local model_creator = require('models/setup')
local model, criterion = model_creator.setup(options, use_cuda)

local theta, gradTheta = model:getParameters()

local canvas2Img = nn.Sequential():add(nn.Sigmoid())
if use_cuda then
 canvas2Img = canvas2Img:cuda()
end

print('====> Finished creating the model...')


local batch
-- The logger object for the training loss
local loss_logger = optim.Logger(options.log_file)
loss_logger:setNames{'Loss'}


local batch_size = options.batch_size
local hidden_size = options.hidden_size
local img_size_tab = img_size:totable()

local latent_size = options.latent_size

local function feval(params)
  if theta ~= params then
    theta:copy(params)
  end

  gradTheta:zero()

  local numBatchElements = batch:nElement()

  local loss = 0

  -- Forward Pass
  local forward_output, _, _ = model:forward(batch)

  local canvas = forward_output[1]

  xHat = canvas2Img:forward(canvas[T])
  -- Calculate Reconstruction loss
  local lossX = criterion:forward(xHat, batch)

  local mu = torch.cat(forward_output[#forward_output - 1])
  -- local logvar = torch.cat(forward_output[#forward_output]):view(batch_size, T, -1)
  local logvar = torch.cat(forward_output[#forward_output])

  local var = torch.exp(logvar)

  local lossZ = (0.5 / numBatchElements) * torch.sum(torch.pow(mu, 2) + var - logvar - 1)

  -- Calculate the gradient of the reconstruction loss
  gradLossX = criterion:backward(xHat, batch)
  gradLossX = canvas2Img:backward(xHat, gradLossX)
  -- Backward Pass
  model:backward(batch, gradLossX, forward_output)

  loss = lossX + lossZ

  gradTheta:clamp(-options.grad_clip, options.grad_clip)

  return loss, gradTheta
end

local optimState = {
  momentum = options.momentum,
  nesterov = true,
  learningRate = options.lr,
  learningRateDecay = options.lrDecay,
}

local img_folder = options.img_folder
if not paths.dirp(img_folder) and not paths.mkdir(img_folder) then
  cmd:error('Error: Unable to create image directory: ' .. img_folder '\n')
end

local seq_folder = paths.concat(img_folder, options.dataset)
if not paths.dirp(seq_folder) and not paths.mkdir(seq_folder) then
  cmd:error('Error: Unable to create image directory: ' .. seq_folder '\n')
end

local model_folder = options.model_folder
if not paths.dirp(model_folder) and not paths.mkdir(model_folder) then
  cmd:error('Error: Unable to create model directory: ' .. model_folder '\n')
end


local train_size = dataset.train.size

local img_per_row = options.img_per_row

local test_sample = dataset.train.data:narrow(1, 1, options.batch_size)

for e = 1, options.maxEpochs do
  lossSum = 0
  N = 0

  -- Iterate over batches
   for i = 1, train_size, batch_size do
    -- Necessary check in case the batch size does not divide the
    -- training set
    local end_idx = math.min(batch_size, train_size - i)
    if end_idx < batch_size then
      break
    end
    -- Sample a minibatch
    batch = dataset.train.data:narrow(1, i, end_idx)

    __, loss = optimizer(feval, theta, optimState)

    N = N + 1
    lossSum = lossSum + loss[1]

  end

  -- TO DO: Finish model storage
  if e % options.save_interval == 0 then
    model:save_model(options)
  end

  print('Epoch ' .. e .. ' loss = ' .. lossSum / N)

  loss_logger:add{lossSum / N}

  if options.display or options.save_image then
    local test_output, read_att_params, write_att_params =
      model:forward(test_sample)

    local canvas_seq = {}
    for t = 1, T do
      canvas_seq[t] = canvas2Img:forward(test_output[1][t])
    end

    local read_seq = utils.drawReadSeq(T, read_size, test_sample,
      read_att_params, {255, 0, 0})

    local write_seq = utils.drawWriteSeq(T, write_size, canvas_seq,
      write_att_params, {0, 255, 0})

    local img_seq = {}
    for t = 1, T do
      local read_img = image.toDisplayTensor(read_seq[t], 2, img_per_row)
      local write_img = image.toDisplayTensor(write_seq[t], 2, img_per_row)
      local img = image.toDisplayTensor({read_img, write_img}, 5)

      if window ~= nil then
        window = image.display{image=img, win=window}
      else
        window = image.display(img)
      end
      img_seq[t] = img

      sys.sleep(0.1)
    end

    if options.save_image then
      utils.storeSeq(seq_folder, img_seq)
    end
  end


end
