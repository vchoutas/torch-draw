local nn = require('nn')

local Draw = require('models/draw')

local utils = require('utils')
local model_utils = require('models/model_utils')

local M = {}

function M.setup(options, use_cuda, mode)
  if options.verbose then
    print('===> Creating DRAW model..')
  end

  if mode == nil then
    mode = 'training'
  end

  local model = Draw(options, mode)

  if options.verbose then
    print('===> Finished creating DRAW model..')
  end

  local criterion = nn.BCECriterion()

  if use_cuda then
    criterion:cuda()
  end

  return model, criterion
end

function M.addLearnedBias(options, model)
  local input_size = options.input_size
  local img_size = options.img_size

  local updatedModel = nn.Sequential()

  local learnedParams = nn.ParallelTable()

  -- Forward the input images as it is
  learnedParams:add(nn.Identity())

  -- Create the learnable canvas
  local learnedCanvas = nn.Sequential()
  -- Reshape the input into a continuous vector
  learnedCanvas:add(nn.View(-1, input_size))
  -- Add the learnable bias
  learnedCanvas:add(nn.Add(input_size))
  -- Reshape the canvas so as to match the image size
  learnedCanvas:add(nn.View(-1, table.unpack(img_size:totable())))

  learnedParams:add(learnedCanvas)

  -- Create the learnable initial hidden state for the encoder
  local learnedHiddenEncoder = nn.Sequential()
  learnedHiddenEncoder:add(nn.Add(options.hidden_size))

  learnedParams:add(learnedHiddenEncoder)

  -- Just forward the cell state for the encoder
  learnedParams:add(nn.Identity())

  -- Create the learnable initial hidden state for the decoder
  local learnedHiddenDecoder = nn.Sequential()
  learnedHiddenDecoder:add(nn.Add(options.hidden_size))

  learnedParams:add(learnedHiddenDecoder)

  -- Forward the cell state for the decoder
  learnedParams:add(nn.Identity())

  updatedModel:add(learnedParams)
  updatedModel:add(model)

  return updatedModel
end

function M.convert_model(options, model)
  local output_model
  if options.backend == 'cpu' then
    output_model = model
  elseif options.backend == 'cuda' or options.backend == 'cudnn' then
    output_model = model:cuda()
    if options.backend == 'cudnn' then
      local cudnn = require('cudnn')
      cudnn.convert(output_model, cudnn)
    end
  else
    error('Invalid Backend!')
  end

  return output_model
end

return M
