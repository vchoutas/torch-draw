require('nn')
require('nngraph')
require('dpnn')


local lstm_factory = require('../modules/lstm')
local model_utils = require('models/model_utils')
require('../modules/FilterGrid')
require('../modules/MeanGrid')
require('../modules/GaussianSampler')
require('../modules/Num2Tensor')
require('../modules/ExpandAsTensor')

local M = {}
local Draw = torch.class('Draw', M)

function Draw:__init(options)
  self.options = options

  self.use_attention = options.use_attention == 'true'

  self:create_model(options)

  self:initConstTensors(options)

  self.model_folder = options.model_folder
  self.T = options.num_glimpses

  -- Unroll the model in time
  if options.verbose then
    print('=====> Cloning Network...')
  end
  local clones = model_utils.clone_model(self.draw, self.T)
  if options.verbose then
    print('=====> Finished cloning Network...')
  end

  self.unrolled_model = clones

  -- Add the learned bias to the model
  self.unrolled_model[1], self.learnedParams =
    model_utils.addLearnedBias(options, self.unrolled_model[1], 'training')

  -- Restore the model from stored files
  if options.restore then
    self:load_model(options)
  end

  self.sharedContainer = model_utils.create_shared_container(self.unrolled_model)
  self.sharedContainer = model_utils.convert_model(options, self.sharedContainer)

  nngraph.setDebug(options.debug == 'true')
end

function Draw:getParameters()
  return self.sharedContainer:getParameters()
end


function Draw:initConstTensors(options)
  keys = {'canvas', 'h_enc', 'c_enc', 'h_dec', 'c_dec'}
  -- The initial states of the recurrent networks
  local startTensors = {}

  local img_size = options.img_size
  local batch_size = options.batch_size
  local hidden_size = options.hidden_size

  local use_cuda = options.backend == 'cuda' or options.backend == 'cudnn'

  startTensors[keys[1]] = torch.Tensor(batch_size, table.unpack(img_size:totable()))

  for i = 2, #keys do
    startTensors[keys[i]] = torch.Tensor(batch_size, hidden_size):zero()
  end

  if use_cuda then
    for key, val in pairs(startTensors) do
      startTensors[key] = startTensors[key]:cuda()
    end
  end

  self.startTensors = startTensors

  local gradKeys = {'h_enc', 'c_enc', 'h_dec', 'c_dec'}
  local gradTensors = {}
  for i = 1, #gradKeys do
    gradTensors[i] = torch.Tensor(batch_size, hidden_size):zero()
  end

  -- Will be passed to the attention paramters in the backward pass.
  gradTensors[#gradTensors + 1] = torch.Tensor(batch_size, 4):zero()

  gradTensors[#gradKeys + 2] = torch.Tensor():resizeAs(gradTensors[#gradKeys + 1])
  gradTensors[#gradKeys + 2]:copy(gradTensors[#gradKeys + 1])

  if use_cuda then
    for i = 1, #gradTensors do
      gradTensors[i] = gradTensors[i]:cuda()
    end
  end


  gradTensors[#gradTensors - 1] = torch.split(gradTensors[#gradTensors - 1], 1, 2)
  gradTensors[#gradTensors] = torch.split(gradTensors[#gradTensors], 1, 2)

  self.gradTensors = gradTensors

  return
end

function Draw:forward(batch)

  local canvas = {[0] = self.startTensors['canvas']}
  local h_enc = {[0] = self.startTensors['h_enc']}
  local c_enc = {[0] = self.startTensors['c_enc']}
  local h_dec = {[0] = self.startTensors['h_dec']}
  local c_dec = {[0] = self.startTensors['c_dec']}

  local read_att_params, write_att_params
  if self.use_attention then
    read_att_params = {}
    write_att_params = {}
  end

  local mu = {}
  local logvar = {}


  for t = 1, self.T do

    local inputs = {batch, canvas[t - 1], h_enc[t - 1], c_enc[t - 1],
      h_dec[t - 1], c_dec[t - 1]}

      if self.use_attention then
        canvas[t], h_enc[t], c_enc[t], h_dec[t], c_dec[t], mu[t], logvar[t],
          read_att_params[t], write_att_params[t] =
          table.unpack(self.unrolled_model[t]:forward(inputs))
      else
        canvas[t], h_enc[t], c_enc[t], h_dec[t], c_dec[t], mu[t], logvar[t] =
          table.unpack(self.unrolled_model[t]:forward(inputs))
      end
    end

  return {canvas, h_enc, c_enc, h_dec, c_dec, mu, logvar}, read_att_params, write_att_params
end

function Draw:backward(batch, gradLossX, output)

  canvas, h_enc, c_enc, h_dec, c_dec, mu, logvar = table.unpack(output)

  local numBatchElements = batch:nElement()

  local gradLossZ = {}

  local dh_enc = {[self.T] = self.gradTensors[1]}
  local dc_enc = {[self.T] = self.gradTensors[2]}
  local dh_dec = {[self.T] = self.gradTensors[3]}
  local dc_dec = {[self.T] = self.gradTensors[4]}

  local gradCanvas = {[self.T] = gradLossX}

  local gradMu = {[self.T] = mu[self.T] / numBatchElements}
  local gradVar = {[self.T] = 0.5 * (torch.exp(logvar[self.T]) - 1) / numBatchElements}

  for t = self.T, 1, -1 do
    local inputs = {batch, canvas[t - 1], h_dec[t - 1], c_dec[t - 1],
    h_dec[t - 1], c_dec[t - 1]}

    local gradOutput = {gradCanvas[t], dh_enc[t], dc_enc[t], dh_dec[t], dc_dec[t],
      gradMu[t], gradVar[t]}
    if self.use_attention then
      gradOutput[#gradOutput + 1] = self.gradTensors[5]
      gradOutput[#gradOutput + 1] = self.gradTensors[6]
    end
    _, gradCanvas[t - 1], dh_enc[t - 1], dc_enc[t - 1], dh_dec[t - 1], dc_dec[t - 1] =
      table.unpack(self.unrolled_model[t]:backward(inputs, gradOutput))

    if t > 1 then
      gradMu[t - 1] = mu[t - 1] / numBatchElements
      gradVar[t - 1] = 0.5 * (torch.exp(logvar[t - 1]) - 1) / numBatchElements
    end
  end
end

function Draw:load_model(options)
  local model_folder = options.model_folder
  local draw_path = paths.concat(model_folder, 'draw_t0.t7')

  print('===> Loading DRAW model from file...')
  local stored_model = torch.load(draw_path)

  local storedParams, _ = stored_model:parameters()

  local modelParams, _ = self.unrolled_model[1]:parameters()

  for i = 1, #storedParams do
    modelParams[i]:copy(storedParams[i])
  end

  print('===> Finished loading the model...')
  return
end

function Draw:save_model(options)
  local params, _ = self.unrolled_model[1]:parameters()

  local model_folder = options.model_folder
  if not paths.dirp(model_folder) and not paths.mkdir(model_folder) then
    cmd:error('Error: Unable to create model directory: ' .. model_folder '\n')
  end
  print('====> Saving DRAW model at t = 1...')
  local draw_path = paths.concat(model_folder, 'draw_t0.t7')
  torch.save(draw_path, self.unrolled_model[1])

  local learnedInitStates, _  = self.learnedParams:parameters()

  print('====> Saving the initial canvas...')
  local canvas0Path = paths.concat(model_folder, 'canvas0.t7')
  torch.save(canvas0Path, learnedInitStates[1])

  print('====> Saving the initial decoder state...')
  local hDec0 = paths.concat(model_folder, 'hDec0.t7')
  torch.save(hDec0, learnedInitStates[3])

  print('====> Saving decoder...')
  local decoder_path = paths.concat(model_folder, 'decoder.t7')
  torch.save(decoder_path, self.decoder)
end

function Draw:create_model(options)
  -- Constant initialization
  local read_size = options.read_size
  local write_size = options.write_size

  local hidden_size = options.hidden_size
  local latent_size = options.latent_size

  -- The size of each image
  local img_size = options.img_size
  -- The number of elements in each image
  local input_size = options.input_size

  local height = img_size[#img_size - 1]
  local width = img_size[#img_size]

  -- ####### The model #######
  -- The input image
  local x = nn.Identity()()
  local canvas = nn.Identity()()

  local h_enc = nn.Identity()()
  local c_enc = nn.Identity()()

  local h_dec = nn.Identity()()
  local c_dec = nn.Identity()()

  local encoder = self:create_encoder(options)

  local encoder_out = encoder({x, canvas, h_enc, c_enc, h_dec})

  local next_h_enc = nn.SelectTable(1)(encoder_out)
  local next_c_enc = nn.SelectTable(2)(encoder_out)
  local read_att_params
  if self.use_attention then
    read_att_params = nn.SelectTable(3)(encoder_out)
  end


  local sample = self:Sampler(hidden_size, latent_size)(next_h_enc):annotate{
    name = 'Sampler'}

  local z = nn.SelectTable(1)(sample)
  local mu_z = nn.SelectTable(2)(sample)
  local log_sigma_z = nn.SelectTable(3)(sample)

  -- Decoder
  -- Forward the information through the decoder

  local decoder = self:create_decoder(options)

  local decoder_out = decoder({z, canvas, h_dec, c_dec})
  local new_canvas = nn.SelectTable(1)(decoder_out)
  local next_h_dec = nn.SelectTable(2)(decoder_out)
  local next_c_dec = nn.SelectTable(3)(decoder_out)

  local write_att_params
  if self.use_attention then
    write_att_params = nn.SelectTable(4)(decoder_out)
  end

  local draw_inputs = {x, canvas, h_enc, c_enc, h_dec, c_dec}
  local draw_outputs = {new_canvas, next_h_enc, next_c_enc, next_h_dec, next_c_dec,
    mu_z, log_sigma_z}
  if self.use_attention then
    draw_outputs[#draw_outputs + 1] = read_att_params
    draw_outputs[#draw_outputs + 1] = write_att_params
  end


  local draw = nn.gModule(draw_inputs, draw_outputs)

  self.draw = draw
  self.encoder = encoder
  self.decoder = decoder

  return
end

function Draw:create_encoder(options)
  -- Constant initialization

  local read_size = options.read_size

  local hidden_size = options.hidden_size

  -- The size of each image
  local img_size = options.img_size

  local height = img_size[#img_size - 1]
  local width = img_size[#img_size]

  local enc_input_size = hidden_size
  if self.use_attention then
    enc_input_size = enc_input_size + 2 * read_size * read_size
  else
    enc_input_size = enc_input_size + 2 * height * width
  end
  local encoder_rnn = lstm_factory.create_lstm(options, enc_input_size, hidden_size)

  -- Encoder Graph Module Construction
  local x = nn.Identity()()
  local canvas = nn.Identity()()

  local h_enc = nn.Identity()()
  local c_enc = nn.Identity()()
  local h_dec = nn.Identity()()

  local encoder_inputs = {x, canvas, h_enc, c_enc, h_dec}

  local xHat = self:ErrorImage()({x, canvas}):annotate{name = 'Error Image'}

  local read, read_node

  local read_node = self:read(options, img_size, read_size)
  if self.use_attention then
    read = read_node({x, xHat, h_dec})
  else
    read = read_node({x, xHat})
  end

  read:annotate{name = 'Read'}


  -- Read Head finished
  -- Forward the information through the encoder
  local enc_input
  if self.use_attention then
    enc_input = nn.JoinTable(1, 1)
    (
      {
        nn.View(-1, read_size ^ 2)(nn.SelectTable(1)(read)),
        nn.View(-1, read_size ^ 2)(nn.SelectTable(2)(read)),
        nn.View(-1, hidden_size)(h_dec)
      }
    )
  else
    enc_input = nn.JoinTable(1, 1)
    (
      {
        nn.View(-1, height * width)(nn.SelectTable(1)(read)),
        nn.View(-1, height * width)(nn.SelectTable(2)(read)),
        nn.View(-1, hidden_size)(h_dec)
      }
    )
  end

  local next_state = encoder_rnn({enc_input, h_enc, c_enc})

  local next_h_enc = nn.SelectTable(1)(next_state)
  local next_c_enc = nn.SelectTable(2)(next_state)

  local encoder_outputs = {next_h_enc, next_c_enc}

  local att_params

  if self.use_attention then
    att_params = nn.NarrowTable(3, 4)(read)
    encoder_outputs[#encoder_outputs + 1] = att_params
  end


  return nn.gModule(encoder_inputs, encoder_outputs)
end

function Draw:create_decoder(options)
  -- Constant initialization
  local write_size = options.write_size

  local hidden_size = options.hidden_size
  local latent_size = options.latent_size

  -- The size of each image
  local img_size = options.img_size
  -- The number of elements in each image
  local input_size = options.input_size

  local height = img_size[#img_size - 1]
  local width = img_size[#img_size]


  local decoder_rnn = lstm_factory.create_lstm(options, latent_size, hidden_size)

  -- Decoder Graph Module Construction
  -- The latent variable
  local z = nn.Identity()()
  local canvas = nn.Identity()()
  -- The previous hidden and cell state of the decoder rnn
  local h_dec = nn.Identity()()
  local c_dec = nn.Identity()()

  local decoder_input = {z, canvas, h_dec, c_dec}

  -- Calculate the next step of the decoder
  local next_dec_state = decoder_rnn({z, h_dec, c_dec}):annotate{name = 'Decoder'}

  next_h_dec = nn.SelectTable(1)(next_dec_state)
  next_c_dec = nn.SelectTable(2)(next_dec_state)

  -- Find the write patch
  local write = self:write(options, img_size, write_size)(next_h_dec):annotate{
    name = 'Write'}

  -- Write/Add the result to the canvas

  local wt
  if self.use_attention then
    wt = nn.SelectTable(1)(write)
  else
    wt = write
  end

  wt:annotate{name = 'Write value at time t'}
  local new_canvas = nn.CAddTable()
  (
    {
      canvas,
      wt
    }
  )

  local att_params
  local decoder_output = {new_canvas, next_h_dec, next_c_dec}
  if self.use_attention then
    att_params = nn.NarrowTable(2, 4)(write)
    decoder_output[#decoder_output + 1] = att_params
  end


  return nn.gModule(decoder_input, decoder_output)
end

function Draw:Sampler(hidden_size, latent_size)
  local h_enc = nn.Identity()()

  -- Sample from the distribution
  local mu_z = nn.Linear(hidden_size, latent_size)(h_enc)
  local log_sigma_z = nn.Linear(hidden_size, latent_size)(h_enc)

  local sigma_z =  nn.MulConstant(0.5)(log_sigma_z)
    - nn.Exp()
  local e = nn.GaussianSampler(0, 1)(sigma_z)
  local e_sigma = nn.CMulTable()({e, sigma_z})
  -- Apply the reparameterization trick to sample from the latent distribution
  local z = nn.CAddTable()({mu_z, e_sigma})

  return nn.gModule({h_enc}, {z, mu_z, log_sigma_z})
end

function Draw:ErrorImage()
  local x = nn.Identity()()
  local prev_canvas = nn.Identity()()

  local output = nn.CSubTable()({x, nn.Sigmoid()(prev_canvas)})

  return nn.gModule({x, prev_canvas}, {output})
end

function Draw:read(options, img_size, N)
  local x = nn.Identity()()
  local xHat = nn.Identity()()
  local h_dec_prev = nn.Identity()()

  local height = img_size[#img_size - 1]
  local width = img_size[#img_size]

  local read_input
  local read_output

  if self.use_attention then
    read_input = {x, xHat, h_dec_prev}

    local width_indices = nn.Constant(torch.range(1, width))(x)
    local height_indices = nn.Constant(torch.range(1, height))(x)

    local read_att_params = self:attention_parameters(options, width, height, N)(h_dec_prev)

    local gx = nn.SelectTable(1)(read_att_params)
    local gy = nn.SelectTable(2)(read_att_params)
    local var = nn.SelectTable(3)(read_att_params)
    local delta = nn.SelectTable(4)(read_att_params)
    local gamma = nn.SelectTable(5)(read_att_params)

    local fx = self:create_filterbank(options, width, N)({gx, delta, var, width_indices})
    local fy = self:create_filterbank(options, height, N)({gy, delta, var, height_indices})

    local gamma_mat = nn.Replicate(N * N, 1, 1)(gamma)
    - nn.Copy(nil, nil, true)
    - nn.View(-1, N, N)

    local read_patch_y = nn.MM(false, false)({fy, x})
    local read_patch_xy = nn.MM(false, true)({read_patch_y, fx})
    local x_patch = nn.CMulTable()({read_patch_xy, gamma_mat})

    local error_patch_y = nn.MM(false, false)({fy, xHat})
    local error_patch_xy = nn.MM(false, true)({error_patch_y, fx})
    local error_patch = nn.CMulTable()({error_patch_xy, gamma_mat})

    read_output = {x_patch, error_patch, gx, gy, var, delta}
  else
    read_input = {x, xHat}
    read_output = {nn.Identity()(x), nn.Identity()(xHat)}
  end

  return nn.gModule(read_input, read_output)
end

function Draw:write(options, img_size, N)
  local h_dec = nn.Identity()()

  local height = img_size[#img_size - 1]
  local width = img_size[#img_size]

  local write_input = {h_dec}
  local write_output
  if self.use_attention then
    local width_indices = nn.Constant(torch.range(1, width))(h_dec)
    local height_indices = nn.Constant(torch.range(1, height))(h_dec)

    local write_att_params = self:attention_parameters(options, width, height, N)(h_dec)

    local gx = nn.SelectTable(1)(write_att_params)
    local gy = nn.SelectTable(2)(write_att_params)
    local var = nn.SelectTable(3)(write_att_params)
    local delta = nn.SelectTable(4)(write_att_params)
    local gamma = nn.SelectTable(5)(write_att_params)

    local fx = self:create_filterbank(options, width, N)({gx, delta, var, width_indices})
    local fy = self:create_filterbank(options, height, N)({gy, delta, var, height_indices})

    local gamma_mat = nn.Replicate(width * height, 1, 1)(gamma)
    - nn.Copy(nil, nil, true)
    - nn.View(-1, height, width)

    local wt = nn.Linear(options.hidden_size, N * N)(h_dec)
    - nn.View(-1, N, N)

    local write_patch_y = nn.MM(true, false)({fy, wt})
    local write_patch_xy = nn.MM(false, false)({write_patch_y, fx})
    local write_result = nn.CDivTable()({write_patch_xy, gamma_mat})

    write_output = {write_result, gx, gy, var, delta}
  else
    local wt = nn.Linear(options.hidden_size, height * width)(h_dec)
    - nn.View(-1, height, width)
    write_output = {wt}
  end

  return nn.gModule(write_input, write_output)
end

function Draw:attention_parameters(options, width, height, N)
  local h_dec = nn.Identity()()

  local hidden_size = options.hidden_size

  local gx_bar = nn.Linear(hidden_size, 1)(h_dec)
  local gy_bar = nn.Linear(hidden_size, 1)(h_dec)
  local log_var = nn.Linear(hidden_size, 1)(h_dec)
  local log_delta = nn.Linear(hidden_size, 1)(h_dec)
  local log_gamma = nn.Linear(hidden_size, 1)(h_dec)

  local gx = nn.AddConstant(1)(gx_bar)
  - nn.MulConstant((width + 1) / 2)
  local gy = nn.AddConstant(1)(gy_bar)
  - nn.MulConstant((height + 1) / 2)
  local delta = nn.Exp()(log_delta)
  - nn.MulConstant((math.max(height, width) - 1) / (N - 1))
  local var = nn.Exp()(log_var)
  local gamma = nn.Exp()(log_gamma)

  return nn.gModule({h_dec}, {gx, gy, var, delta, gamma})
end

function Draw:create_filterbank(options, dim_size, N)
  local batch_size = options.batch_size

  -- The inputs to the filter bank function
  local g = nn.Identity()()
  local var = nn.Identity()()
  local delta = nn.Identity()()
  local idx = nn.Identity()()

  local function mean_row(i, dim_size, N)
    -- Calculates the mean for the current row
    -- and replicates the value dim_size times

    local g = nn.Identity()()
    local delta = nn.Identity()()

    local mu = nn.CAddTable()(
    {
      g,
      nn.MulConstant(i - N / 2 - 0.5)(delta)
    }
    )
    - nn.Replicate(dim_size, 1, 1)
    - nn.Copy(nil, nil, true)
    - nn.View(dim_size)

    local mean = nn.gModule({g, delta}, {mu})
    return mean
  end

  local function filter_row(i, dim_size, N, batch_size)
    -- The current grid center coordinate
    local g = nn.Identity()()
    -- The variance of the attention filter
    local var = nn.Identity()()
    -- The size of the attention window
    local delta = nn.Identity()()

    -- An extra parameters used to pass a constant Tensor of
    -- indices in order to perform the calculations
    local idx = nn.Identity()()

    -- Calculates the numerator of the exponent
    local num = nn.CSubTable()
    (
    {
      nn.Replicate(batch_size, 1, 2)(idx),
      -- Create the array holding the mean for the current row
      mean_row(i, dim_size, N)({g, delta})
    }
    )

    -- Calculate the exponent of the filter
    local exponent = nn.CDivTable()
    (
    {
      nn.Power(2)(num) - nn.MulConstant(-1),
      nn.Replicate(dim_size, 1, 1)(var) - nn.MulConstant(2)
    }
    )

    local filter = nn.Exp()(exponent)
      - nn.Normalize(1)
    - nn.Unsqueeze(1, 1)

    return nn.gModule({g, delta, var, idx}, {filter})
  end

  local filters = {}
  for i = 1, N do
    filters[i] = filter_row(i, dim_size, N, batch_size)({g, delta, var, idx})
  end

  local filter_mat = nn.JoinTable(1, 2)(filters)

  return nn.gModule({g, delta, var, idx}, {filter_mat})
end

return M.Draw
