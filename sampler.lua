require('image')
local nn = require('nn')

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)

local Draw = require('models/draw')
local utils = require('utils')
local model_utils = require('models/model_utils')

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


if options.dataset == 'mnist' then
  options.img_size = torch.LongStorage({28, 28})
  options.input_size = 28 * 28
elseif options.dataset == 'cifar10' then
  options.img_size = torch.LongStorage({3, 32, 32})
  options.input_size = 3 * 32 * 32
else
  error('Invalid dataset!')
end

local T = options.num_glimpses
local write_size = options.write_size
local latent_size = options.latent_size
local batch_size = options.batch_size
local hidden_size = options.hidden_size
local img_size = options.img_size


local use_attention = options.use_attention == 'true'

local model_folder = paths.concat(options.model_folder, use_attention
  and 'attention' or 'no_attention')
-- Load the decoder
local decoder_path = paths.concat(model_folder, 'decoder.t7')
local decoder = torch.load(decoder_path)

print('===> Unrolling model in time...')
local unrolled_decoder = model_utils.clone_model(decoder, T)
unrolled_decoder[1], learnedParams = model_utils.addLearnedBias(options, unrolled_decoder[1], 'validation')

local params, _ = learnedParams:parameters()
local canvas0Path = paths.concat(model_folder, 'canvas0.t7')

-- Load the initial canvas and decoder state from the file
local storedCanvas = torch.load(canvas0Path)
params[1]:copy(storedCanvas)

local hDec0 = paths.concat(model_folder, 'hDec0.t7')

local storedH0 = torch.load(hDec0)
params[2]:copy(storedH0)

local sharedContainer = model_utils.create_shared_container(unrolled_decoder)
print('===> Finished unrolling model in time...')

if use_cuda then
  sharedContainer = sharedContainer:cuda()
end

local z = torch.Tensor(batch_size, latent_size)
if use_cuda then
  z = z:cuda()
end

local canvas0 = torch.Tensor(batch_size, table.unpack(img_size:totable()))
local h0 = torch.Tensor(batch_size, hidden_size)
local c0 = torch.Tensor(batch_size, hidden_size)

if use_cuda then
  canvas0 = canvas0:cuda()
  h0 = h0:cuda()
  c0 = c0:cuda()
end

local canvas = {[0] = canvas0}
local h_dec = {[0] = h0}
local c_dec = {[0] = c0}

local inputs

local img_seq = {}

local canvas2Img = nn.Sequential():add(nn.Sigmoid())
if use_cuda then
 canvas2Img = canvas2Img:cuda()
end

for t = 1, T do
  inputs = {z:normal(), canvas[t - 1], h_dec[t - 1], c_dec[t - 1]}

  canvas[t], h_dec[t], c_dec[t], att_params = table.unpack(unrolled_decoder[t]:forward(inputs))
  img = canvas2Img:forward(canvas[t])

  local rgb_canvas = utils.toRGB(img):float()

  local displayImg
  if use_attention then
    displayImg = utils.drawAttentionRect(write_size, rgb_canvas, att_params, {255, 0, 0})
  else
    displayImg = rgb_canvas
  end

  img_seq[t]  = image.toDisplayTensor(displayImg, 2, options.img_per_row)

  if window ~= nil then
    window = image.display{image=img_seq[t], win=window}
  else
    window = image.display(img_seq[t])
  end
  sys.sleep(0.1)
end

local img_folder = options.img_folder
if not paths.dirp(img_folder) and not paths.mkdir(img_folder) then
  cmd:error('Error: Unable to create image directory: ' .. img_folder '\n')
end

local seq_folder = paths.concat(img_folder, options.dataset, 'sampler')
if not paths.dirp(seq_folder) and not paths.mkdir(seq_folder) then
  cmd:error('Error: Unable to create image directory: ' .. seq_folder '\n')
end

print('Saving image sequence...')
utils.storeSeq(seq_folder, img_seq)
print('Finished saving image sequence...')
