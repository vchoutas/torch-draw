require('nn')
require('image')

local utils = {}

function utils.center2Corners(g, delta, N, dim_size)
  local p1 = math.floor(g + (1 -  N/ 2 - 0.5) * delta)
  p1 = math.max(math.min(p1, dim_size), 1)
  local p2 = math.ceil(g + (N / 2 - 0.5) * delta)
  p2 = math.max(math.min(p2, dim_size), 1)
  return p1, p2
end

function utils.toRGB(batch)
  local output = torch.Tensor():typeAs(batch):resizeAs(batch)
  output:copy(batch)
  if #batch:size() <= 3 then
    -- Batch Input
    if #batch:size() > 2 then
      local batch_size, height, width = table.unpack(batch:size():totable())
      output = output:resize(batch_size, 1, height, width):repeatTensor(1, 3, 1, 1)
    else
      local height, width = table.unpack(batch:size():totable())
      output = output:resize(1, height, width):repeatTensor(3, 1, 1)
    end
  else
    local depth = batch:size(2)
    if depth < 3 then
      output = output:repeatTensor(1, 3, 1, 1)
    end
  end
  return output
end

function utils.storeSeq(folder, img_seq)
  for t = 1, #img_seq do
    local img_name = paths.concat(folder, 'seq' .. t .. '.png')
    image.save(img_name, img_seq[t])
  end
end

function utils.drawReadSeq(T, N, input, att_params, line_color)
  local batch = utils.toRGB(input)
  if batch:type() == 'torch.CudaTensor' then
    batch = batch:float()
  end

  local img_seq = {}
  for t = 1, T do
    img_seq[t] = utils.drawAttentionRect(N, batch, att_params[t], line_color)
  end

  return img_seq
end

function utils.drawWriteSeq(T, N, input, att_params, line_color)
  local img_seq = {}

  for t = 1, T do
    local batch = utils.toRGB(input[t])
    if batch:type() == 'torch.CudaTensor' then
      batch = batch:float()
    end
    img_seq[t] = utils.drawAttentionRect(N, batch, att_params[t], line_color)
  end

  return img_seq
end

function utils.drawAttentionRect(N, input, att_params, line_color)

  local batch_size, _, height, width = table.unpack(input:size():totable())

  local output = torch.Tensor():typeAs(input):resizeAs(input)
  output:copy(input)

  local gx, gy, var, delta = table.unpack(att_params)

  local rectOptions = {color = line_color, inplace = true}
  for i = 1, batch_size do
    local x1, x2 = utils.center2Corners(gx[i]:squeeze(), delta[i]:squeeze(), N, width)
    local y1, y2 = utils.center2Corners(gy[i]:squeeze(), delta[i]:squeeze(), N, height)

    image.drawRect(output[i], x1, y1, x2, y2, rectOptions)
  end

  return output
end

return utils
