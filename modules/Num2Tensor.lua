require('nn')

local Num2Tensor, parent = torch.class('nn.Num2Tensor', 'nn.Module')

function Num2Tensor:__init(dims)
  self.dims = dims
  self.num_elements = torch.Tensor(table.unpack(dims)):nElement()
  parent.__init(self)
end

function Num2Tensor:updateOutput(input)
  local val = input[1]
  local batch = input[2]
  if batch:dim() > #self.dims then
    local num_elements = self.num_elements
    self.output = val:repeatTensor(1, num_elements)
    self.output:resize(batch:size(1), table.unpack(self.dims))
  else
    self.output:resize(table.unpack(self.dims)):fill(val:squeeze())
  end
  return self.output
end

function Num2Tensor:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  return self.gradInput
end

function Num2Tensor:cuda()
  self.output = self.output:cuda()
  self.gradInput = self.gradInput:cuda()
end
