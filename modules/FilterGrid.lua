require('nn')

local FilterGrid, parent = torch.class('nn.FilterGrid', 'nn.Module')

function FilterGrid:__init(N, dim_size, nInputDim)
  self.grid = torch.range(1, dim_size):resize(1, dim_size):repeatTensor(N, 1)
  self.nInputDim = nInputDim
  parent.__init(self)
end

function FilterGrid:updateOutput(input)
  if self.nInputDim and input:dim()> self.nInputDim then
    local grid_size = self.grid:size():totable()
    self.output:resize(input:size(1), table.unpack(grid_size))
    local grid = self.grid:view(1, table.unpack(grid_size))
    self.output:copy(grid:expand(self.output:size()))
  else
    self.output:resize(self.grid:size())
    self.output:copy(self.grid)
  end
  return self.output
end

function FilterGrid:cuda()
  self.output = self.grid:cuda()
end

function FilterGrid:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  return self.gradInput
end
