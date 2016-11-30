require('nn')
local MeanGrid, parent = torch.class('nn.MeanGrid', 'nn.Module')

function MeanGrid:__init(N, dim_size, nInputDim)
  self.grid = torch.range(1, N):resize(N, 1):repeatTensor(1, dim_size)
  self.nInputDim = nInputDim
  parent.__init(self)
end

function MeanGrid:updateOutput(input)
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

function MeanGrid:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  return self.gradInput
end

function MeanGrid:cuda()
  self.grid = self.grid:cuda()
end

