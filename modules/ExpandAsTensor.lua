require('nn')

local ExpandAsTensor, parent = torch.class('nn.ExpandAsTensor', 'nn.Module')

function ExpandAsTensor:__init(nInputDim)
  self.nInputDim = nInputDim
  parent.__init(self)
end

function ExpandAsTensor:updateOutput(input)
  -- TO DO: Add check for batch and non batch input
  -- if self.nInputDim and input[2]:dim() > self.nInputDim then

  -- else
  -- end
  self.output:resizeAs(input[1]):copy(input[1])
  self.output = self.output:expandAs(input[2])
  return self.output
end

function ExpandAsTensor:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  return self.gradInput
end


function ExpandAsTensor:cuda()
  self.output = self.output:cuda()
end

