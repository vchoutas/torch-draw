local nn = require('nn')

local GaussianSampler, parent = torch.class('nn.GaussianSampler', 'nn.Module')

function GaussianSampler:__init(mean, std)
  parent.__init(self)

  -- Initialize the parameters of the Gaussian
  self.mean = mean or 0
  self.std = std or 1
end

function GaussianSampler:updateOutput(input)
  self.output:resizeAs(input)
  self.output:normal(self.mean, self.std)
  return self.output
end

function GaussianSampler:updateGradInput(input, gradOutput, scale)
  self.gradInput:resizeAs(input)
  self.gradInput:zero()
  return self.gradInput
end
