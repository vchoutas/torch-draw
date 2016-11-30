require('nn')
require('nngraph')

local M = {}

function M.create_model(options)
  local read_size = options.read_size
  local write_size = options.write_size

  local read_net

  local write_net

  return read_net, write_net
end

return M
