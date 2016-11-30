require('nn')
require('nngraph')

local M = {}


function M.create_lstm(options, input_size, hidden_size)

  -- The input of the LSTM
  local x = nn.Identity()()
  -- The previous hidden state.
  local h_prev = nn.Identity()()
  -- The previous cell state
  local c_prev = nn.Identity()()

  local inputs = {x, h_prev, c_prev}

  -- The input to hidden transition
  local i2h = nn.Linear(input_size, 4 * hidden_size)(x)

  -- The hidden to hidden transition
  local h2h = nn.Linear(hidden_size, 4 * hidden_size)(h_prev)

  -- Add the transitions
  local input = nn.CAddTable()({i2h, h2h})

  -- Create the input gate
  local in_gate = nn.Narrow(2, 1, hidden_size)(input)
    - nn.Sigmoid()
  -- Create the forget gate
  local forget_gate = nn.Narrow(2, hidden_size + 1, hidden_size)(input)
    - nn.Sigmoid()
  -- Create the output gate
  local output_gate = nn.Narrow(2, 2 * hidden_size + 1, hidden_size)(input)
    - nn.Sigmoid()
  -- Create the gate that is used to decide which part of the input
  -- to write to the cell state.
  local g_gate = nn.Narrow(2, 3 * hidden_size + 1, hidden_size)(input)
    - nn.Tanh()

  -- What to forget from the previous cell state
  local cell_forget = nn.CMulTable()({c_prev, forget_gate})
  -- What to write on the cell state
  local input_write = nn.CMulTable()({in_gate, g_gate})
  -- The new cell state
  local next_c = nn.CAddTable()({cell_forget, input_write})

  local scaled_cell = nn.Tanh()(next_c)
  local next_h = nn.CMulTable()({output_gate, scaled_cell})

  local outputs = {}

  table.insert(outputs, next_h)
  table.insert(outputs, next_c)

  -- Return the graph module that wraps the LSTM cell
  return nn.gModule(inputs, outputs)
end

return M
