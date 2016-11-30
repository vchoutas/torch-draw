
--[[ An implementation of Eve https://arxiv.org/pdf/1611.01505v2.pdf
ARGS:
- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- `config.learningRateDecay` : learning rate decay
- 'config.beta1'             : first moment coefficient
- 'config.beta2'             : second moment coefficient
- 'config.epsilon'           : for numerical stability
- 'config.weightDecay'       : weight decay
- 'config.beta3'             : moment coefficient for relative change
- 'config.k'                 : lower threshold for relative change
- 'config.K'                 : upper threshold for relative change
- 'state'                    : a table describing the state of the optimizer; after each
                              call the state is modified
RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update
]]


function eve(opfunc, x, config, state)
  local config = config or {}
  local state = state or config
  local lr = config.learningRate or 0.001
  local lrd = config.learningRateDecay or 0

  -- Adam parameters
  local beta1 = config.beta1 or 0.9
  local beta2 = config.beta2 or 0.999
  local epsilon = config.epsilon or 1e-8
  local wd = config.weightDecay or 0

  -- Eve parameters
  local beta3 = config.beta3 or 0.999
  local k = config.k or 0.1
  local K = config.K or 10

  local delta, Delta

  -- (1) evaluate f(x) and df/dx
   local fx, dfdx = opfunc(x)

   -- (2) weight decay
   if wd ~= 0 then
      dfdx:add(wd, x)
   end

   -- Initialization
   state.t = state.t or 0

   -- Adam parameters
   -- Exponential moving average of gradient values
   state.m = state.m or x.new(dfdx:size()):zero()
   -- Exponential moving average of squared gradient values
   state.v = state.v or x.new(dfdx:size()):zero()
   -- A tmp tensor to hold the sqrt(v) + epsilon
   state.denom = state.denom or x.new(dfdx:size()):zero()
   --
   -- (3) learning rate decay (annealing)
   local clr = lr / (1 + state.t*lrd)

   -- state.f_hat_t1 = state.f_hat_t1 or 0
   -- state.f_hat_t2 = state.f_hat_t2 or 0
   state.smooth_val = state.smooth_val or 0

   state.t = state.t + 1
   state.d = state.d or 1

   if state.t > 1 then
     if fx >= state.smooth_val then
       delta = k + 1
       Delta = K + 1
     else
       delta = 1 / (K + 1)
       Delta = 1 / (k + 1)
     end

     local c = math.min(math.max(delta, fx / (state.smooth_val + epsilon) ), Delta)

     local prev_smooth_val = state.smooth_val
     state.smooth_val = c * state.smooth_val

     local r = 0

     r = math.abs(state.smooth_val - prev_smooth_val)
     r = r / (math.min(state.smooth_val, prev_smooth_val) + epsilon)

     state.d = beta3 * state.d + (1 - beta3) * r

   else
     state.smooth_val = fx
     state.d = 1
   end


   -- Decay the first and second moment running average coefficient
   state.m:mul(beta1):add(1-beta1, dfdx)
   state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)
   state.denom:copy(state.v):sqrt():mul(state.d):add(epsilon)

   local biasCorrection1 = 1 - beta1^state.t
   local biasCorrection2 = 1 - beta2^state.t

   local stepSize = clr * math.sqrt(biasCorrection2)/biasCorrection1
   -- (4) update x
   x:addcdiv(-stepSize, state.m, state.denom)

   -- return x*, f(x) before optimization
   return x, {fx}
end

-- return M
