require 'torch'
require 'nn'
require 'optim'

opt = {
   dataset = 'lsun',       -- imagenet / lsun / folder
   batchSize = 64,
   loadSize = 96,
   fineSize = 64,
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 25,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 0,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment1',
   noise = 'normal',       -- uniform / normal
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

opt["dataset"] = "folder"
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
if gpu then
    SpatialConvolution = nn.SpatialConvolutionMM
end
local SpatialFullConvolution = nn.SpatialFullConvolution

local gpu = true

-- 000, 018, 036, ... 180
-- base CASIA gait database
local pose_dim = 11 
local noise_dim = 50
local id_dim = 60

local netG = nn.Sequential()
local fx = nn.Sequential()
-- input is 1x96x96, going into a convolution

local input_channel = 1
conv_kernel = {32, 64, 64, 64, 128, 128, 96, 192, 192, 128, 256, 256, 160, 320}
conv_size = 3
padding_size = 1
conv_tride = {1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1}

fx:add(SpatialConvolution(input_channel, conv_kernel[1], conv_size
        , conv_size, conv_tride[1], conv_tride[1], padding_size, padding_size))
fx:add(SpatialBatchNormalization( conv_kernel[1] )):add(nn.ELU(true))

for i=2, #conv_kernel do
    fx:add(SpatialConvolution(conv_kernel[i-1], conv_kernel[i], conv_size
            , conv_size, conv_tride[i], conv_tride[i], padding_size, padding_size))
    fx:add(SpatialBatchNormalization( conv_kernel[i] )):add(nn.ELU(true))
end
-- output {batchsize}*6*6*320

fx:add(nn.SpatialAveragePooling(6, 6))
-- output {batchsize}*1*1*320, this is f(x)

fx:add(nn.Reshape(1, 320))

-- input dim of G_decoder is 320 + noise dim + pose dim
local G_decoder_dim = 320 + noise_dim + pose_dim

-- input = torch.randn(32, 381, 1, 1)
-- G_decoder:forward(input)

local G_decoder = nn.Sequential()
G_decoder:add(SpatialFullConvolution(G_decoder_dim, 320, 6, 6))
-- output 6*6*320
netG:add(SpatialBatchNormalization(320)):add(nn.ReLU(true))

for i = #conv_kernel - 1, 1, -1 do
    if conv_tride[i] == 1 then
        G_decoder:add(SpatialFullConvolution(
            conv_kernel[i+1], conv_kernel[i], conv_size, conv_size, 1, 1, 1, 1))
        -- output 6*6*160
    else
        G_decoder:add(SpatialFullConvolution(
            conv_kernel[i+1], conv_kernel[i], conv_size, conv_size, 2, 2, 1, 1,
            1, 1))
    end
    netG:add(SpatialBatchNormalization(conv_kernel[i])):add(nn.ReLU(true))
end

G_decoder:add(SpatialFullConvolution(32, 1, 3, 3, 1, 1, 1, 1))
-- output 96*96*1

G_decoder:add(nn.Tanh())

local netG = nn.Sequential()

-- first is X, second pose, then noise
local para_input_G = nn.ParallelTable()
para_input_G:add(fx)
para_input_G:add(nn.Identity())
para_input_G:add(nn.Identity())

netG:add(para_input_G)
netG:add(nn.JoinTable(2))
netG:add(nn.Reshape(G_decoder_dim, 1, 1))
netG:add(G_decoder)

local D = nn.Sequential()
D:add(SpatialConvolution(input_channel, conv_kernel[1], conv_size
        , conv_size, conv_tride[1], conv_tride[1], padding_size, padding_size))
D:add(SpatialBatchNormalization( conv_kernel[1] )):add(nn.ELU(true))

for i=2, #conv_kernel do
    D:add(SpatialConvolution(conv_kernel[i-1], conv_kernel[i], conv_size
            , conv_size, conv_tride[i], conv_tride[i], padding_size, padding_size))
    D:add(SpatialBatchNormalization( conv_kernel[i] )):add(nn.ELU(true))
end
-- output {batchsize}*6*6*320

D:add(nn.SpatialAveragePooling(6, 6))
-- output {batchsize}*1*1*320, this is f(x)

D:add(nn.Reshape(1, 320))

local duplicate = nn.ConcatTable()
duplicate:add(nn.Identity())
duplicate:add(nn.Identity())

local id_layer = nn.Sequential()
id_layer:add(nn.Linear(320, id_dim))
id_layer:add(nn.LogSoftMax())

local pose_layer = nn.Sequential()
pose_layer:add(nn.Linear(320, pose_dim))
pose_layer:add(nn.LogSoftMax())

local para = nn.ParallelTable()
para:add(id_layer)
para:add(pose_layer)

local netD = nn.Sequential()
netD:add(D)
netD:add(duplicate)
netD:add(para)

local criterion = nn.BCECriterion()
local crit = nn.SuperCriterion()
crit:add(nn.ClassNLLCriterion(), 1)
crit:add(nn.ClassNLLCriterion(), 1)

---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, 96, 96)
local noise = torch.Tensor(opt.batchSize, 1, noise_dim)
local label = torch.Tensor(opt.batchSize, 1, id_dim)
local pose = torch.Tensor(opt.batchSize, 1, pose_dim)

local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  label = label:cuda()
   pose = poes:cuda()

   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(netG, cudnn)
      cudnn.convert(netD, cudnn)
   end
   netD:cuda();           netG:cuda();           criterion:cuda()
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

noise_vis = noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   local real = data:getBatch()
   data_tm:stop()
   input:copy(real)
   label:fill(real_label)

   local output = netD:forward(input)
   local errD_real = criterion:forward(output, {label, pose})
   local df_do = criterion:backward(output, {label, pose})
   netD:backward(input, df_do)

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end
   local fake = netG:forward({input, pose, noise})
   input:copy(fake)
   label:fill(fake_label)

   local output = netD:forward(input)
   local errD_fake = criterion:forward(output, {label, pose})
   local df_do = criterion:backward(output, {label, pos})
   netD:backward(input, df_do)

   errD = errD_real + errD_fake

   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()

   --[[ the three lines below were already executed in fDx, so save computation
   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward(noise)
   input:copy(fake) ]]--
   label:fill(real_label) -- fake labels are real for generator cost

   local output = netD.output -- netD:forward(input) was already executed in fDx, so save computation
   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg = netD:updateGradInput(input, df_do)

   netG:backward(noise, df_dg)
   return errG, gradParametersG
end

-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDx, parametersD, optimStateD)

      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx, parametersG, optimStateG)

      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
          local fake = netG:forward(noise_vis)
          local real = data:getBatch()
          disp.image(fake, {win=opt.display_id, title=opt.name})
          disp.image(real, {win=opt.display_id * 3, title=opt.name})
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
      end
   end
   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
