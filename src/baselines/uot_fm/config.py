import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 42
    config.t1 = 1.0
    config.dt0 = 0.0
    config.solver = 'tsit5'

    config.training = training = ml_collections.ConfigDict()
    training.tau_a = 0.95
    training.tau_b = 0.95
    training.print_freq = 500
    training.num_steps = 5000
    training.batch_size = 500
    training.epsilon = 0.5
    training.method = 'flow'
    training.flow_sigma = 0.01
    training.gamma = 'constant'

    config.optim = optim = ml_collections.ConfigDict()
    optim.ema_decay = 0.9999
    optim.schedule = 'constant'
    optim.learning_rate = 0.0001
    optim.warmup = 0.0
    optim.optimizer = 'adam'
    optim.weight_decay = 0.0
    optim.beta_one = 0.9
    optim.beta_two = 0.999
    optim.eps = 1e-8
    optim.grad_clip = 1.0

    return config