class RegConfig:
    train_size = 1024
    batch_size = 128
    lr = 1e-3
    epochs = 50
    train_samples = 10                   # number of train samples for MC gradients
    test_samples = 10                   # number of test samples for MC averaging
    num_test_points = 400               # number of test points
    experiment = 'regression'
    hidden_units = 400                  # number of hidden units
    noise_tolerance = 1.0                # log likelihood sigma
    mu_init = [-0.2, 0.2]               # range for mean 
    rho_init = [-4, -3]                 # range for rho_param
    prior_init = [0.5, -0, -6]        # mixture weight, log(stddev_1), log(stddev_2)
   

class RLConfig:
    data_dir = '/kaggle/input/mushroom-dataset/mushroom/agaricus-lepiota.data'
    batch_size = 64
    num_batches = 64
    buffer_size = batch_size * num_batches  # buffer to track latest batch of mushrooms
    lr = 1e-4
    training_steps = 10000
    experiment = 'regression'
    hidden_units = 100                      # number of hidden units
    mu_init = [-0.2, 0.2]                   # range for mu 
    rho_init = [-5, -4]                     # range for rho
    prior_init = [0.5, -0, -6]               # mixture weight, log(stddev_1), log(stddev_2)

class ClassConfig:
    batch_size = 128
    lr = 1e-3 
    epochs = 600 
    hidden_units = 1200
    experiment = 'classification'
    dropout = False
    train_samples = 1 
    test_samples = 10
    x_shape = 28 * 28                       # x shape
    classes = 10                            # number of output classes
    mu_init = [-0.2, 0.2]                   # range for mean 
    rho_init = [-5, -4]                     # range for rho_param
    prior_init = [0.75, 0, -7]             # mixture weight, log(stddev_1), log(stddev_2)