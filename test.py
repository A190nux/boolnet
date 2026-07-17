from bool_net import *

# Parity on 4 inputs (XOR chain)
def parity4(x):
    return x[0] ^ x[1] ^ x[2] ^ x[3]

# Use all 16 rows for training
train_data, test_data = make_dataset(n=4, target_fn=parity4, observed_fraction=1.0, seed=99)

# Train with restarts
net = train_with_restarts(
    n_inputs=4,
    layer_widths=[8, 4],
    train_data=train_data,
    pair_strategy='aggressive',
    max_steps=500,          # steps per restart
    n_restarts=10,
    verbose=True
)

evaluate(net, train_data, test_data, verbose=True)