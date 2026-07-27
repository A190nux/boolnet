from bool_net import *

# Parity on 4 inputs (XOR chain)
def my_fn(x):
    return x[0] ^ x[1] ^ x[2] ^ x[3]

train_data, test_data = make_dataset(n=4, target_fn=my_fn, observed_fraction=0.75)
net = BooleanNetwork(n_inputs=4, layer_widths=[4, 4], seed=42)
train(net, train_data, max_steps=500, verbose=True)
evaluate(net, train_data, test_data, verbose=True)