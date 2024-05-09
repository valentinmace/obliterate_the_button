# A very dumb idea for a very dumb game https://elendow.itch.io/the-button
# This is a simple script to obliterate the world record on 'The Button',
# basically we run billions of tries in parallel and try to beat the record.

import time
import jax
import jax.numpy as jnp
from jax import random
jnp.set_printoptions(precision=2, floatmode='fixed', suppress=True)


# tpu_device = jax.devices('tpu')[0]    # if you are rich
gpu_device = jax.devices('gpu')[0]
cpu_device = jax.devices('cpu')[0]
which_device = cpu_device   # chose here on which device you want to run the code


def closest_multiple(A, B):
    closest_factor = round(A / B)
    return closest_factor * B


# Runs starts here

seed = 0
batch_size = 100000    # number of games played at once
print_every = 1000000    # how many games we want to play before printing again ?
print_every_adapted = closest_multiple(print_every, batch_size)     # round to the closest multiple so there's no issue

best_so_far = 0
start = 0.0
with jax.default_device(which_device):
    for i in range(10000000000000000000000000):    # lol
        key = random.PRNGKey(seed)

        # Get all probabilities of losing [1%, 2%, ...]
        array_of_prob = jnp.arange(100)

        # Sample random values to compare with probabilities of loosing (this is where we "play" the game)
        random_values = random.uniform(key, shape=(batch_size, 100)) * 100
        results = random_values < array_of_prob

        # Get first True value, indicating where we lost
        first_true_index = jnp.argmax(results, axis=1)

        # Print stuff
        best = jnp.max(first_true_index)
        seed += 1
        if best > best_so_far: best_so_far = best

        if i % (print_every_adapted/batch_size) == 0:
            print(f"\n\niteration {i}")
            print(f"Max score since last print {best}")
            print(f"Max score overall {best_so_far}")
            print(f"Total number of tries : {(i+1)*batch_size:,}")
            print(f"Duration for ~{print_every:,} tries, (precisely {print_every_adapted:,}) : {(time.time() - start):.2f}")
            start = time.time()
