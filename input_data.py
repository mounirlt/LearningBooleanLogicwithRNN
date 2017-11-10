import numpy as np
from random import randint

__author__ = "Mounir Ouled Ltaief"
__email__ = "ouledl01@uni-passau.de"
__copyright__ = "Copyright 2017"

gate_type = dict([(itx[1], itx[0]) for itx in [
    ("nor", 1),
    ("xq", 2),
    ("abj", 3),
    ("xor", 4),
    ("nand", 5),
    ("and", 6),
    ("xnor", 7),
    ("ifthen", 8),
    ("thenif", 9),
    ("or", 10)
]])


def not_gate(x):
    return 1 if np.logical_not(x) == True else 0


def and_gate(x1, x2):
    return 1 if np.logical_and(x1, x2) == True else 0


def or_gate(x1, x2):
    return 1 if np.logical_or(x1, x2) == True else 0


def xor_gate(x1, x2):
    return 1 if np.logical_xor(x1, x2) == True else 0


def nor_gate(x1, x2):
    return 1 if np.logical_or(x1, x2) == False else 0


def xq_gate(x1, x2):
    return 1 if x1 == 0 and x2 == 1 else 0


def abj_gate(x1, x2):
    return 1 if x1 == 1 and x2 == 0 else 0


def nand_gate(x1, x2):
    return 1 if np.logical_and(x1, x2) == 0 else 0


def xnor_gate(x1, x2):
    return 1 if np.logical_xor(x1, x2) == False else 0


def ifthen_gate(x1, x2):
    return 1 if abj_gate(x1, x2) == 0 else 0


def thenif_gate(x1, x2):
    return 1 if xq_gate(x1, x2) == 0 else 0


def generate_input_vector(b_0=None):
    X = np.zeros(102, dtype=int);

    B = randint(1, 10)

    b0 = randint(0, 1) if b_0 == None else b_0
    b1 = randint(0, 1)

    X[0] = b0
    X[1] = b1

    start = 2
    end = B * 10 + 1

    while start <= end:
        gate = randint(1, 10)
        X[start + gate - 1] = 1
        start += 10

    return X, compute_binary_target(X, B)


def compute_binary_target(input_vector, number_of_chunks):
    b_0 = input_vector[0]
    b_1 = input_vector[1]

    counter = 2
    end = 10 * number_of_chunks + 1

    while counter <= end:
        if input_vector[counter] == 1:
            if counter % 10 == 0:
                gate_number = 9
            elif counter % 10 == 1:
                gate_number = 10
            else:
                gate_number = counter % 10 - 1
            b_B = eval(gate_type.get(gate_number) + "_gate(" + str(b_1) + "," + str(b_0) + ")")
            b_0 = b_1
            b_1 = b_B
        counter += 1

    return b_B


def load_data(nb_examples):
    target_vector = np.zeros(shape=(nb_examples, 1), dtype=int)
    initial_input = generate_input_vector()
    input_seq = initial_input[0]
    target_vector[0, 0] = initial_input[1]
    data_sequence = np.zeros(shape=(nb_examples, 102), dtype=int)
    data_sequence[0,] = input_seq
    for i in range(nb_examples - 1):
        new_input_vector = generate_input_vector(target_vector[i, 0])
        data_sequence[i + 1,] = new_input_vector[0]
        target_vector[i + 1, 0] = new_input_vector[1]
    return data_sequence, target_vector  # -*- coding: utf-8 -*-
