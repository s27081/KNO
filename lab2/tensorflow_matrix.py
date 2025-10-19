import tensorflow as tf
import argparse
import numpy as np
@tf.function
def rotate(points, theta):
    rotation_matrix = tf.stack(
        [tf.cos(theta), -tf.sin(theta), tf.sin(theta), tf.cos(theta)]
    )
    rotation_matrix = tf.reshape(rotation_matrix, (2, 2))
    return tf.matmul(rotation_matrix, points)

def solve_linear_system(A, B):
    det = tf.linalg.det(A)
    if det == 0:
        print("Wyznacznik równy zero")
        return
    X = tf.linalg.solve(A, B)
    return X

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--degree", type=int)
    parser.add_argument("-m", "--matrix_size", type=int)
    parser.add_argument("-A", "--Amatrix", type=int, nargs="+")
    parser.add_argument("-B", "--Bmatrix", type=int, nargs="+")
    args = parser.parse_args()

    if args.degree:
        print(rotate(tf.constant([[2.0, 0.0]], shape=(2, 1)), np.deg2rad(args.degree, dtype=np.float32)))

    if args.Amatrix and args.Bmatrix:
        A = tf.constant(args.Amatrix, shape=(args.matrix_size, args.matrix_size), dtype=tf.float32)
        B = tf.constant(args.Bmatrix, shape=(args.matrix_size, 1), dtype=tf.float32)
        result = solve_linear_system(A, B)
        print(result)

if __name__ == "__main__":
    main()
