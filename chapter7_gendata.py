'''Batch-generate spherical MNIST train/test splits with Gaussian image noise.

Running:
    python gendata.py

Default behavior:
    - generate 5 training sets with std in {0, 25, 50, 75, 100}
    - generate 21 test sets with std in {0, 5, 10, ..., 100}

Outputs:
    generated_data/
        train_std_0/s2_mnist_train.gz
        train_std_25/s2_mnist_train.gz
        train_std_50/s2_mnist_train.gz
        train_std_75/s2_mnist_train.gz
        train_std_100/s2_mnist_train.gz

        test_std_0/s2_mnist_test.gz
        test_std_5/s2_mnist_test.gz
        ...
        test_std_100/s2_mnist_test.gz
'''

import os
import gzip
import pickle
import argparse
import numpy as np
import lie_learn.spaces.S2 as S2
from torchvision import datasets


NORTHPOLE_EPSILON = 1e-3


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection:
        0   -> no rotation
        1.0 -> completely random rotation
    """
    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi
    phi = phi * 2.0 * np.pi
    z = z * 2.0 * deflection

    r = np.sqrt(z)
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0),
                  (-st, ct, 0),
                  (0, 0, 1)))
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def rotate_grid(rot, grid):
    x, y, z = grid
    xyz = np.array((x, y, z))
    x_r, y_r, z_r = np.einsum('ij,jab->iab', rot, xyz)
    return x_r, y_r, z_r


def get_projection_grid(b, grid_type="Driscoll-Healy"):
    theta, phi = S2.meshgrid(b=b, grid_type=grid_type)
    x_ = np.sin(theta) * np.cos(phi)
    y_ = np.sin(theta) * np.sin(phi)
    z_ = np.cos(theta)
    return x_, y_, z_


def project_sphere_on_xy_plane(grid, projection_origin):
    sx, sy, sz = projection_origin
    x, y, z = grid
    z = z.copy() + 1

    t = -z / (z - sz)
    qx = t * (x - sx) + x
    qy = t * (y - sy) + y

    xmin = 0.5 * (-1 - sx) + -1
    ymin = 0.5 * (-1 - sy) + -1

    rx = (qx - xmin) / (2 * np.abs(xmin))
    ry = (qy - ymin) / (2 * np.abs(ymin))
    return rx, ry


def sample_within_bounds(signal, x, y, bounds):
    xmin, xmax, ymin, ymax = bounds
    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    if len(signal.shape) > 2:
        sample = np.zeros((signal.shape[0], x.shape[0], x.shape[1]), dtype=np.float64)
        sample[:, idxs] = signal[:, x[idxs], y[idxs]]
    else:
        sample = np.zeros((x.shape[0], x.shape[1]), dtype=np.float64)
        sample[idxs] = signal[x[idxs], y[idxs]]
    return sample


def sample_bilinear(signal, rx, ry):
    signal_dim_x = signal.shape[1]
    signal_dim_y = signal.shape[2]

    rx = rx * signal_dim_x
    ry = ry * signal_dim_y

    ix = rx.astype(int)
    iy = ry.astype(int)

    ix0 = ix
    iy0 = iy
    ix1 = ix + 1
    iy1 = iy + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    fx1 = (ix1 - rx) * signal_00 + (rx - ix0) * signal_10
    fx2 = (ix1 - rx) * signal_01 + (rx - ix0) * signal_11

    return (iy1 - ry) * fx1 + (ry - iy0) * fx2


def normalize_to_uint8(sample):
    sample_min = sample.min(axis=(1, 2), keepdims=True)
    sample_max = sample.max(axis=(1, 2), keepdims=True)
    denom = np.maximum(sample_max - sample_min, 1e-8)

    sample = (sample - sample_min) / denom
    sample *= 255.0
    sample = np.clip(sample, 0.0, 255.0)
    return sample.astype(np.uint8)


def add_gaussian_noise_uint8(images, std, rng):
    """
    Add Gaussian noise to uint8 images.

    std is measured in pixel scale [0, 255].
    """
    if std <= 0:
        return images

    noisy = images.astype(np.float32) + rng.normal(
        loc=0.0,
        scale=std,
        size=images.shape
    ).astype(np.float32)

    noisy = np.clip(noisy, 0.0, 255.0)
    return noisy.astype(np.uint8)


def project_2d_on_sphere(signal, grid, projection_origin=None):
    if projection_origin is None:
        projection_origin = (0, 0, 2 + NORTHPOLE_EPSILON)

    rx, ry = project_sphere_on_xy_plane(grid, projection_origin)
    sample = sample_bilinear(signal, rx, ry)

    # Only south hemisphere is projected
    sample *= (grid[2] <= 1).astype(np.float64)

    sample = normalize_to_uint8(sample)
    return sample


def get_mnist_arrays(trainset, testset):
    if hasattr(trainset, "data"):
        train_images = trainset.data.numpy()
        train_labels = trainset.targets.numpy() if hasattr(trainset.targets, "numpy") else np.array(trainset.targets)

        test_images = testset.data.numpy()
        test_labels = testset.targets.numpy() if hasattr(testset.targets, "numpy") else np.array(testset.targets)
    else:
        train_images = trainset.train_data.numpy()
        train_labels = trainset.train_labels.numpy()

        test_images = testset.test_data.numpy()
        test_labels = testset.test_labels.numpy()

    mnist_train = {
        "images": train_images,
        "labels": train_labels
    }
    mnist_test = {
        "images": test_images,
        "labels": test_labels
    }
    return mnist_train, mnist_test


def generate_split(data, grid, rotate, rot_noise, image_noise_std, chunk_size, rng, split_name):
    print(f"projecting {split_name} split with image_noise_std={image_noise_std}")

    current = 0
    signals = data["images"].reshape(-1, 28, 28).astype(np.float64)
    n_signals = signals.shape[0]

    projections = np.ndarray(
        (n_signals, grid[0].shape[0], grid[0].shape[1]),
        dtype=np.uint8
    )

    while current < n_signals:
        if rotate:
            rot = rand_rotation_matrix(deflection=rot_noise)
            rotated_grid = rotate_grid(rot, grid)
        else:
            rotated_grid = grid

        idxs = np.arange(current, min(n_signals, current + chunk_size))
        chunk = signals[idxs]

        projected_chunk = project_2d_on_sphere(chunk, rotated_grid)
        projected_chunk = add_gaussian_noise_uint8(projected_chunk, image_noise_std, rng)

        projections[idxs] = projected_chunk
        current += len(idxs)
        print(f"\r{current}/{n_signals}", end="")

    print("")

    return {
        "images": projections,
        "labels": data["labels"]
    }


def save_split(split_data, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with gzip.open(output_path, "wb") as f:
        pickle.dump(split_data, f)

    print(f"saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bandwidth", type=int, default=30, help="bandwidth of the S2 signal")
    parser.add_argument("--noise", type=float, default=1.0, help="rotational noise / rotation deflection")
    parser.add_argument("--chunk_size", type=int, default=500, help="size of image chunk with same rotation")
    parser.add_argument("--mnist_data_folder", type=str, default="MNIST_data", help="folder for MNIST data")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--output_root", type=str, default="generated_data", help="root folder for outputs")

    parser.add_argument("--no_rotate_train", action="store_true", help="do not rotate training data")
    parser.add_argument("--no_rotate_test", action="store_true", help="do not rotate test data")

    args = parser.parse_args()

    # Fixed experimental design requested by user
    train_std_list = [0, 25, 50, 75, 100]
    test_std_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                     50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    print("getting MNIST data")
    trainset = datasets.MNIST(root=args.mnist_data_folder, train=True, download=True)
    testset = datasets.MNIST(root=args.mnist_data_folder, train=False, download=True)

    mnist_train, mnist_test = get_mnist_arrays(trainset, testset)
    grid = get_projection_grid(b=args.bandwidth)

    print("\nGenerating training sets...")
    for std in train_std_list:
        rng = np.random.default_rng(args.seed + 1000 + std)

        train_data = generate_split(
            data=mnist_train,
            grid=grid,
            rotate=not args.no_rotate_train,
            rot_noise=args.noise,
            image_noise_std=std,
            chunk_size=args.chunk_size,
            rng=rng,
            split_name=f"train std={std}"
        )

        train_output = os.path.join(
            args.output_root,
            f"train_std_{std}",
            "s2_mnist_train.gz"
        )
        save_split(train_data, train_output)

    print("\nGenerating test sets...")
    for std in test_std_list:
        rng = np.random.default_rng(args.seed + 2000 + std)

        test_data = generate_split(
            data=mnist_test,
            grid=grid,
            rotate=not args.no_rotate_test,
            rot_noise=args.noise,
            image_noise_std=std,
            chunk_size=args.chunk_size,
            rng=rng,
            split_name=f"test std={std}"
        )

        test_output = os.path.join(
            args.output_root,
            f"test_std_{std}",
            "s2_mnist_test.gz"
        )
        save_split(test_data, test_output)

    print("\nAll datasets generated successfully.")
    print(f"Output root: {args.output_root}")
    print(f"Training sets: {len(train_std_list)}")
    print(f"Test sets: {len(test_std_list)}")


if __name__ == "__main__":
    main()