"""
Create LMDB shards with latents + UniAnimate-style pose conditioning.

Expected per-sample shapes (without batch dimension):
  - latents: [21, 16, 104, 60]
  - dwpose_data: [3, 81, 832, 480]
  - random_ref_dwpose: [832, 480, 3]
  - first_frame: [832, 480, 3]

Example:
python create_pose_lmdb_shards.py \
  --data_path /path/to/pose_pairs \
  --lmdb_path /path/to/pose_lmdb \
  --num_shards 16
"""
from tqdm import tqdm
import numpy as np
import argparse
import torch
import lmdb
import glob
import os

from utils.lmdb import store_arrays_to_lmdb


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def _ensure_batch(x, target_ndim):
    if x.ndim == target_ndim - 1:
        return np.expand_dims(x, axis=0)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to pose pairs")
    parser.add_argument("--lmdb_path", type=str, required=True, help="path to lmdb output")
    parser.add_argument("--num_shards", type=int, default=16, help="num_shards")
    args = parser.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.data_path, "**", "*.pt"), recursive=True))

    map_size = int(1e12)  # 1TB by default
    os.makedirs(args.lmdb_path, exist_ok=True)
    envs = []
    for shard_id in range(args.num_shards):
        path = os.path.join(args.lmdb_path, f"shard_{shard_id}")
        env = lmdb.open(
            path,
            map_size=map_size,
            subdir=True,
            readonly=False,
            metasync=True,
            sync=True,
            lock=True,
            readahead=False,
            meminit=False
        )
        envs.append(env)

    counters = [0] * args.num_shards
    shapes = None

    expected_latents = (21, 16, 104, 60)
    expected_dwpose = (3, 81, 832, 480)
    expected_frame = (832, 480, 3)

    for idx, file in tqdm(enumerate(all_files), total=len(all_files)):
        try:
            data = torch.load(file)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

        if not isinstance(data, dict) or "prompts" not in data:
            continue

        prompts = data["prompts"]
        if isinstance(prompts, str):
            prompts = [prompts]

        latents = _ensure_batch(_to_numpy(data["latents"]), 5).astype(np.float16, copy=False)
        dwpose = _ensure_batch(_to_numpy(data["dwpose_data"]), 5).astype(np.uint8, copy=False)
        random_ref = _ensure_batch(_to_numpy(data["random_ref_dwpose"]), 4).astype(np.uint8, copy=False)
        first_frame = _ensure_batch(_to_numpy(data["first_frame"]), 4).astype(np.uint8, copy=False)

        if latents.shape[1:] != expected_latents:
            continue
        if dwpose.shape[1:] != expected_dwpose:
            continue
        if random_ref.shape[1:] != expected_frame or first_frame.shape[1:] != expected_frame:
            continue
        if latents.shape[0] != len(prompts):
            continue

        arrays = {
            "prompts": prompts,
            "latents": latents,
            "dwpose_data": dwpose,
            "random_ref_dwpose": random_ref,
            "first_frame": first_frame,
        }

        shard_id = idx % args.num_shards
        store_arrays_to_lmdb(envs[shard_id], arrays, start_index=counters[shard_id])
        counters[shard_id] += len(prompts)

        if shapes is None:
            shapes = {k: (len(prompts),) if k == "prompts" else v.shape for k, v in arrays.items()}

    if shapes is None:
        raise RuntimeError("No valid samples found to write.")

    for shard_id, env in enumerate(envs):
        with env.begin(write=True) as txn:
            for key, shape in shapes.items():
                array_shape = np.array(shape)
                array_shape[0] = counters[shard_id]
                shape_key = f"{key}_shape".encode()
                shape_str = " ".join(map(str, array_shape))
                txn.put(shape_key, shape_str.encode())

    print(f"Finished writing pose LMDB into {args.num_shards} shards under {args.lmdb_path}")


if __name__ == "__main__":
    main()
