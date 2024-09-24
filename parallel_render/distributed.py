import glob
import json
import multiprocessing
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional
import os
import concurrent.futures

import boto3
import tyro
import wandb
import math

with open("/home/ec2-user/zjc_renderobjaverse/model_paths_notblender_final.txt", "r") as file:
    lines = file.readlines()
model_notblender = [line.strip() for line in lines]


@dataclass
class Args:
    workers_per_gpu: int
    """number of workers per gpu"""

    input_models_path: str
    """Path to a json file containing a list of 3D object files"""

    upload_to_s3: bool = True
    """Whether to upload the rendered images to S3"""

    log_to_wandb: bool = False
    """Whether to log the progress to wandb"""

    num_gpus: int = -1
    """number of gpus to use. -1 means all available gpus"""


def upload_file_to_s3(local_file_path, s3, bucket_name, s3_key):
    try:
        s3.upload_file(local_file_path, bucket_name, s3_key)
        print(f"Uploaded {local_file_path} to S3 bucket {bucket_name} as {s3_key}")
    except Exception as e:
        print(f"Failed to upload {local_file_path} to S3: {e}")


def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    s3: Optional[boto3.client],
    bucket_name: Optional[str]
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break

        view_path = os.path.join('.objaverse2_renderagain/hf-objaverse-v1/views_whole_sphere', item.split('/')[-1][:-4])
        if os.path.exists(view_path):
            queue.task_done()
            print('========', item, 'already rendered', '========')
            continue
        else:
            os.makedirs(view_path, exist_ok=True)

        # Perform the rendering operation
        print(f"Rendering {item} on GPU {gpu}")
        command = (
            f"CUDA_VISIBLE_DEVICES={gpu} "
            f"./blender -b -P /home/ec2-user/zjc_renderobjaverse/blender_script.py --"
            f" --object_path {item} --output_dir {view_path}"
        )
        print(command)
        subprocess.run(command, shell=True)

        if s3:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for root, dirs, files in os.walk(view_path):
                    for file in files:
                        local_file_path = os.path.join(root, file)
                        s3_key = os.path.relpath(local_file_path, start=view_path)
                        futures.append(executor.submit(upload_file_to_s3, local_file_path, s3, bucket_name, s3_key))
                
                # Wait for all uploads to complete
                for future in concurrent.futures.as_completed(futures):
                    future.result()

        with count.get_lock():
            count.value += 1

        queue.task_done()


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.upload_to_s3:
        # AWS credentials and region
        aws_access_key_id = ''
        aws_secret_access_key = ''
        region_name = ''

        # Initialize S3 client
        s3 = boto3.client('s3',
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key,
                          region_name=region_name)

        # Specify the S3 bucket
        bucket_name = 'zjcrenderdataagain'
    else:
        s3 = None
        bucket_name = None

    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    if args.log_to_wandb:
        wandb.init(project="objaverse-rendering", entity="prior-ai2")

    # Start worker processes on each of the GPUs
    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            worker_process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_i, s3, bucket_name)
            )
            worker_process.daemon = True
            worker_process.start()

    # Add items to the queue
    with open(args.input_models_path, "r") as f:
        model_paths = json.load(f)

    model_keys = list(model_paths.keys())

    for item in model_keys:
        object_name = model_paths[item].split('/')[-1][:-4]
        if object_name in model_notblender:
            queue.put(os.path.join('/home/ec2-user/zjc_renderobjaverse/objaverse_render/objaverse_data_zjc/hf-objaverse-v1', model_paths[item]))
        else:
            print(f"Skipping {item} as it is already completed")
        #queue.put(os.path.join('/home/ec2-user/zjc_renderobjaverse/objaverse_render/objaverse_data_zjc/hf-objaverse-v1', model_paths[item]))

    # Update the wandb count
    if args.log_to_wandb:
        while True:
            time.sleep(5)
            wandb.log(
                {
                    "count": count.value,
                    "total": len(model_paths),
                    "progress": count.value / len(model_paths),
                }
            )
            if count.value == len(model_paths):
                break

    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(args.num_gpus * args.workers_per_gpu):
        queue.put(None)
