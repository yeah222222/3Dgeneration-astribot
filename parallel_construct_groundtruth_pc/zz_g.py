import torch
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm

import pymeshlab,os,sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--save_dir", type=str, default="/home/ec2-user/zjc_renderobjaverse/output_pointcloud_sample")
argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

if not os.path.exists(args.save_dir):
    os.mkdirs(args.save_dir)


if __name__ == '__main__':
    fp = args.object_path
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.join('/home/stardust/zjc/normalized_model_grasp',fp,'normalized_textured.obj'))
    ms.load_filter_script('color.mlx')
    ms.apply_filter_script()
    object_uid = os.path.basename(fp).split(".")[0]
    # print('object uid',object_uid)
    s_p = os.path.join(args.save_dir,object_uid+'.ply')
    ms.save_current_mesh(s_p)

  

  