# import argparse
# import os,sys
# import pymeshlab

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--object_path",
#     type=str,
#     required=True,
#     help="Path to the object file",
# )
# parser.add_argument("--save_dir", type=str, default="/home/ec2-user/zjc_renderobjaverse/output_pointcloud_sample")
# argv = sys.argv[sys.argv.index("--") + 1 :]
# args = parser.parse_args(argv)

# if not os.path.exists(args.save_dir):
#     os.mkdirs(args.save_dir)

# # object_uid = os.path.basename(fp).split(".")[0]
# def runcode(fp):
#     ms = pymeshlab.MeshSet()
#     ms.load_new_mesh(fp)
#     ms.load_filter_script('color.mlx')
#     ms.apply_filter_script()
#     mesh_inmem = ms.current_mesh()
#     return mesh_inmem
    
# if __name__ == "__main__":
#     fp = args.object_path
#     memory_in_mesh = runcode(fp=fp)
#     # object_uid = os.path.basename(fp).split(".")[0]
#     # print('object uid',object_uid)
#     # s_p = os.path.join(args.save_dir,object_uid+'.ply')
#     # print("sp=====================================================",s_p)


import argparse
from io import BytesIO
import pymeshlab,os,sys
import boto3

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
#parser.add_argument("--save_dir", type=str, default="/home/ec2-user/zjc_renderobjaverse/output_pointcloud_sample")
argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

s3 = boto3.client(
    's3',
    aws_access_key_id='AKIATCKAOQWQJ2G2TUJX',
    aws_secret_access_key='ADuHS/K8a9i6JGPXHR6xt/u5XobmtREW5ubNp7oK',
    region_name='us-west-2'
)

# if not os.path.exists(args.save_dir):
#     os.mkdirs(args.save_dir)




if __name__ == '__main__':
    fp = args.object_path
    object_uid = os.path.basename(fp).split(".")[0]
    bucket_download = 'astribot-aws-bucket'
    s3_key = os.path.join('objaverse_data_zjc/hf-objaverse-v1/', fp)
    s3.download_file(bucket_download, s3_key, object_uid + '.glb')
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(object_uid + '.glb')
    ms.load_filter_script('color.mlx')
    ms.apply_filter_script()

    # 导出处理后的 mesh 到 PLY 文件
    output_ply_file = object_uid + '.ply'
    ms.save_current_mesh(output_ply_file)

    # 读取文件并上传到 S3
    with open(output_ply_file, 'rb') as ply_file:
        mesh_bytes = ply_file.read()

    bucket_name = 'zjc-sample-pointcloud-gt'
    file_name = os.path.join('colorbatchtest', object_uid + '.ply')
    file_type = 'text/ply'
    s3.put_object(Body=BytesIO(mesh_bytes), Bucket=bucket_name, Key=file_name, ContentType=file_type)

    # 上传完成后删除本地文件
    if os.path.exists(output_ply_file):
        os.remove(output_ply_file)

    # 删除下载的 glb 文件
    downloaded_glb_file = object_uid + '.glb'
    if os.path.exists(downloaded_glb_file):
        os.remove(downloaded_glb_file)
  