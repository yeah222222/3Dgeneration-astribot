import bpy
import math
import mathutils
import os
#当前的obj文件距离单位球的体积过小，不容易blender，因此进行preprocess

# 清除当前场景中的所有对象


# 加载 OBJ 文件
obj_Abs_path = '/home/stardust/zjc/models'
save_Abs_path = '/home/stardust/zjc/normalized_model_grasp'
obj_path_list = []
for name in os.listdir(obj_Abs_path):
    name_path = os.path.join(obj_Abs_path,name)
    if os.path.isdir(name_path):
        obj_path_list.append(os.path.join(obj_Abs_path,name,'textured.obj'))
obj_path_list = sorted(obj_path_list)
for obj_path in obj_path_list:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.obj(filepath=obj_path)

    # 获取导入的对象
    obj = bpy.context.selected_objects[0]

    # 计算包围盒的尺寸
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    min_corner = mathutils.Vector((min([v[i] for v in bbox]) for i in range(3)))
    max_corner = mathutils.Vector((max([v[i] for v in bbox]) for i in range(3)))
    dimensions = max_corner - min_corner

    # 计算缩放因子
    max_dimension = max(dimensions)
    scale_factor = 1.0 / max_dimension

    # 应用缩放
    obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # 将对象移到原点
    center = (min_corner + max_corner) / 2
    obj.location -= center
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    # 保存归一化后的 OBJ 文件
    save_path_dir = os.path.join(save_Abs_path, obj_path.split('/')[5])
    if not os.path.exists(save_path_dir):
        os.mkdir(save_path_dir)
    bpy.ops.export_scene.obj(filepath=os.path.join(save_path_dir, 'normalized_textured.obj'))

def normalize_glbfile(input)