# 3Dgeneration-astribot
## render_and_topartialpc folder
includes:
1) normalized the 3D object
2) render the multiview RGB images from 3D object and EXR file(depth file) from 3D object
3) This is single-thread running

## parallel_render
includes:
same with *render_and_topartialpc* folder
but this is for parallel computing

## parallel_construct_groundtruth_pc
includes:
1) generate colored point cloud from 3D object
2) filter script of meshlab
3) This is for the parallel computing

