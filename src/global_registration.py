import open3d as o3d
import numpy as np
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    #source_temp.paint_uniform_color([1, 0.706, 0])
    #target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    #o3d.io.write_point_cloud("../data/nero2.ply", source, write_ascii=True)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size, path_cad):
    #print(":: Load two point clouds and disturb initial pose.")
    #target = o3d.io.read_point_cloud("../data/cloud_obj.pcd")
    #source = o3d.io.read_point_cloud("../data/oil_m.ply")#cad_our.pcd")
    target = o3d.io.read_point_cloud("/home/monica/ros_catkin_ws_mine/src/detect_mechanical_parts/data/cloud_obj.pcd")
    source = o3d.io.read_point_cloud(path_cad)#cad_our.pcd") # !!!!!! MANDATORY: THE CAD MUST BE THE SOURCE
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    #trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                         [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    #source.transform(trans_init)
    #draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    #print(":: RANSAC registration on downsampled point clouds.")
    #print("   Since the downsampling voxel size is %.3f," % voxel_size)
    #print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    #print(":: Point-to-plane ICP registration is applied on original point")
    #print("   clouds to refine the alignment. This time we use a strict")
    #print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def main_method(path_cad):
    voxel_size = 0.005  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, path_cad)
        
    
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    #print(result_ransac)
    #print(result_ransac.transformation)
    #draw_registration_result(source_down, target_down, result_ransac.transformation)


    result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                    voxel_size, result_ransac)
    #print(result_icp)
    #print(result_icp.transformation)
    #draw_registration_result(source, target, result_icp.transformation)

    # Controllo sulla bonta' e reiterazione del metodo
    attemps = 50 #100
    times = 1
    while(result_icp.fitness <= 0.7 and times < attemps):
        result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
        #print(result_ransac)
        result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                    voxel_size, result_ransac)
        #print(result_icp)
        times = times + 1
    

    #print("DRAW")
    #print(times)
    #print(result_icp.fitness)
    #print(result_icp.transformation)
    #draw_registration_result(source, target, result_icp.transformation)
    return [result_icp.fitness, result_icp.transformation]


# STIMA DELLA POSA
def computePose(objMcad):
    #wMe = np.array([[-0.112324, 0.993599, 0.0111902, -0.0613361],
    #                [0.993653, 0.112363, -0.00296296, 0.34795],
    #                [-0.00420137, 0.0107864, -0.999933, 0.382541],
    #                [0, 0, 0, 1]])

    wMe = np.array([[-0.102082, 0.994612, 0.0175242, -0.104391],
                    [0.993976, 0.102687, -0.0380448, 0.455123],
                    [-0.0396393, 0.013535, -0.999122, 0.217175],
                    [0, 0, 0, 1]])

    eMc = np.array([[0.01999945272, -0.9990596861, 0.03846772089, 0.05834485203],
                    [0.9997803621, 0.01974315714, -0.007031030191, -0.03476564525],
                    [0.006264944557, 0.03859988867, 0.999235107, -0.06760482074],
                    [0, 0, 0, 1]])

    cMcad = np.array([[1, 0, 0, 0.11],# 0.087],
                      [0, 1, 0, -0.266],# 0.043],
                      [0, 0, 1, -0.015], #  0.28],
                      [0, 0, 0, 1]])

    Tf = np.array([[0, -1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])


    print()
    #cMobj = np.matmul(cMcad, cadMobj)
    cMobj = np.matmul(objMcad, cMcad)
    print("cMobj")
    print(cMobj)

    eMobj = np.matmul(eMc, cMobj)
    print("eMobj")
    print(eMobj)

    wMobj = np.matmul(wMe, eMobj)
    print("wMobj")
    print(wMobj)
    
    wMobj_f = wMobj
    wMobj_f[:3,:3] = np.matmul(Tf[:3,:3], wMobj[:3,:3])

    print(wMobj_f[0,3], wMobj_f[1,3], wMobj_f[2,3])
    print("Rd")
    print(wMobj_f[0,0], wMobj_f[0,1], wMobj_f[0,2])
    print(wMobj_f[1,0], wMobj_f[1,1], wMobj_f[1,2])
    print(wMobj_f[2,0], wMobj_f[2,1], wMobj_f[2,2])
    #print("wMobj_f")
    #print(wMobj_f)

#computePose(result_icp.transformation)

# MAIN
def main():
    names = ["../data/cad_vari/obj_12.ply",
            "../data/cad_vari/obj_13.ply",
            "../data/cad_vari/obj_14.ply",
            "../data/cad_vari/obj_15.ply",
            "../data/cad_vari/obj89_910_pti.ply"]

    fits = []

    for v in range(5):
        fit = main_method(names[v])
        fits.append(fit)

    for v in range (5):
        print(names[v] + " " + str(fits[v]))

if __name__ == "__main__":
    main()
                            

