#!/usr/bin/env python3
import sys
#print(sys.version_info[0])
#print(sys.version_info[1])
import os
#from pathlib import Path
import cv2 as cv
import open3d as o3d

colors = []
classi = []

class Utils:

    @staticmethod
    def hex_to_rgb(value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    @staticmethod
    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % rgb

    @staticmethod
    def load_color_match(path):
        f = open(Utils.absolute_path(path), 'r')
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split()
            colors.append(tokens[2])
            classi.append(tokens[0])

    @staticmethod
    def get_color(class_id):
        return colors[class_id]
    
    @staticmethod
    def get_detection_class(class_id):
        return classi[class_id]

    @staticmethod
    def root():
        #return Path(__file__).parent.parent
        return os.path.abspath(os.path.join(__file__, "../../"))
    
    @staticmethod
    def absolute_path(rel_path):
        return os.path.join(Utils.root(), rel_path.replace('/', os.sep))

    @staticmethod
    def load_classes(path):
        labels = []
        with open(Utils.absolute_path(path), "r") as f:
            labels = [cname.strip() for cname in f.readlines()]
        return labels

    @staticmethod
    def is_cuda_cv(): 
        try:
            count = cv.cuda.getCudaEnabledDeviceCount()
            if count > 0:
                return 1
            else:
                return 0
        except:
            return 0

    @staticmethod
    def __draw_registration_result(source, target, transformation):
        import copy
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    @staticmethod
    def __preprocess_point_cloud(pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    @staticmethod
    def __prepare_dataset(voxel_size, source_file, target_file):
        source = o3d.io.read_point_cloud(source_file) #"data_glob/obj.pcd")
        target = o3d.io.read_point_cloud(target_file) #"data_glob/cad_obj.ply")
        source_down, source_fpfh = Utils.__preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = Utils.__preprocess_point_cloud(target, voxel_size)
        return source, target, source_down, target_down, source_fpfh, target_fpfh

    @staticmethod
    def __execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        criterio = o3d.registration.RANSACConvergenceCriteria(max_iteration=100000, max_validation=100)
        result = o3d.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False),
            4, [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
        return result
    
    @staticmethod
    def __refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
        distance_threshold = voxel_size * 0.4
        result = o3d.registration.registration_icp(
            source, target, distance_threshold, result_ransac.transformation,
            o3d.registration.TransformationEstimationPointToPlane())
        return result

    @staticmethod
    def overlap(source_file, target_file):
        voxel_size = 0.05  # means 5cm for this dataset
        source, target, source_down, target_down, source_fpfh, target_fpfh = Utils.__prepare_dataset(voxel_size, source_file, target_file)
        result_ransac = Utils.__execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        result_icp = Utils.__refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac)
        Utils.__draw_registration_result(source, target, result_icp.transformation)
        print(result_icp)
        return result_icp.transformation

    @staticmethod
    def r2quat(R):
        import numpy as np
        #q(0) e` l'elemento "scalare" in uscita
        q = [0, 0, 0, 0]
        r1 = R[0,0]
        r2 = R[1,1]
        r3 = R[2,2]
        r4 = r1 + r2 + r3
        j = 1
        rj = r1
        if r2>rj:
            j = 2
            rj = r2
        if r3>rj:
            j = 3
            rj = r3
        if r4>rj:
            j = 4
            rj = r4
        pj = 2* np.sqrt(1+2*rj-r4)
        if j == 1:
            p1 = pj/4
            p2 = (R[1,0]+R[0,1])/pj
            p3 = (R[0,2]+R[2,0])/pj
            p4 = (R[2,1]-R[1,2])/pj
        elif j == 2:
            p1 = (R[1,0]+R[0,1])/pj
            p2 = pj/4;
            p3 = (R[2,1]+R[1,2])/pj
            p4 = (R[0,2]-R[2,0])/pj
        elif j == 3:
            p1 = (R[0,2]+R[2,0])/pj
            p2 = (R[2,1]+R[1,2])/pj
            p3 = pj/4;
            p4 = (R[1,0]-R[0,1])/pj
        else:
            p1 = (R[2,1]-R[1,2])/pj
            p2 = (R[0,2]-R[2,0])/pj
            p3 = (R[1,0]-R[0,1])/pj
            p4 = pj/4
        if p4 > 0:
            q[1] =  p1
            q[2] =  p2
            q[3] =  p3
            q[0] =  p4
        else:
            q[1] =  -p1
            q[2] =  -p2
            q[3] =  -p3
            q[0] =  -p4
        return q



