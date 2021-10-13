import open3d as o3d
import numpy as np
import copy
from global_registration import main_method, draw_registration_result

db = []
names_view = []

def caricaGraspingPose(filename):
    file1 = open(filename, 'r')
    lines = file1.readlines() 
    for i in range(1, len(lines)):
        line = lines[i]
        line = line.rstrip("\n")
        array = line.split(" ")
        db.append(array)
        names_view.append(array[0])

print("Load Grasping Poses")
caricaGraspingPose("/home/monica/ros_catkin_ws_mine/src/detect_mechanical_parts/data/views/grasp_pose.txt")



def get_matrix(normal, y_w):
    R = np.array([[-0.094378, 0.995485, -0.009164],
                [0.995355, 0.094186, -0.019453],
                [-0.018502, -0.010957, -0.999769]])
    v_ = R.dot(normal)
    x = np.cross(normal,y_w)
    x = x/np.linalg.norm(x)
    y = np.cross(normal, x)
    y = y/np.linalg.norm(y)
    return np.array([x, y, normal]).T 

def start_comparison():
    #print("Load Grasping Poses")
    #caricaGraspingPose("/home/monica/ros_catkin_ws_mine/src/detect_mechanical_parts/data/views/grasp_pose.txt")

    #print("Perform comparison")
    fits = []
    transformations = []

    for v in range(len(names_view)):
        print("Comparison ... {} %".format((v+1)*9.1))
        [fit, transformation] = main_method(names_view[v])
        fits.append(fit)
        transformations.append(transformation)
        if (fit >= 0.9):
            break

    #for v in range(len(names_view)):
    #    print(names_view[v] + " " + str(fits[v]))

    #print("Find Best Match")
    best_index = fits.index(max(fits)) #0-based
    point_ = db[best_index][1:4]
    point = [float(i) for i in point_]
    vx_ = db[best_index][4:7]
    vx = [float(i) for i in vx_]
    vy_ = db[best_index][7:10]
    vy = [float(i) for i in vy_]
    vz_ = db[best_index][10:]
    vz = [float(i) for i in vz_]

    To = np.eye(4)
    To[:3,0] = vx
    To[:3,1] = vy
    To[:3,2] = vz
    To[:3,3] = point
    oTcad = transformations[best_index]

    target = o3d.io.read_point_cloud("/home/monica/ros_catkin_ws_mine/src/detect_mechanical_parts/data/cloud_obj.pcd")
    cad = o3d.io.read_point_cloud(names_view[best_index])
    cad.transform(oTcad)
    o3d.visualization.draw_geometries([cad, target])


    print("Compute pose in world frame")
    '''
    wMe = np.array([[-0.0973994, 0.994866, 0.0271286, -0.0720888],
                    [0.993369, 0.0988494, -0.0585517, 0.374031],
                    [-0.0609328, 0.0212458,	-0.997916, 0.292703],
                    [0,	0, 0, 1]])
    '''
    wMe = np.array([[-0.000830282, 0.999947, 0.00930422, -0.0155212],
                    [0.99961, 0.00108627, -0.0275425, 0.38384],
                    [-0.0275512, 0.00927772, -0.999577,	0.438877],
                    [0, 0, 0, 1]])

    eMc = np.array([[0.01999945272, -0.9990596861, 0.03846772089, 0.05834485203],
                    [0.9997803621, 0.01974315714, -0.007031030191, -0.03476564525],
                    [0.006264944557, 0.03859988867, 0.999235107, -0.06760482074],
                    [0, 0, 0, 1]])

    T_to_rob = wMe.dot(eMc.dot(oTcad.dot(To)))

    print("-------------------------------------------------")
    print(T_to_rob[0,3], T_to_rob[1,3], T_to_rob[2,3])
    print("Rd")
    print(T_to_rob[0,0], T_to_rob[0,1], T_to_rob[0,2])
    print(T_to_rob[1,0], T_to_rob[1,1], T_to_rob[1,2])
    print(T_to_rob[2,0], T_to_rob[2,1], T_to_rob[2,2])
    return T_to_rob

if __name__ == "__main__":
    print("Main method pf main_registration.py script")