from tkinter import Y
from scipy.io import loadmat, savemat
import numpy as np
import os
#import open3d as o3d
matDir = '/home/ndprra/TI_ws/ti_ouster/bhpcd'
"""for i in os.listdir(matDir):
    if i.endswith('.mat'):
        p = loadmat(os.path.join(matDir,i))
    #print(len(p['y']))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p['y'][:,:3])
        o3d.io.write_point_cloud(os.path.join(matDir,i.split('.')[0]+'.pcd'), pcd)
        print(pcd)"""

    #o3d.visualization.draw_geometries([pcd])



import copy
import numpy
import open3d
import pyquaternion


class Keyframe(object):
	def __init__(self, id, cloud, odom):
		self.id = id
		self.odom = copy.deepcopy(odom)
		self.cloud = copy.deepcopy(cloud)
		self.transformed = copy.deepcopy(cloud)

		self.node = open3d.pipelines.registration.PoseGraphNode(odom)
		self.update_transformed()

	def update_transformed(self):
		self.transformed.points = self.cloud.points
		self.transformed.normals = self.cloud.normals
		self.transformed.transform(self.node.pose)


class GraphSLAM(object):
	def __init__(self):
		self.graph = open3d.pipelines.registration.PoseGraph()

		self.keyframes = []
		self.last_frame_transformation = numpy.identity(4)

		self.keyframe_angle_thresh_deg = 15.0
		self.keyframe_trans_thresh_m = 1.0

		self.vis = open3d.visualization.Visualizer()
		self.vis.create_window()

	def update(self, cloud):
		#cloud = open3d.geometry.voxel_down_sample(cloud, voxel_size=0.05)

		if not len(self.keyframes):
			self.keyframes.append(Keyframe(0, cloud, numpy.identity(4)))
			self.vis.add_geometry(self.keyframes[-1].transformed)
			self.graph.nodes.append(self.keyframes[-1].node)
			return

		if not self.update_keyframe(cloud):
			return

		print ('optimizing...')
		option = open3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=0.3, edge_prune_threshold=0.05, reference_node=0)
		print(open3d.pipelines.registration.global_optimization(self.graph, open3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(), open3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option))

		for keyframe in self.keyframes:
			keyframe.update_transformed()
        #cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
		self.vis.update_geometry(cloud)
		self.vis.poll_events()
		self.vis.update_renderer()

	def update_keyframe(self, cloud):
		criteria = open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
		reg = open3d.pipelines.registration.registration_icp(cloud, self.keyframes[-1].cloud, 1.0, self.last_frame_transformation, open3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria=criteria)

		angle = pyquaternion.Quaternion(matrix=reg.transformation[:3, :3]).degrees
		trans = numpy.linalg.norm(reg.transformation[:3, 3])

		if abs(angle) < self.keyframe_angle_thresh_deg and abs(trans) < self.keyframe_trans_thresh_m:
			self.last_frame_transformation = reg.transformation
			return False

		odom = reg.transformation.dot(self.keyframes[-1].odom)
		self.keyframes.append(Keyframe(len(self.keyframes), cloud, odom))
		self.graph.nodes.append(self.keyframes[-1].node)
		self.vis.add_geometry(self.keyframes[-1].transformed)

		self.last_frame_transformation = numpy.identity(4)

		information = open3d.pipelines.registration.get_information_matrix_from_point_clouds(self.keyframes[-1].cloud, self.keyframes[-2].cloud, 1.0, reg.transformation)
		edge = open3d.pipelines.registration.PoseGraphEdge(self.keyframes[-1].id, self.keyframes[-2].id, reg.transformation, information, uncertain=False)

		self.graph.edges.append(edge)

		return True

	def generate_map(self):
		map_cloud = open3d.PointCloud()
		for keyframe in self.keyframes:
			transformed = copy.deepcopy(keyframe.cloud)
			transformed.transform(keyframe.node.pose)
			map_cloud += transformed

		return open3d.voxel_down_sample(map_cloud, voxel_size=0.95)


def main():
	dataset_path = matDir
	cloud_files = sorted([dataset_path + '/' + x for x in os.listdir(dataset_path) if '.pcd' in x])

	graph_slam = GraphSLAM()

	for cloud_file in cloud_files[:1500]:
		print(cloud_file)
		cloud = open3d.io.read_point_cloud(cloud_file)
		graph_slam.update(cloud)


if __name__ == '__main__':
	main()
