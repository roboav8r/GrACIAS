#!/usr/bin/env python3

import numpy as np
from scipy.optimize import linear_sum_assignment

from rclpy.time import Duration
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped

def compute_az_match(self, az, az_frame):
    similarity_vector = np.zeros(len(self.semantic_objects.keys()))

    # Ensure az in range [-pi, pi] # TODO make this a helper function, or make sure output of az speech rec in -pi,pi
    while np.abs(az) > np.pi:
        az -= np.sign(az)*2*np.pi

    for ii,key in enumerate(self.semantic_objects.keys()):
        # Convert object position into az frame
        tracker_az_tf = self.tf_buffer.lookup_transform(az_frame,self.tracker_frame,Duration(seconds=.1))
        pos_in_tracker_frame = PointStamped()
        pos_in_tracker_frame.point.x = self.semantic_objects[key].pos_x
        pos_in_tracker_frame.point.y = self.semantic_objects[key].pos_y
        pos_in_tracker_frame.point.z = self.semantic_objects[key].pos_z
        pos_in_az_frame = do_transform_point(pos_in_tracker_frame,tracker_az_tf)

        # Compute azimuth in az frame
        obj_az = np.arctan2(pos_in_az_frame.point.y,pos_in_az_frame.point.x) # in range [-pi,pi]

        similarity_vector[ii] = np.linalg.norm([az - obj_az])

    # Return match if below angular match threshold
    if (similarity_vector.size > 0) and (similarity_vector[np.argmin(similarity_vector)] < self.ang_match_threshold):
        return list(self.semantic_objects.keys())[np.argmin(similarity_vector)] if similarity_vector[np.argmin(similarity_vector)] < self.ang_match_threshold else None
    else:
        return None
    

def compute_az_from_pos(tf_buffer,target_frame,source_frame,object):
    # Target frame = what you want to have, usually microphone array
    # Source frame = what you do have, e.g. object pos in tracker frame

    # Compute position in target frame
    source_target_tf = tf_buffer.lookup_transform(target_frame,source_frame,Duration(seconds=.1))
    pos_in_source_frame = PointStamped()
    pos_in_source_frame.point.x = object.pos_x
    pos_in_source_frame.point.y = object.pos_y
    pos_in_source_frame.point.z = object.pos_z
    pos_in_target_frame = do_transform_point(pos_in_source_frame,source_target_tf)

    # Compute and return azimuth in target frame
    az_in_target_frame = np.arctan2(pos_in_target_frame.point.y,pos_in_target_frame.point.x)
    return az_in_target_frame

def compute_delta_az(az_1, az_2):
    delta_az = az_1 - az_2

    while np.abs(delta_az) > np.pi:
        delta_az -= np.sign(delta_az)*2*np.pi

    return np.abs(delta_az)

def compute_pos_match(self, pos):
    similarity_vector = np.zeros(len(self.semantic_objects.keys()))

    for ii,key in enumerate(self.semantic_objects.keys()): 
        similarity_vector[ii] = np.linalg.norm([pos.x - self.semantic_objects[key].pos_x, pos.y - self.semantic_objects[key].pos_y, pos.z - self.semantic_objects[key].pos_z])

    return list(self.semantic_objects.keys())[np.argmin(similarity_vector)] if self.semantic_objects.keys() else -1

def compute_delta_pos(pos_1, pos_2):
    # Euclidean distance between positions
    return np.linalg.norm(pos_1 - pos_2)

def hungarian_assignment(cost_matrix):
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    # tracker.det_asgn_idx, tracker.trk_asgn_idx = list(row_idx), list(col_idx)
    return np.stack((row_idx, col_idx), axis=1)

def greedy_assignment(cost_matrix):
    # association in the greedy manner
    # refer to https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/master/main.py
    # and also https://github.com/xinshuoweng/AB3DMOT/blob/61f3bd72574093e367916c757b4747ca445f978c/AB3DMOT_libs/matching.py 

    num_dets, num_trks = cost_matrix.shape[0], cost_matrix.shape[1]

    # sort all costs and then convert to 2D
    distance_1d = cost_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)

    # assign matches one by one given the sorting, but first come first serves
    det_matches_to_trk = [-1] * num_dets
    trk_matches_to_det = [-1] * num_trks
    matched_indices = []
    for sort_i in range(index_2d.shape[0]):
        det_id = int(index_2d[sort_i][0])
        trk_id = int(index_2d[sort_i][1])

        # if both id has not been matched yet
        if trk_matches_to_det[trk_id] == -1 and det_matches_to_trk[det_id] == -1:
            trk_matches_to_det[trk_id] = det_id
            det_matches_to_trk[det_id] = trk_id
            matched_indices.append([det_id, trk_id])

    return np.asarray(matched_indices)

def solve_assignment_matrix(method, cost_matrix, assignment_threshold):
    
    # Compute assignment vectors: each vector contains indices of matched detections and matched tracks
    if method == 'hungarian':
        assignment_indices = hungarian_assignment(cost_matrix) 	
        
    elif method == 'greedy':
        assignment_indices = greedy_assignment(cost_matrix) 		
    
    else:
        raise TypeError('Invalid assignment algorithm: %s' % method)

	# Do not consider matches with cost >= threshold
    # cost matrix: rows = detection index, columns = track index
    assignments = []
    for m in assignment_indices:
        if (cost_matrix[m[0], m[1]] < assignment_threshold):
            assignments.append(m.reshape(1, 2))
            
    if len(assignments) == 0: 
        assignments = np.empty((0, 2),dtype=int)
    
    else: assignments = np.concatenate(assignments, axis=0)

    return assignments
