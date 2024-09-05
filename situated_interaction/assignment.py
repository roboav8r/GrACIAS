#!/usr/bin/env python3

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

def compute_pos_match(self, pos):
    similarity_vector = np.zeros(len(self.semantic_objects.keys()))

    for ii,key in enumerate(self.semantic_objects.keys()): 
        similarity_vector[ii] = np.linalg.norm([pos.x - self.semantic_objects[key].pos_x, pos.y - self.semantic_objects[key].pos_y, pos.z - self.semantic_objects[key].pos_z])

    return list(self.semantic_objects.keys())[np.argmin(similarity_vector)] if self.semantic_objects.keys() else -1