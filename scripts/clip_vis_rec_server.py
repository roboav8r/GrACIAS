#!/usr/bin/env python3

from situated_hri_interfaces.srv import ObjectVisRec

from PIL import Image as PILImage
import cv2
import cv_bridge

import rclpy
from rclpy.node import Node


class CLIPVisRecServer(Node):

    def __init__(self):
        super().__init__('clip_vis_rec_server')
        self.srv = self.create_service(ObjectVisRec, 'clip_object_rec', self.clip_rec_callback)

        self.bridge = cv_bridge.CvBridge()

        # Generate object att/state variable dictionary
        self.object_params = {}
        self.declare_parameter('object_classes', rclpy.Parameter.Type.STRING_ARRAY)
        self.object_classes = self.get_parameter('object_classes').get_parameter_value().string_array_value

        for obj in self.object_classes:

            self.object_params[obj] = {}

            self.object_params[obj]['attributes'] = {}
            self.declare_parameter(obj + '.attributes.variables', rclpy.Parameter.Type.STRING_ARRAY)
            att_vars = self.get_parameter(obj + '.attributes.variables').get_parameter_value().string_array_value

            for att_var in att_vars:
                self.declare_parameter(obj + '.attributes.' + att_var + '.labels', rclpy.Parameter.Type.STRING_ARRAY)
                self.declare_parameter(obj + '.attributes.' + att_var + '.descriptions', rclpy.Parameter.Type.STRING_ARRAY)

                self.object_params[obj]['attributes'][att_var] = {}
                self.object_params[obj]['attributes'][att_var]['labels'] = self.get_parameter(obj + '.attributes.' + att_var + '.labels').get_parameter_value().string_array_value
                self.object_params[obj]['attributes'][att_var]['descriptions'] = self.get_parameter(obj + '.attributes.' + att_var + '.descriptions').get_parameter_value().string_array_value

            self.object_params[obj]['states'] = {}
            self.declare_parameter(obj + '.states.variables', rclpy.Parameter.Type.STRING_ARRAY)
            state_vars = self.get_parameter(obj + '.states.variables').get_parameter_value().string_array_value

            for state_var in state_vars:
                self.declare_parameter(obj + '.states.' + state_var + '.labels', rclpy.Parameter.Type.STRING_ARRAY)
                self.declare_parameter(obj + '.states.' + state_var + '.descriptions', rclpy.Parameter.Type.STRING_ARRAY)

                self.object_params[obj]['states'][state_var] = {}
                self.object_params[obj]['states'][state_var]['labels'] = self.get_parameter(obj + '.states.' + state_var + '.labels').get_parameter_value().string_array_value
                self.object_params[obj]['states'][state_var]['descriptions'] = self.get_parameter(obj + '.states.' + state_var + '.descriptions').get_parameter_value().string_array_value

        self.get_logger().info("Object dictionary: %s" % self.object_params)

    def clip_rec_callback(self, req, resp):

        cv_image = self.bridge.imgmsg_to_cv2(req.image, desired_encoding='passthrough')
        pil_im = PILImage.fromarray(cv_image)
        pil_im.show()
        
        self.get_logger().info('Incoming request\nid: %d img: %d' % (req.person_id, req.image))

        return resp


def main(args=None):
    rclpy.init(args=args)

    clip_vis_rec_server = CLIPVisRecServer()

    rclpy.spin(clip_vis_rec_server)

    rclpy.shutdown()


if __name__ == '__main__':
    main()