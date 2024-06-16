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