from situated_interaction.srv import PersonVisRec

from PIL import Image as PILImage
import cv2
import cv_bridge

import rclpy
from rclpy.node import Node


class VisRecPersonServer(Node):

    def __init__(self):
        super().__init__('person_vis_rec_server')
        self.srv = self.create_service(PersonVisRec, 'person_vis_rec', self.person_vis_rec_callback)

        self.bridge = cv_bridge.CvBridge()

    def person_vis_rec_callback(self, req, resp):

        cv_image = self.bridge.imgmsg_to_cv2(req.image, desired_encoding='passthrough')
        pil_im = PILImage.fromarray(cv_image)
        pil_im.show()
        
        self.get_logger().info('Incoming request\nid: %d img: %d' % (req.person_id, req.image))

        return resp


def main(args=None):
    rclpy.init(args=args)

    vis_rec_person_server = VisRecPersonServer()

    rclpy.spin(vis_rec_person_server)

    rclpy.shutdown()


if __name__ == '__main__':
    main()