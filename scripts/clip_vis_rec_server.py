#!/usr/bin/env python3

import torch
import clip

from PIL import Image as PILImage
import cv2
import cv_bridge

import rclpy
from rclpy.node import Node

from situated_hri_interfaces.msg import CategoricalDistribution
from situated_hri_interfaces.srv import ObjectVisRec

class CLIPVisRecServer(Node):

    def __init__(self):

        # ROS objects
        super().__init__('clip_vis_rec_server')
        self.srv = self.create_service(ObjectVisRec, 'clip_object_rec', self.clip_rec_callback)

        # Generate CLIP model and device
        self.bridge = cv_bridge.CvBridge()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.declare_parameter('clip_model', rclpy.Parameter.Type.STRING)
        self.clip_model = self.get_parameter('clip_model').get_parameter_value().string_value
        self.model, self.preprocess = clip.load(self.clip_model, device=self.device)

        # Generate object att/state variable dictionary
        self.create_obj_param_dict()
        # self.get_logger().info(f"PARAM DICT: {self.object_params}")

    def create_obj_param_dict(self):
        self.object_params = {}
        self.declare_parameter('object_classes', rclpy.Parameter.Type.STRING_ARRAY)
        self.object_classes = self.get_parameter('object_classes').get_parameter_value().string_array_value

        for obj in self.object_classes:

            self.object_params[obj] = {}

            self.object_params[obj]['attributes'] = {}
            self.declare_parameter(obj + '.attributes.variables', rclpy.Parameter.Type.STRING_ARRAY)
            att_vars = self.get_parameter(obj + '.attributes.variables').get_parameter_value().string_array_value

            for att_var in att_vars:
                if att_var != '':
                    self.declare_parameter(obj + '.attributes.' + att_var + '.labels', rclpy.Parameter.Type.STRING_ARRAY)
                    self.declare_parameter(obj + '.attributes.' + att_var + '.descriptions', rclpy.Parameter.Type.STRING_ARRAY)

                    self.object_params[obj]['attributes'][att_var] = {}
                    self.object_params[obj]['attributes'][att_var]['labels'] = self.get_parameter(obj + '.attributes.' + att_var + '.labels').get_parameter_value().string_array_value
                    self.object_params[obj]['attributes'][att_var]['descriptions'] = self.get_parameter(obj + '.attributes.' + att_var + '.descriptions').get_parameter_value().string_array_value
                    self.object_params[obj]['attributes'][att_var]['text_tokens'] = clip.tokenize(self.object_params[obj]['attributes'][att_var]['descriptions']).to(self.device)
                    self.object_params[obj]['attributes'][att_var]['text_features'] = self.model.encode_text(self.object_params[obj]['attributes'][att_var]['text_tokens'])

            self.object_params[obj]['states'] = {}
            self.declare_parameter(obj + '.states.variables', rclpy.Parameter.Type.STRING_ARRAY)
            state_vars = self.get_parameter(obj + '.states.variables').get_parameter_value().string_array_value

            for state_var in state_vars:
                self.declare_parameter(obj + '.states.' + state_var + '.labels', rclpy.Parameter.Type.STRING_ARRAY)
                self.declare_parameter(obj + '.states.' + state_var + '.descriptions', rclpy.Parameter.Type.STRING_ARRAY)

                self.object_params[obj]['states'][state_var] = {}
                self.object_params[obj]['states'][state_var]['labels'] = self.get_parameter(obj + '.states.' + state_var + '.labels').get_parameter_value().string_array_value
                self.object_params[obj]['states'][state_var]['descriptions'] = self.get_parameter(obj + '.states.' + state_var + '.descriptions').get_parameter_value().string_array_value
                self.object_params[obj]['states'][state_var]['text_tokens'] = clip.tokenize(self.object_params[obj]['states'][state_var]['descriptions']).to(self.device)
                self.object_params[obj]['states'][state_var]['text_features'] = self.model.encode_text(self.object_params[obj]['states'][state_var]['text_tokens'])

            # Get comms
            self.object_params[obj]['comms'] = {}

            self.declare_parameter(obj + '.comms.gesture_descriptions', rclpy.Parameter.Type.STRING_ARRAY)
            self.object_params[obj]['comms']['gesture_descriptions'] = self.get_parameter(obj + '.comms.gesture_descriptions').get_parameter_value().string_array_value

            if self.object_params[obj]['comms']['gesture_descriptions'] != ['']:
                self.declare_parameter(obj + '.comms.labels', rclpy.Parameter.Type.STRING_ARRAY)
                self.object_params[obj]['comms']['labels']  = self.get_parameter(obj + '.comms.labels').get_parameter_value().string_array_value
                self.object_params[obj]['comms']['text_tokens'] = clip.tokenize(self.object_params[obj]['comms']['gesture_descriptions']).to(self.device)
                self.object_params[obj]['comms']['text_features'] = self.model.encode_text(self.object_params[obj]['comms']['text_tokens'])

    def clip_rec_callback(self, req, resp):

        # self.get_logger().info("Got req to identify %s %s" % (req.class_string, req.object_id))

        resp.class_string = req.class_string
        resp.object_id = req.object_id

        with torch.no_grad():

            self.cv_image_bgr = self.bridge.imgmsg_to_cv2(req.image,desired_encoding="bgr8")
            self.cv_image_rgb = cv2.cvtColor(self.cv_image_bgr,cv2.COLOR_BGR2RGB)
            self.pil_image = PILImage.fromarray(self.cv_image_rgb)
            # pil_image.save('/home/jd/sit_int_ws/server_pil_img_%s.png' % req.object_id)
            self.clip_image = self.preprocess(self.pil_image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(self.clip_image)

            # Iterate through attributes and compute probabilities
            for att in req.attributes_to_estimate:
                att_dist = CategoricalDistribution()
                att_dist.variable = att
                att_dist.categories = self.object_params[req.class_string]['attributes'][att]['labels']
                self.object_params[req.class_string]['attributes'][att]['text_features'] = self.model.encode_text(self.object_params[req.class_string]['attributes'][att]['text_tokens'])
                logits_per_image, _ = self.model(self.clip_image, self.object_params[req.class_string]['attributes'][att]['text_tokens'])
                att_dist.probabilities = logits_per_image.softmax(dim=-1).cpu().numpy()[0].tolist()
                resp.attributes.append(att_dist)

            # Iterate through states and compute probabilities
            for state in req.states_to_estimate:
                state_dist = CategoricalDistribution()
                state_dist.variable = state
                state_dist.categories = self.object_params[req.class_string]['states'][state]['labels']
                self.object_params[req.class_string]['states'][state]['text_features'] = self.model.encode_text(self.object_params[req.class_string]['states'][state]['text_tokens'])
                logits_per_image, _ = self.model(self.clip_image, self.object_params[req.class_string]['states'][state]['text_tokens'])
                state_dist.probabilities = logits_per_image.softmax(dim=-1).cpu().numpy()[0].tolist()
                resp.states.append(state_dist)

            if req.estimate_comms:
                comm_dist = CategoricalDistribution()
                comm_dist.variable = 'gesture_comms'
                comm_dist.categories = self.object_params[req.class_string]['comms']['labels']
                self.object_params[req.class_string]['attributes'][att]['text_features'] = self.model.encode_text(self.object_params[req.class_string]['attributes'][att]['text_tokens'])
                logits_per_image, _ = self.model(self.clip_image, self.object_params[req.class_string]['comms']['text_tokens'])
                comm_dist.probabilities = logits_per_image.softmax(dim=-1).cpu().numpy()[0].tolist()
                resp.comms = comm_dist
            
            resp.stamp = req.stamp

            # cv2.imwrite('/home/jd/sit_int_ws/server_img_%s.png' % req.object_id,self.cv_image_bgr)
            return resp


def main(args=None):
    rclpy.init(args=args)

    clip_vis_rec_server = CLIPVisRecServer()
    rclpy.spin(clip_vis_rec_server)

    clip_vis_rec_server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()