#!/usr/bin/env python3
from situated_hri_interfaces.msg import HierarchicalCommand, HierarchicalCommands

from diagnostic_msgs.msg import KeyValue
from foxglove_msgs.msg import SceneUpdate, SceneEntity, TextPrimitive, LinePrimitive
from geometry_msgs.msg import Point

def foxglove_visualization(semantic_fusion_node):
    semantic_fusion_node.scene_out_msg = SceneUpdate()

    for idx in semantic_fusion_node.semantic_objects.keys():
        obj = semantic_fusion_node.semantic_objects[idx]
        entity_msg = SceneEntity()

        # Populate entity message with header / object data
        entity_msg.frame_id = semantic_fusion_node.tracks_msg.header.frame_id
        entity_msg.timestamp = semantic_fusion_node.tracks_msg.header.stamp
        entity_msg.id = str(obj.track_id)
        entity_msg.frame_locked = True
        entity_msg.lifetime.nanosec = int(semantic_fusion_node.pub_loop_time_sec*1000000000)

        # Add callout lines
        line = LinePrimitive()
        line.thickness = .01
        line.color.r = 1.
        line.color.g = 1.
        line.color.b = 1.
        line.color.a = 1.
        top_point = Point()
        top_point.x = obj.pos_x + semantic_fusion_node.x_label_offset
        top_point.y = obj.pos_y + semantic_fusion_node.y_label_offset
        top_point.z = obj.pos_z + semantic_fusion_node.z_label_offset        
        mid_point = Point()
        mid_point.x = obj.pos_x + semantic_fusion_node.x_label_offset
        mid_point.y = obj.pos_y + semantic_fusion_node.y_label_offset
        mid_point.z = obj.pos_z + 0.5*semantic_fusion_node.z_label_offset
        bottom_point = Point()
        bottom_point.x = obj.pos_x
        bottom_point.y = obj.pos_y
        bottom_point.z = obj.pos_z
        line.points.append(top_point)
        line.points.append(mid_point)
        line.points.append(bottom_point)
        entity_msg.lines.append(line)

        # Add text box
        text = TextPrimitive()
        text.billboard = True
        text.font_size = 16.
        text.scale_invariant = True
        text.color.a = 1.0
        text.pose.position.x = obj.pos_x + semantic_fusion_node.x_label_offset
        text.pose.position.y = obj.pos_y + semantic_fusion_node.y_label_offset
        text.pose.position.z = obj.pos_z + semantic_fusion_node.z_label_offset
        text.text = "%s #%s:\n" % (obj.class_string, obj.track_id)
        
        for att in obj.attributes:
            text.text += "%s: %s %2.0f%%\n" % (att, obj.attributes[att].var_labels[obj.attributes[att].probs.argmax()], 100*obj.attributes[att].probs(obj.attributes[att].probs.argmax()))

        for state in obj.states:
            text.text += "%s: %s %2.0f%%\n" % (state, obj.states[state].var_labels[obj.states[state].probs.argmax()], 100*obj.states[state].probs(obj.states[state].probs.argmax()))

        text.text += "command: %s %2.0f%%\n" % (obj.comms.var_labels[obj.comms.probs.argmax()], 100*obj.comms.probs(obj.comms.probs.argmax()))
        entity_msg.texts.append(text)

        semantic_fusion_node.scene_out_msg.entities.append(entity_msg)

    semantic_fusion_node.semantic_scene_pub.publish(semantic_fusion_node.scene_out_msg)

def publish_hierarchical_commands(semantic_fusion_node):
    semantic_fusion_node.hierarchical_cmds_msg = HierarchicalCommands()
    semantic_fusion_node.hierarchical_cmds_msg.header.frame_id = semantic_fusion_node.tracker_frame
    semantic_fusion_node.hierarchical_cmds_msg.header.stamp = semantic_fusion_node.tracks_msg.header.stamp

    for idx in semantic_fusion_node.semantic_objects.keys():
        obj = semantic_fusion_node.semantic_objects[idx]
        hierarchical_cmd_msg = HierarchicalCommand()

        hierarchical_cmd_msg.pose.position.x = obj.pos_x
        hierarchical_cmd_msg.pose.position.y = obj.pos_y
        hierarchical_cmd_msg.pose.position.z = obj.pos_z

        hierarchical_cmd_msg.class_string = obj.class_string
        hierarchical_cmd_msg.object_id = obj.track_id

        # Populate message
        for att in obj.attributes:
            kv_msg = KeyValue()
            kv_msg.key = att
            kv_msg.value = obj.attributes[att].var_labels[obj.attributes[att].probs.argmax()]
            hierarchical_cmd_msg.attributes.append(kv_msg)

        for state in obj.states:
            kv_msg = KeyValue()
            kv_msg.key = state
            kv_msg.value = obj.states[state].var_labels[obj.states[state].probs.argmax()]
            hierarchical_cmd_msg.states.append(kv_msg)
        
        hierarchical_cmd_msg.comms = obj.comms.var_labels[obj.comms.probs.argmax()]

        semantic_fusion_node.hierarchical_cmds_msg.commands.append(hierarchical_cmd_msg)

    semantic_fusion_node.hierarchical_cmd_pub.publish(semantic_fusion_node.hierarchical_cmds_msg)