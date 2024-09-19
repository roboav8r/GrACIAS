#!/usr/bin/env python3

from foxglove_msgs.msg import SceneUpdate, SceneEntity, TextPrimitive

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
        entity_msg.lifetime.nanosec = 500000000

        text = TextPrimitive()
        text.billboard = True
        text.font_size = 16.
        text.scale_invariant = True
        text.color.a = 1.0
        text.pose.position.x = obj.pos_x
        text.pose.position.y = obj.pos_y
        text.pose.position.z = obj.pos_z
        # text.text = "%s #%s: \nID: %s \nAuth: %s\n" % (obj.class_string, obj.track_id, obj.identity, obj.auth)
        text.text = "%s #%s: \n" % (obj.class_string, obj.track_id)
        
        text.text += '\n'
        for att in obj.attributes:
            text.text += "%s: %s %2.0f%%\n" % (att, obj.attributes[att].var_labels[obj.attributes[att].probs.argmax()], 100*obj.attributes[att].probs(obj.attributes[att].probs.argmax()))

        text.text += '\n'
        for state in obj.states:
            text.text += "%s: %s %2.0f%%\n" % (state, obj.states[state].var_labels[obj.states[state].probs.argmax()], 100*obj.states[state].probs(obj.states[state].probs.argmax()))

        text.text += '\n'
        text.text += "command: %s %2.0f%%\n" % (obj.comm_labels[obj.comms.probs.argmax()], 100*obj.comms.probs(obj.comms.probs.argmax()))
        entity_msg.texts.append(text)

        semantic_fusion_node.scene_out_msg.entities.append(entity_msg)

    semantic_fusion_node.semantic_scene_pub.publish(semantic_fusion_node.scene_out_msg)