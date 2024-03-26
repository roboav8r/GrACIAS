#include "rclcpp/rclcpp.hpp"
#include "ar_track_alvar_msgs/msg/alvar_markers.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "foxglove_msgs/msg/scene_update.hpp"
#include "foxglove_msgs/msg/scene_entity.hpp"
#include "gracias_interfaces/msg/auth.hpp"
#include "gracias_interfaces/msg/comm.hpp"
#include "gracias_interfaces/msg/identity.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

class ArPreprocNode : public rclcpp::Node
{
public:
    ArPreprocNode()
    : Node("ar_preproc_node")
    {
        // Subscribe to ar_pose_marker topic
        subscription_ar_pose_marker_ = this->create_subscription<ar_track_alvar_msgs::msg::AlvarMarkers>(
            "ar_pose_marker", 10,
            std::bind(&ArPreprocNode::arPoseMarkerCallback, this, std::placeholders::_1));
        
        // Subscribe to visualization_marker topic
        subscription_visualization_marker_ = this->create_subscription<visualization_msgs::msg::Marker>(
            "visualization_marker", 10,
            std::bind(&ArPreprocNode::visualizationMarkerCallback, this, std::placeholders::_1));

        this->declare_parameter("tracker_frame",rclcpp::ParameterType::PARAMETER_STRING);
        this->tracker_frame_ = this->get_parameter("tracker_frame").as_string();
        this->declare_parameter("words",rclcpp::ParameterType::PARAMETER_STRING_ARRAY);
        this->words_ = this->get_parameter("words").as_string_array();
        this->declare_parameter("types",rclcpp::ParameterType::PARAMETER_STRING_ARRAY);
        this->types_ = this->get_parameter("types").as_string_array();

        scene_pub_ = this->create_publisher<foxglove_msgs::msg::SceneUpdate>("ar_scene", 10);
        auth_pub_ = this->create_publisher<gracias_interfaces::msg::Auth>("authentication", 10);
        comm_pub_ = this->create_publisher<gracias_interfaces::msg::Comm>("communication", 10);
        id_pub_ = this->create_publisher<gracias_interfaces::msg::Identity>("identification", 10);

        this->tf_buffer_ =std::make_unique<tf2_ros::Buffer>(this->get_clock());
        this->tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    }

private:
    // Callback for ar_pose_marker topic
    void arPoseMarkerCallback(const ar_track_alvar_msgs::msg::AlvarMarkers::SharedPtr msg)
    {
        // RCLCPP_INFO(this->get_logger(), "Received AR pose marker");
        // RCLCPP_INFO(this->get_logger(), std::to_string(msg->markers.size()).c_str());

        for (auto it = msg->markers.begin(); it != msg->markers.end(); it++)
        {
            // RCLCPP_INFO(this->get_logger(), std::to_string(it->id).c_str());
            std::string type = this->types_[it->id - 1];
            // RCLCPP_INFO(this->get_logger(), type.c_str());
            std::string word = this->words_[it->id - 1];
            // RCLCPP_INFO(this->get_logger(), word.c_str());

            // transform ar tag pose into tracker frame
            this->ar_pose_det_frame_ = geometry_msgs::msg::PoseStamped();
            this->ar_pose_det_frame_.header = it->header;
            // this->ar_pose_det_frame_.header.stamp = rclcpp::Time(0);
            this->ar_pose_det_frame_.pose = it->pose.pose;
            this->ar_pose_trk_frame_ = this->tf_buffer_->transform(this->ar_pose_det_frame_,this->tracker_frame_);
            // this->det_msg_.pose = ar_pose_trk_frame_.pose;

            if (type=="Auth"){
                gracias_interfaces::msg::Auth msg;
                msg.pose.header = it->header;
                msg.pose.header.frame_id = this->tracker_frame_;
                msg.pose.pose = ar_pose_trk_frame_.pose;
                msg.authenticated = true;
                auth_pub_->publish(msg);
                
            } else if (type=="Comm") {
                gracias_interfaces::msg::Comm msg;
                msg.pose.header = it->header;
                msg.pose.header.frame_id = this->tracker_frame_;
                msg.pose.pose = ar_pose_trk_frame_.pose;
                msg.comm = word;
                comm_pub_->publish(msg);

            } else if (type=="Identity") {
                gracias_interfaces::msg::Identity msg;
                msg.pose.header = it->header;
                msg.pose.header.frame_id = this->tracker_frame_;
                msg.pose.pose = ar_pose_trk_frame_.pose;
                msg.identity = word;
                id_pub_->publish(msg);
            }

        }
    }
    
    // Callback for visualization_marker topic
    void visualizationMarkerCallback(const visualization_msgs::msg::Marker::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received Visualization marker");
        // Implement your handling logic here
    }
    
    rclcpp::Subscription<ar_track_alvar_msgs::msg::AlvarMarkers>::SharedPtr subscription_ar_pose_marker_;
    rclcpp::Subscription<visualization_msgs::msg::Marker>::SharedPtr subscription_visualization_marker_;
    
    rclcpp::Publisher<foxglove_msgs::msg::SceneUpdate>::SharedPtr scene_pub_;
    rclcpp::Publisher<gracias_interfaces::msg::Auth>::SharedPtr auth_pub_;
    rclcpp::Publisher<gracias_interfaces::msg::Comm>::SharedPtr comm_pub_;
    rclcpp::Publisher<gracias_interfaces::msg::Identity>::SharedPtr id_pub_;

    std::vector<std::string> types_;
    std::vector<std::string> words_;

    std::string tracker_frame_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

    geometry_msgs::msg::PoseStamped ar_pose_det_frame_;
    geometry_msgs::msg::PoseStamped ar_pose_trk_frame_;
    
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArPreprocNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}