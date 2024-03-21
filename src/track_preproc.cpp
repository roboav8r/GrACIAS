#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "tracking_msgs/msg/tracks3_d.hpp"
#include "tracking_msgs/msg/track3_d.hpp"

using std::placeholders::_1;

class TrackPreProc : public rclcpp::Node
{
  public:
    TrackPreProc()
    : Node("track_preproc_node")
    {
      this->subscription_ = this->create_subscription<tracking_msgs::msg::Tracks3D>(
      "tracks", 10, std::bind(&TrackPreProc::topic_callback, this, _1));
      this->publisher_ = this->create_publisher<tracking_msgs::msg::Tracks3D>("tracked_persons", 10);


      this->det_msg_ = tracking_msgs::msg::Track3D();
      this->dets_msg_ = tracking_msgs::msg::Tracks3D();

    }

  private:
    void topic_callback(const tracking_msgs::msg::Tracks3D::SharedPtr msg)
    {
      rclcpp::Time time_det_rcvd = this->get_clock()->now();
      diagnostic_msgs::msg::KeyValue kv;     

      this->dets_msg_ = tracking_msgs::msg::Detections3D();
      this->dets_msg_.header.stamp = msg->header.stamp;     
      this->dets_msg_.header.frame_id = this->tracker_frame_;

      // Add metadata for later analysis
      kv.key = "time_det_rcvd";
      kv.value = std::to_string(time_det_rcvd.nanoseconds());
      this->dets_msg_.metadata.emplace_back(kv);
      kv.key = "num_dets_rcvd";
      kv.value = std::to_string(msg->detections.size());
      this->dets_msg_.metadata.emplace_back(kv);

      for (auto it = msg->detections.begin(); it != msg->detections.end(); it++)
      {
            this->det_msg_ = tracking_msgs::msg::Detection3D();

            // Convert spatial information
            this->obj_pose_det_frame_ = geometry_msgs::msg::PoseStamped();
            this->obj_pose_det_frame_.header = msg->header;
            this->obj_pose_det_frame_.header.stamp = rclcpp::Time(0);
            this->obj_pose_det_frame_.pose = it->results[0].pose.pose;
            this->obj_pose_trk_frame_ = this->tf_buffer_->transform(this->obj_pose_det_frame_,this->tracker_frame_);
            this->det_msg_.pose = obj_pose_trk_frame_.pose;
            this->det_msg_.bbox.center = obj_pose_trk_frame_.pose;
            this->det_msg_.bbox.size.x = 0;
            this->det_msg_.bbox.size.y = 0;
            this->det_msg_.bbox.size.z = 0;

            // Add semantic information
            this->det_msg_.class_string = this->labels_[std::stoi(it->results[0].hypothesis.class_id)];
            this->det_msg_.class_confidence = it->results[0].hypothesis.score;

            this->dets_msg_.detections.emplace_back(this->det_msg_);
      }

      this->publisher_->publish(this->dets_msg_);
           
    }

    std::string tracker_frame_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    geometry_msgs::msg::PoseStamped obj_pose_det_frame_;
    geometry_msgs::msg::PoseStamped obj_pose_trk_frame_;

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr subscription_;
    rclcpp::Publisher<tracking_msgs::msg::Detections3D>::SharedPtr publisher_;
    tracking_msgs::msg::Detections3D dets_msg_;
    tracking_msgs::msg::Detection3D det_msg_;
    int max_dets_{250}; 

    std::vector<std::string> labels_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DepthAIPreProc>());
  rclcpp::shutdown();
  return 0;
}