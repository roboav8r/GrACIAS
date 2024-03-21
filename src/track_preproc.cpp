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


      this->trks_msg_ = tracking_msgs::msg::Tracks3D();
      this->persons_msg_ = tracking_msgs::msg::Tracks3D();

    }

  private:
    void topic_callback(const tracking_msgs::msg::Tracks3D::SharedPtr msg)
    {

      this->persons_msg_ = tracking_msgs::msg::Tracks3D();
      this->persons_msg_.header = msg->header;     

      for (auto it = msg->tracks.begin(); it != msg->tracks.end(); it++)
      {
            if ( it->class_string != "person"){
              continue;
            } else {
              this->persons_msg_.tracks.emplace_back(*it);
            }
      }

      this->publisher_->publish(this->persons_msg_);
           
    }

    rclcpp::Subscription<tracking_msgs::msg::Tracks3D>::SharedPtr subscription_;
    rclcpp::Publisher<tracking_msgs::msg::Tracks3D>::SharedPtr publisher_;
    tracking_msgs::msg::Tracks3D trks_msg_;
    tracking_msgs::msg::Tracks3D persons_msg_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrackPreProc>());
  rclcpp::shutdown();
  return 0;
}