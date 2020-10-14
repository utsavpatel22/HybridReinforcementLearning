// Useful tutorial for writing sub and pub with class https://github.com/wsnewman/ros_class/blob/master/example_ros_class/src/example_ros_class.h

#include <ros/ros.h>
#include <vector>
#include <gazebo_msgs/ModelStates.h>
#include<nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
// #include "tf/LinearMath/Transform.h"
#include <tf/transform_listener.h>

class Goal_Pub {

private:
  ros::NodeHandle nh;
  ros::Publisher pub;
  ros::Subscriber sub;
  ros::Subscriber location;

public:
  // Data members
  float goal_x;
  float goal_y;
  float waypoint[4];
  int count;


  // Member functions
  // Constructor
  Goal_Pub(){
    init_sub();
    init_pub();
    float y[5] = {3,6,9,12.5};

    goal_x = 12.5;
    goal_y = 0.0;
    waypoint[0] = 3.0;
    waypoint[1] = 6.0;
    waypoint[2]  =9.0;
    waypoint[3] = 12.5;
    count = 0;
    ROS_INFO("Inside Contructor");
  }

  void init_sub() {
    ROS_INFO("Initializing Subscribers");
    sub = nh.subscribe("/turtlebot0/ground_truth/state", 100, &Goal_Pub::stateCb, this);
    }

  void init_pub() {
    ROS_INFO("Initializing Publishers");
    pub = nh.advertise<geometry_msgs::Twist>("/jackal/target/position", 100, true);

  }

  void stateCb(const nav_msgs::Odometry& msg) {

	geometry_msgs::Twist rel_goal;
  
      // std::cout<< "Jackal Position = " << msg.pose[i].position.x << "   " << msg.pose[i].position.y << std::endl;
      rel_goal.linear.x = goal_x - msg.pose.pose.position.x;
      rel_goal.linear.y = goal_y - msg.pose.pose.position.y;

      if((goal_x - msg.pose.pose.position.x)*(goal_x - msg.pose.pose.position.x) + (goal_y - (msg.pose.pose.position.y))*(goal_y - (msg.pose.pose.position.y)) < .25){
        goal_x = waypoint[count];
        count++;
      }

      tf::Pose pose;
      tf::poseMsgToTF(msg.pose.pose, pose);
      double yaw = tf::getYaw(pose.getRotation());
      // std::cout<<" Yaw = " << yaw * 57.296 << std::endl;
      // yaw = yaw * 57.296;
      rel_goal.linear.z = yaw;
      pub.publish(rel_goal);
      std::cout<<"Publishing "<<std::endl;
      
  }
};

int main(int argc, char **argv){
  ros::init(argc, argv,"Rel_Goal_Pub");
  Goal_Pub obj;
  ros::spin();
  return 0;
}
