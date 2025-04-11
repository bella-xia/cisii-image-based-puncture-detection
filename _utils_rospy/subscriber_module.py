# python lib imports
from collections import namedtuple

# rospy imports
import rospy
from sensor_msgs.msg import Image

SubRosTopic = namedtuple(
    "SubRosTopic",
    ["subscribed_topic", "message_type", "subscriber_attr", "assign_attr"],
)


class RosTopicSubscriber:
    def __init__(self, ros_subcription_list):

        self.ros_subcription_list = ros_subcription_list
        self.init_subscribers()

    def init_subscribers(self):
        self.data_attributes = set(
            [
                rostopic_instance.assign_attr
                for rostopic_instance in self.ros_subcription_list
            ]
        )
        self.sub_attributes = set(
            [
                rospic_instance.subscriber_attr
                for rospic_instance in self.ros_subcription_list
            ]
        )

        for ros_topic in self.ros_subcription_list:
            setattr(
                self,
                ros_topic.subscriber_attr,
                rospy.Subscriber(
                    ros_topic.subscribed_topic,
                    ros_topic.message_type,
                    self.custom_callback(ros_topic.message_type, ros_topic.assign_attr),
                ),
            )

    def custom_callback(self, message_type, attr_name):
        return lambda data: (
            setattr(self, attr_name, data)
            if message_type == Image
            else setattr(self, attr_name, data.data)
        )

    def get_data(self, attr_name):
        if attr_name not in self.data_attributes:
            return None
        return getattr(self, attr_name, None)

    # feel like this should never be used, TODO later
    def get_subscriber(self, attr_name):
        if attr_name not in self.sub_attributes:
            return None
        return getattr(self, attr_name, None)
