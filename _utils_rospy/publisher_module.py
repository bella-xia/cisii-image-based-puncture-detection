# python lib imports
from collections import namedtuple
import numpy as np

# rospy imports
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64, Float32
from std_msgs.msg import Int32

PubRosTopic = namedtuple(
    "PubRosTopic", ["published_topic", "message_type", "publisher_attr"]
)


class RosTopicPublisher:
    def __init__(self, ros_publisher_list):

        self.ros_publisher_list = ros_publisher_list
        self.init_publishers()

    def init_publishers(self):
        self.publish_typespec = {
            rostopic_instance.publisher_attr: self.define_data_attributes(
                rostopic_instance.message_type
            )
            for rostopic_instance in self.ros_publisher_list
        }

        for ros_topic in self.ros_publisher_list:
            setattr(
                self,
                ros_topic.publisher_attr,
                rospy.Publisher(
                    ros_topic.published_topic, ros_topic.message_type, queue_size=1
                ),
            )

    def define_data_attributes(self, rospy_type):
        if rospy_type == Int32:
            return int
        elif rospy_type in {Float32, Float64}:
            return float
        elif rospy_type == Bool:
            return bool
        elif rospy_type == Image:
            return Image

        print(f"Unknown rospy type: {rospy_type}. Defaulting to any.")
        return any

    def publish_data(self, publisher_attrs, data):
        for publisher_attr, data_instance in zip(publisher_attrs, data):
            publisher_type = self.publish_typespec.get(publisher_attr, None)
            publisher = getattr(self, publisher_attr, None)
            if publisher_attr and publisher_type and publisher:
                try:
                    publisher.publish(data_instance)
                except Exception as e:
                    print(f"Failed to publish data {data_instance}: {e}")
                    pass
            else:
                print(f"Publisher is not defined, skipping publishing.")
