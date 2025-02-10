import json
import os
import time
from functools import cached_property
from typing import List

import cv2
import numpy as np
import open3d as o3d
import openai
import rclpy
import supervision as sv
import tf2_ros
import torch
import torchvision
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
# from groundingdino.util.inference import Model
from openai.types.beta import Assistant
from openai.types.beta.threads import RequiredActionFunctionToolCall
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation
# from segment_anything import SamPredictor, sam_model_registry
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Int64, String
from sensor_msgs.msg import JointState

from pymoveit2 import GripperInterface, MoveIt2

# from .point_cloud_conversion import point_cloud_to_msg


class TabletopHandyBotNode(Node):
    """Main ROS 2 node for Tabletop HandyBot."""

    # TODO: make args rosparams
    def __init__(
            self,
            annotate: bool = False,
            publish_point_cloud: bool = False,
            assistant_id: str = "",
            # Adjust these offsets to your needs:
            offset_x: float = 0.015,
            offset_y: float = -0.015,
            offset_z: float = 0.08,  # accounts for the height of the gripper
    ):
        super().__init__("simple_bot_node")

        self.logger = self.get_logger()

        # self.cv_bridge = CvBridge()
        self.gripper_joint_name = "gripper_joint"
        callback_group = ReentrantCallbackGroup()
        # Create MoveIt 2 interface
        self.arm_joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
        ]
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.arm_joint_names,
            base_link_name="panda_link0",
            end_effector_name="panda_link7",
            group_name="panda_arm",
            callback_group=callback_group,
        )
        self.moveit2.planner_id = "RRTConnectkConfigDefault"
        self.gripper_interface = GripperInterface(
            node=self,
            gripper_joint_names=["gripper_jaw1_joint"],
            open_gripper_joint_positions=[-0.012],
            closed_gripper_joint_positions=[0.0],
            gripper_group_name="ar_gripper",
            callback_group=callback_group,
            gripper_command_action_name="/gripper_controller/gripper_cmd",
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.arm_joint_state: JointState | None = None
        self._last_detections: sv.Detections | None = None
        self._object_in_gripper: bool = False
        self.gripper_squeeze_factor = 0.5
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_z = offset_z

        self.joint_states_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_states_callback, 10)

        self.release_at_sub = self.create_subscription(Int64, "/release_above",
                                                       self.release_above_cb,
                                                       10)
        self.pick_object_sub = self.create_subscription(
            Int64, "/pick_object", self.pick_object_cb, 10)

        self.logger.info("Tabletop HandyBot node initialized.")


        self.go_home()
        self.logger.info("Task completed.")

    def go_home(self):
        joint_positions = [0., 0., 0., 0., 0., 0.]
        self.logger.info("go_home: Joint_pose set")
        self.moveit2.move_to_configuration(joint_positions,
                                           self.arm_joint_names,
                                           tolerance=0.005)
        self.logger.info("go_home: move_to_configuration sent")
        self.moveit2.wait_until_executed()

    def make_pose_msg(self):
        "Copy the format of pick object, make sime pose to move to to test system"
        gripper_rotation = 50
        gripper_opening = 10
        grasp_pose = Pose()
        grasp_pose.position.x = 0.3
        grasp_pose.position.y = 0.3
        grasp_pose.position.z = 0.3
        top_down_rot = Rotation.from_quat([0, 1, 0, 0])
        extra_rot = Rotation.from_euler("z", gripper_rotation, degrees=True)
        grasp_quat = (extra_rot * top_down_rot).as_quat()
        grasp_pose.orientation.x = grasp_quat[0]
        grasp_pose.orientation.y = grasp_quat[1]
        grasp_pose.orientation.z = grasp_quat[2]
        grasp_pose.orientation.w = grasp_quat[3]
        self.grasp_at(grasp_pose, gripper_opening)

    def grasp_at(self, msg: Pose, gripper_opening: float):
        self.logger.info(f"Grasp at: {msg} with opening: {gripper_opening}")

        self.gripper_interface.open()
        self.gripper_interface.wait_until_executed()

        # move 5cm above the item first
        msg.position.z += 0.05
        self.move_to(msg)
        time.sleep(0.05)

    def move_to(self, msg: Pose):
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "base_link"
        pose_goal.pose = msg

        self.moveit2.move_to_pose(pose=pose_goal)
        self.moveit2.wait_until_executed()

    def pick_object_old(self, object_index: int, detections: sv.Detections,
                    depth_image: np.ndarray):
        """Perform a top-down grasp on the object."""
        # mask out the depth image except for the detected objects
        masked_depth_image = np.zeros_like(depth_image, dtype=np.float32)
        mask = detections.mask[object_index]
        masked_depth_image[mask] = depth_image[mask]
        masked_depth_image /= 1000.0

        # convert the masked depth image to a point cloud
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(masked_depth_image),
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
        )
        pcd.transform(self.cam_to_base_affine)
        points = np.asarray(pcd.points)
        grasp_z = points[:, 2].max()

        near_grasp_z_points = points[points[:, 2] > grasp_z - 0.008]
        xy_points = near_grasp_z_points[:, :2]
        xy_points = xy_points.astype(np.float32)
        center, dimensions, theta = cv2.minAreaRect(xy_points)

        gripper_rotation = theta
        if dimensions[0] > dimensions[1]:
            gripper_rotation -= 90
        if gripper_rotation < -90:
            gripper_rotation += 180
        elif gripper_rotation > 90:
            gripper_rotation -= 180

        gripper_opening = min(dimensions)
        grasp_pose = Pose()
        grasp_pose.position.x = center[0] + self.offset_x
        grasp_pose.position.y = center[1] + self.offset_y
        grasp_pose.position.z = grasp_z + self.offset_z
        top_down_rot = Rotation.from_quat([0, 1, 0, 0])
        extra_rot = Rotation.from_euler("z", gripper_rotation, degrees=True)
        grasp_quat = (extra_rot * top_down_rot).as_quat()
        grasp_pose.orientation.x = grasp_quat[0]
        grasp_pose.orientation.y = grasp_quat[1]
        grasp_pose.orientation.z = grasp_quat[2]
        grasp_pose.orientation.w = grasp_quat[3]
        self.grasp_at(grasp_pose, gripper_opening)

    # def grasp_at(self, msg: Pose, gripper_opening: float):
    #     self.logger.info(f"Grasp at: {msg} with opening: {gripper_opening}")

    #     self.gripper_interface.open()
    #     self.gripper_interface.wait_until_executed()

    #     # move 5cm above the item first
    #     msg.position.z += 0.05
    #     self.move_to(msg)
    #     time.sleep(0.05)

    #     # grasp the item
    #     msg.position.z -= 0.05
    #     self.move_to(msg)
    #     time.sleep(0.05)

    #     gripper_pos = -gripper_opening / 2. * self.gripper_squeeze_factor
    #     gripper_pos = min(gripper_pos, 0.0)
    #     self.gripper_interface.move_to_position(gripper_pos)
    #     self.gripper_interface.wait_until_executed()

    #     # lift the item
    #     msg.position.z += 0.12
    #     self.move_to(msg)
    #     time.sleep(0.05)

    def release_above(self, object_index: int, detections: sv.Detections,
                      depth_image: np.ndarray):
        """Move the robot arm above the object and release the gripper."""
        masked_depth_image = np.zeros_like(depth_image, dtype=np.float32)
        mask = detections.mask[object_index]
        masked_depth_image[mask] = depth_image[mask]
        masked_depth_image /= 1000.0

        # convert the masked depth image to a point cloud
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(masked_depth_image),
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
        )
        pcd.transform(self.cam_to_base_affine)

        points = np.asarray(pcd.points).astype(np.float32)
        # release 5cm above the object
        drop_z = np.percentile(points[:, 2], 95) + 0.05
        median_z = np.median(points[:, 2])

        xy_points = points[points[:, 2] > median_z, :2]
        xy_points = xy_points.astype(np.float32)
        center, _, _ = cv2.minAreaRect(xy_points)

        drop_pose = Pose()
        drop_pose.position.x = center[0] + self.offset_x
        drop_pose.position.y = center[1] + self.offset_y
        drop_pose.position.z = drop_z + self.offset_z
        # Straight down pose
        drop_pose.orientation.x = 0.0
        drop_pose.orientation.y = 1.0
        drop_pose.orientation.z = 0.0
        drop_pose.orientation.w = 0.0

        self.release_at(drop_pose)

    def release_gripper(self):
        self.gripper_interface.open()
        self.gripper_interface.wait_until_executed()

    def flick_wrist_while_release(self):
        joint_positions = self.arm_joint_state.position
        joint_positions[4] -= np.deg2rad(25)
        self.moveit2.move_to_configuration(joint_positions,
                                           self.arm_joint_names,
                                           tolerance=0.005)
        time.sleep(3)

        self.gripper_interface.open()
        self.gripper_interface.wait_until_executed()
        self.moveit2.wait_until_executed()


    @cached_property
    def cam_to_base_affine(self):
        cam_to_base_link_tf = self.tf_buffer.lookup_transform(
            target_frame="panda_link0",
            source_frame="camera_color_frame", # maybe change this to /camera_depth_optical_frame
            time=Time(),
            timeout=Duration(seconds=5))
        cam_to_base_rot = Rotation.from_quat([
            cam_to_base_link_tf.transform.rotation.x,
            cam_to_base_link_tf.transform.rotation.y,
            cam_to_base_link_tf.transform.rotation.z,
            cam_to_base_link_tf.transform.rotation.w,
        ])
        cam_to_base_pos = np.array([
            cam_to_base_link_tf.transform.translation.x,
            cam_to_base_link_tf.transform.translation.y,
            cam_to_base_link_tf.transform.translation.z,
        ])
        affine = np.eye(4)
        affine[:3, :3] = cam_to_base_rot.as_matrix()
        affine[:3, 3] = cam_to_base_pos
        return affine

    def release_at(self, msg: Pose):
        # NOTE: straight down is wxyz 0, 0, 1, 0
        # good pose is 0, -0.3, 0.35
        self.logger.info(f"Releasing at: {msg}")
        self.move_to(msg)

        self.gripper_interface.open()
        self.gripper_interface.wait_until_executed()

    def pick_object_cb(self, msg: Int64):
        if self._last_detections is None or self._last_depth_msg is None:
            self.logger.warning("No detections or depth image available.")
            return

        depth_image = self.cv_bridge.imgmsg_to_cv2(self._last_depth_msg)
        self.pick_object(msg.data, self._last_detections, depth_image)

    def release_above_cb(self, msg: Int64):
        if self._last_detections is None or self._last_depth_msg is None:
            self.logger.warning("No detections or depth image available.")
            return

        depth_image = self.cv_bridge.imgmsg_to_cv2(self._last_depth_msg)
        self.release_above(msg.data, self._last_detections, depth_image)

    def joint_states_callback(self, msg: JointState):
        joint_state = JointState()
        joint_state.header = msg.header
        for name in self.arm_joint_names:
            for i, joint_state_joint_name in enumerate(msg.name):
                if name == joint_state_joint_name:
                    joint_state.name.append(name)
                    joint_state.position.append(msg.position[i])
                    joint_state.velocity.append(msg.velocity[i])
                    joint_state.effort.append(msg.effort[i])

        self.arm_joint_state = joint_state
        # self.logger.info(f"joint_state in: {msg}")
        # self.logger.info(f"joint_state out: {self.arm_joint_state}")



def main():
    rclpy.init()
    node = TabletopHandyBotNode()
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == "__main__":
    main()
