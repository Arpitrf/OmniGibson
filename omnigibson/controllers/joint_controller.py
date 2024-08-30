import math

import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.controllers import (
    ControlType,
    GripperController,
    IsGraspingState,
    LocomotionController,
    ManipulationController,
)
from omnigibson.macros import create_module_macros
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)
m.DEFAULT_JOINT_POS_KP = 50.0
m.DEFAULT_JOINT_POS_DAMPING_RATIO = 1.0  # critically damped
m.DEFAULT_JOINT_VEL_KP = 2.0


class JointController(LocomotionController, ManipulationController, GripperController):
    """
    Controller class for joint control. Because omniverse can handle direct position / velocity / effort
    control signals, this is merely a pass-through operation from command to control (with clipping / scaling built in).

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2a. If using delta commands, then adds the command to the current joint state
        2b. Clips the resulting command by the motor limits
    """

    def __init__(
        self,
        control_freq,
        motor_type,
        control_limits,
        dof_idx,
        command_input_limits="default",
        command_output_limits="default",
        kp=None,
        damping_ratio=None,
        use_impedances=False,
        use_gravity_compensation=False,
        use_cc_compensation=True,
        use_delta_commands=False,
        compute_delta_in_quat_space=None,
    ):
        """
        Args:
            control_freq (int): controller loop frequency
            motor_type (str): type of motor being controlled, one of {position, velocity, effort}
            control_limits (Dict[str, Tuple[Array[float], Array[float]]]): The min/max limits to the outputted
                control signal. Should specify per-dof type limits, i.e.:

                "position": [[min], [max]]
                "velocity": [[min], [max]]
                "effort": [[min], [max]]
                "has_limit": [...bool...]

                Values outside of this range will be clipped, if the corresponding joint index in has_limit is True.
            dof_idx (Array[int]): specific dof indices controlled by this robot. Used for inferring
                controller-relevant values during control computations
            command_input_limits (None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]]):
                if set, is the min/max acceptable inputted command. Values outside this range will be clipped.
                If None, no clipping will be used. If "default", range will be set to (-1, 1)
            command_output_limits (None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]]):
                if set, is the min/max scaled command. If both this value and @command_input_limits is not None,
                then all inputted command values will be scaled from the input range to the output range.
                If either is None, no scaling will be used. If "default", then this range will automatically be set
                to the @control_limits entry corresponding to self.control_type
            kp (None or float): If @motor_type is "position" or "velocity" and @use_impedances=True, this is the
                proportional gain applied to the joint controller. If None, a default value will be used.
            damping_ratio (None or float): If @motor_type is "position" and @use_impedances=True, this is the
                damping ratio applied to the joint controller. If None, a default value will be used.
            use_impedances (bool): If True, will use impedances via the mass matrix to modify the desired efforts
                applied
            use_gravity_compensation (bool): If True, will add gravity compensation to the computed efforts. This is
                an experimental feature that only works on fixed base robots. We do not recommend enabling this.
            use_cc_compensation (bool): If True, will add Coriolis / centrifugal compensation to the computed efforts.
            use_delta_commands (bool): whether inputted commands should be interpreted as delta or absolute values
            compute_delta_in_quat_space (None or List[(rx_idx, ry_idx, rz_idx), ...]): if specified, groups of
                joints that need to be processed in quaternion space to avoid gimbal lock issues normally faced by
                3 DOF rotation joints. Each group needs to consist of three idxes corresponding to the indices in
                the input space. This is only used in the delta_commands mode.
        """
        # Store arguments
        assert_valid_key(key=motor_type.lower(), valid_keys=ControlType.VALID_TYPES_STR, name="motor_type")
        self._motor_type = motor_type.lower()
        self._use_delta_commands = use_delta_commands
        self._compute_delta_in_quat_space = [] if compute_delta_in_quat_space is None else compute_delta_in_quat_space

        # Store control gains
        if self._motor_type == "position":
            kp = m.DEFAULT_JOINT_POS_KP if kp is None else kp
            damping_ratio = m.DEFAULT_JOINT_POS_DAMPING_RATIO if damping_ratio is None else damping_ratio
        elif self._motor_type == "velocity":
            kp = m.DEFAULT_JOINT_VEL_KP if kp is None else kp
            assert damping_ratio is None, "Cannot set damping_ratio for JointController with motor_type=velocity!"
        else:  # effort
            assert kp is None, "Cannot set kp for JointController with motor_type=effort!"
            assert damping_ratio is None, "Cannot set damping_ratio for JointController with motor_type=effort!"
        self.kp = kp
        self.kd = None if damping_ratio is None else 2 * math.sqrt(self.kp) * damping_ratio
        self._use_impedances = use_impedances
        self._use_gravity_compensation = use_gravity_compensation
        self._use_cc_compensation = use_cc_compensation

        # Warn the user about gravity compensation being experimental.
        if self._use_gravity_compensation:
            log.warning(
                "JointController is using gravity compensation. This is an experimental feature that only works on "
                "fixed base robots. We do not recommend enabling this."
            )

        # When in delta mode, it doesn't make sense to infer output range using the joint limits (since that's an
        # absolute range and our values are relative). So reject the default mode option in that case.
        assert not (
            self._use_delta_commands and type(command_output_limits) == str and command_output_limits == "default"
        ), "Cannot use 'default' command output limits in delta commands mode of JointController. Try None instead."

        # Run super init
        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=dof_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
        )

    def _update_goal(self, command, control_dict):
        # Compute the base value for the command
        base_value = control_dict[f"joint_{self._motor_type}"][self.dof_idx]

        # If we're using delta commands, add this value
        if self._use_delta_commands:

            # Apply the command to the base value.
            target = base_value + command

            # Correct any gimbal lock issues using the compute_delta_in_quat_space group.
            for rx_ind, ry_ind, rz_ind in self._compute_delta_in_quat_space:
                # Grab the starting rotations of these joints.
                start_rots = base_value[[rx_ind, ry_ind, rz_ind]]

                # Grab the delta rotations.
                delta_rots = command[[rx_ind, ry_ind, rz_ind]]

                # Compute the final rotations in the quaternion space.
                _, end_quat = T.pose_transform(
                    th.zeros(3), T.euler2quat(delta_rots), th.zeros(3), T.euler2quat(start_rots)
                )
                end_rots = T.quat2euler(end_quat)

                # Update the command
                target[[rx_ind, ry_ind, rz_ind]] = end_rots

        # Otherwise, goal is simply the command itself
        else:
            target = command

        # Clip the command based on the limits
        target = target.clip(
            self._control_limits[ControlType.get_type(self._motor_type)][0][self.dof_idx],
            self._control_limits[ControlType.get_type(self._motor_type)][1][self.dof_idx],
        )

        return dict(target=target)

    def compute_control(self, goal_dict, control_dict):
        """
        Converts the (already preprocessed) inputted @command into deployable (non-clipped!) joint control signal

        Args:
            goal_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                goals necessary for controller computation. Must include the following keys:
                    target: desired N-dof absolute joint values used as setpoint
            control_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                states necessary for controller computation. Must include the following keys:
                    joint_position: Array of current joint positions
                    joint_velocity: Array of current joint velocities
                    joint_effort: Array of current joint effort

        Returns:
            Array[float]: outputted (non-clipped!) control signal to deploy
        """
        base_value = control_dict[f"joint_{self._motor_type}"][self.dof_idx]
        target = goal_dict["target"]

        # Convert control into efforts
        if self._use_impedances:
            if self._motor_type == "position":
                # Run impedance controller -- effort = pos_err * kp + vel_err * kd
                position_error = target - base_value
                vel_pos_error = -control_dict[f"joint_velocity"][self.dof_idx]
                u = position_error * self.kp + vel_pos_error * self.kd
            elif self._motor_type == "velocity":
                # Compute command torques via PI velocity controller plus gravity compensation torques
                velocity_error = target - base_value
                u = velocity_error * self.kp
            else:  # effort
                u = target

            dof_idxs_mat = th.meshgrid(self.dof_idx, self.dof_idx, indexing="xy")
            mm = control_dict["mass_matrix"][dof_idxs_mat]
            u = mm @ u

            # Add gravity compensation
            if self._use_gravity_compensation:
                u += control_dict["gravity_force"][self.dof_idx]

            # Add Coriolis / centrifugal compensation
            if self._use_cc_compensation:
                u += control_dict["cc_force"][self.dof_idx]

        else:
            # Desired is the exact goal
            u = target

        # Return control
        return u

    def compute_no_op_goal(self, control_dict):
        # Compute based on mode
        if self._motor_type == "position":
            # Maintain current qpos
            target = control_dict[f"joint_{self._motor_type}"][self.dof_idx]
        else:
            # For velocity / effort, directly set to 0
            target = th.zeros(self.control_dim)

        return dict(target=target)

    def _get_goal_shapes(self):
        return dict(target=(self.control_dim,))

    def is_grasping(self):
        # No good heuristic to determine grasping, so return UNKNOWN
        return IsGraspingState.UNKNOWN

    @property
    def use_delta_commands(self):
        """
        Returns:
            bool: Whether this controller is using delta commands or not
        """
        return self._use_delta_commands

    @property
    def motor_type(self):
        """
        Returns:
            str: The type of motor being simulated by this controller. One of {"position", "velocity", "effort"}
        """
        return self._motor_type

    @property
    def control_type(self):
        return ControlType.EFFORT if self._use_impedances else ControlType.get_type(type_str=self._motor_type)

    @property
    def command_dim(self):
        return len(self.dof_idx)
