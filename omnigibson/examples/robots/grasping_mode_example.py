"""
Example script demo'ing robot manipulation control with grasping.
"""
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.sensors import VisionSensor
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController

def choose_controllers(robot, random_selection=False):
    """
    For a given robot, iterates over all components of the robot, and returns the requested controller type for each
    component.

    :param robot: BaseRobot, robot class from which to infer relevant valid controller options
    :param random_selection: bool, if the selection is random (for automatic demo execution). Default False

    :return dict: Mapping from individual robot component (e.g.: base, arm, etc.) to selected controller names
    """
    # Create new dict to store responses from user
    controller_choices = dict()

    # Grab the default controller config so we have the registry of all possible controller options
    default_config = robot._default_controller_config

    # Iterate over all components in robot
    for component, controller_options in default_config.items():
        # Select controller
        options = list(sorted(controller_options.keys()))
        choice = choose_from_options(
            options=options, name="{} controller".format(component), random_selection=random_selection
        )

        # Add to user responses
        controller_choices[component] = choice

    return controller_choices

CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

GRASPING_MODES = dict(
    sticky="Sticky Mitten - Objects are magnetized when they touch the fingers and a CLOSE command is given",
    assisted="Assisted Grasping - Objects are magnetized when they touch the fingers, are within the hand, and a CLOSE command is given",
    physical="Physical Grasping - No additional grasping assistance applied",
)

# Don't use GPU dynamics and Use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Robot grasping mode demo with selection
    Queries the user to select a type of grasping mode
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose type of grasping
    grasping_mode = choose_from_options(options=GRASPING_MODES, name="grasping mode", random_selection=random_selection)

    # Create environment configuration to use
    scene_cfg = dict(type="Scene")
    robot0_cfg = dict(
        type="Tiago",
        obs_modalities=["rgb"],     # we're just doing a grasping demo so we don't need all observation modalities
        action_type="continuous",
        action_normalize=True,
        grasping_mode=grasping_mode,
    )

    # Define objects to load
    table_cfg = dict(
        type="DatasetObject",
        name="table",
        category="breakfast_table",
        model="lcsizg",
        bounding_box=[0.5, 0.5, 0.8],
        fixed_base=True,
        position=[0.7, -0.1, 0.6],
        orientation=[0, 0, 0.707, 0.707],
    )

    chair_cfg = dict(
        type="DatasetObject",
        name="chair",
        category="straight_chair",
        model="amgwaw",
        bounding_box=None,
        fixed_base=False,
        position=[0.45, 0.65, 0.425],
        orientation=[0, 0, -0.9990215, -0.0442276],
    )

    box_cfg = dict(
        type="PrimitiveObject",
        name="box",
        primitive_type="Cube",
        rgba=[1.0, 0, 0, 1.0],
        size=0.05,
        position=[0.53, -0.1, 0.97],
    )

    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg], objects=[table_cfg, chair_cfg, box_cfg])

    # Create the environment
    env = og.Environment(configs=cfg)

    #  # Choose robot controller to use
    # robot = env.robots[0]
    # controller_choices = choose_controllers(robot=robot, random_selection=random_selection)

    # print("controller_choices: ", controller_choices)

    # # Choose control mode
    # if random_selection:
    #     control_mode = "random"
    # else:
    #     control_mode = choose_from_options(options=CONTROL_MODES, name="control mode")

    # print("control_mode: ", control_mode)

    # # Update the control mode of the robot
    # controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    # print("controller_config: ", controller_config)
    # robot.reload_controllers(controller_config=controller_config)

    # # Because the controllers have been updated, we need to update the initial state so the correct controller state
    # # is preserved
    # env.scene.update_initial_state()

    # Reset the robot
    robot = env.robots[0]
    robot.set_position([0, 0, 0])
    robot.reset()
    robot.keep_still()

    # Make the robot's camera(s) high-res
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.image_height = 720
            sensor.image_width = 720

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-2.39951,  2.26469,  2.66227]),
        orientation=np.array([-0.23898481,  0.48475231,  0.75464013, -0.37204802]),
    )

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    # Print out relevant keyboard info if using keyboard teleop
    action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo with grasping mode {}.".format(grasping_mode))
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0
    while step != max_steps:
        action, _ = action_generator.get_random_action() if random_selection else action_generator.get_teleop_action()
        for _ in range(10):
            env.step(action)
            step += 1

    # Always shut down the environment cleanly at the end
    env.close()


if __name__ == "__main__":
    main()
