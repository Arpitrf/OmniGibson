import os

import omnigibson as og

from omnigibson.scenes.scene_base import Scene
from omnigibson.tasks.task_base import BaseTask
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.sim_utils import land_object
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)

class DummyTask(BaseTask):
    """
    Dummy task
    """

    def _load(self, env):
        # Do nothing here
        pass

    def _create_termination_conditions(self):
        # Do nothing
        return dict()

    def _create_reward_functions(self):
        # Do nothing
        return dict()

    def _get_obs(self, env):
        # No task-specific obs of any kind
        return dict(), dict()

    def _load_non_low_dim_observation_space(self):
        # No non-low dim observations so we return an empty dict
        return dict()
    
    def save_task(self, path):
        """
        Writes the current scene configuration to a .json file

        Args:
            path (None or str): If specified, absolute fpath to the desired path to write the .json. Default is
                <gm.DATASET_PATH>/scenes/<SCENE_MODEL>/json/...>
            override (bool): Whether to override any files already found at the path to write the task .json
        """
        # Write metadata and then save
        self.write_task_metadata()
        og.sim.save(json_paths=[path])

    @classproperty
    def valid_scene_types(cls):
        # Any scene works
        return {Scene}

    @classproperty
    def default_termination_config(cls):
        # Empty dict
        return {}

    @classproperty
    def default_reward_config(cls):
        # Empty dict
        return {}
