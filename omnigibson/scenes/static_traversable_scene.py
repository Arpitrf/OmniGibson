import os

import numpy as np

import omnigibson as og
from omnigibson.prims.geom_prim import CollisionVisualGeomPrim
from omnigibson.scenes.traversable_scene import TraversableScene
from omnigibson.utils.asset_utils import get_scene_path
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import add_asset_to_stage

# Create module logger
log = create_module_logger(module_name=__name__)


class StaticTraversableScene(TraversableScene):
    """
    Static traversable scene class for OmniGibson, where scene is defined by a singular mesh (no intereactable objects)
    """

    def __init__(
        self,
        scene_model,
        scene_file=None,
        trav_map_resolution=0.1,
        default_erosion_radius=0.0,
        trav_map_with_objects=True,
        num_waypoints=10,
        waypoint_resolution=0.2,
    ):
        """
        Args:
            scene_model (str): Scene model name, e.g.: Adrian
            scene_file (None or str): If specified, full path of JSON file to load (with .json).
                None results in no additional objects being loaded into the scene
            trav_map_resolution (float): traversability map resolution
            default_erosion_radius (float): default map erosion radius in meters
            trav_map_with_objects (bool): whether to use objects or not when constructing graph
            num_waypoints (int): number of way points returned
            waypoint_resolution (float): resolution of adjacent way points
        """
        # Store and initialize additional variables
        self._floor_heights = None
        self._scene_mesh = None

        # Run super init
        assert og.sim.floor_plane, "Floor plane must be enabled for StaticTraversableScene"
        super().__init__(
            scene_model=scene_model,
            scene_file=scene_file,
            trav_map_resolution=trav_map_resolution,
            default_erosion_radius=default_erosion_radius,
            trav_map_with_objects=trav_map_with_objects,
            num_waypoints=num_waypoints,
            waypoint_resolution=waypoint_resolution,
        )

    def _load(self):
        # Run super first
        super()._load()

        # Load the scene mesh (use downsampled one if available)
        filename = os.path.join(get_scene_path(self.scene_model), "mesh_z_up_downsampled.obj")
        if not os.path.isfile(filename):
            filename = os.path.join(get_scene_path(self.scene_model), "mesh_z_up.obj")

        scene_prim = add_asset_to_stage(
            asset_path=filename,
            prim_path=f"/World/scene_{self.scene_model}",
        )

        # Grab the actual mesh prim
        self._scene_mesh = CollisionVisualGeomPrim(
            prim_path=f"/World/scene_{self.scene_model}/mesh_z_up/{self.scene_model}_mesh_texture",
            name=f"{self.scene_model}_mesh",
        )

        # Load floor metadata
        floor_height_path = os.path.join(get_scene_path(self.scene_model), "floors.txt")
        assert os.path.isfile(floor_height_path), f"floor_heights.txt cannot be found in model: {self.scene_model}"
        with open(floor_height_path, "r") as f:
            self.floor_heights = sorted(list(map(float, f.readlines())))
            log.debug("Floors {}".format(self.floor_heights))

        # Move the floor plane to the first floor by default
        self.move_floor_plane(floor=0)

        # Filter the collision between the scene mesh and the floor plane
        self._scene_mesh.add_filtered_collision_pair(prim=og.sim.floor_plane)

        # Load the traversability map
        self._trav_map.load_map(get_scene_path(self.scene_model))

    def move_floor_plane(self, floor=0, additional_elevation=0.02, height=None):
        """
        Resets the floor plane to a new floor

        Args:
            floor (int): Integer identifying the floor to move the floor plane to
            additional_elevation (float): Additional elevation with respect to the height of the floor
            height (None or float): If specified, alternative parameter to directly control the height of the ground
                plane. Note that this will override @additional_elevation and @floor!
        """
        height = height if height is not None else self.floor_heights[floor] + additional_elevation
        # TODO(parallel): Have the simulator manage the position of this & make sure there are no conflicting requests.
        og.sim.floor_plane.set_position(np.array([0, 0, height]))

    def get_floor_height(self, floor=0):
        """
        Return the current floor height (in meter)

        Returns:
            int: current floor height
        """
        return self.floor_heights[floor]

    @property
    def n_floors(self):
        return len(self._floor_heights)
