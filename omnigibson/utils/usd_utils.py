import math
from collections.abc import Iterable
import os

import omnigibson.lazy as lazy

import numpy as np
import trimesh

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.constants import JointType, PRIMITIVE_MESH_TYPES, PrimType, GEOM_TYPES
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.ui_utils import suppress_omni_log

import omnigibson.utils.transform_utils as T


def array_to_vtarray(arr, element_type):
    """
    Converts array @arr into a Vt-typed array, where each individual element of type @element_type.

    Args:
        arr (n-array): An array of values. Can be, e.g., a list, or numpy array
        element_type (type): Per-element type to convert the elements from @arr into.
            Valid options are keys of GF_TO_VT_MAPPING

    Returns:
        Vt.Array: Vt-typed array, of specified type corresponding to @element_type
    """
    GF_TO_VT_MAPPING = {
        lazy.pxr.Gf.Vec3d: lazy.pxr.Vt.Vec3dArray,
        lazy.pxr.Gf.Vec3f: lazy.pxr.Vt.Vec3fArray,
        lazy.pxr.Gf.Vec3h: lazy.pxr.Vt.Vec3hArray,
        lazy.pxr.Gf.Quatd: lazy.pxr.Vt.QuatdArray,
        lazy.pxr.Gf.Quatf: lazy.pxr.Vt.QuatfArray,
        lazy.pxr.Gf.Quath: lazy.pxr.Vt.QuathArray,
        int: lazy.pxr.Vt.IntArray,
        float: lazy.pxr.Vt.FloatArray,
        bool: lazy.pxr.Vt.BoolArray,
        str: lazy.pxr.Vt.StringArray,
        chr: lazy.pxr.Vt.CharArray,
    }

    # Make sure array type is valid
    assert_valid_key(key=element_type, valid_keys=GF_TO_VT_MAPPING, name="array element type")

    # Construct list of values
    arr_list = []

    # Check first to see if elements are vectors or not. If this is an iterable value that is not a string,
    # then this is a vector and we have to map it to the correct type via *
    is_vec_element = (isinstance(arr[0], Iterable)) and (not isinstance(arr[0], str))

    # Loop over array and set values
    for ele in arr:
        arr_list.append(element_type(*ele) if is_vec_element else ele)

    return GF_TO_VT_MAPPING[element_type](arr_list)


def get_prim_nested_children(prim):
    """
    Grabs all nested prims starting from root @prim via depth-first-search

    Args:
        prim (Usd.Prim): root prim from which to search for nested children prims

    Returns:
        list of Usd.Prim: nested prims
    """
    prims = []
    for child in lazy.omni.isaac.core.utils.prims.get_prim_children(prim):
        prims.append(child)
        prims += get_prim_nested_children(prim=child)

    return prims


def get_camera_params(viewport):
    """
    Get active camera intrinsic and extrinsic parameters.

    Returns:
        dict: Keyword-mapped values of the active camera's parameters:

            pose (numpy.ndarray): camera position in world coordinates,
            fov (float): horizontal field of view in radians
            focal_length (float)
            horizontal_aperture (float)
            view_projection_matrix (numpy.ndarray(dtype=float64, shape=(4, 4)))
            resolution (dict): resolution as a dict with 'width' and 'height'.
            clipping_range (tuple(float, float)): Near and Far clipping values.
    """
    stage = lazy.omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(viewport.get_active_camera())
    prim_tf = lazy.omni.usd.get_world_transform_matrix(prim)
    view_params = lazy.omni.syntheticdata.helpers.get_view_params(viewport)
    fov = 2 * math.atan(view_params["horizontal_aperture"] / (2 * view_params["focal_length"]))
    view_proj_mat = lazy.omni.syntheticdata.helpers.get_view_proj_mat(view_params)

    return {
        "pose": np.array(prim_tf).T,        # omni natively gives transposed pose so we have to "un"-transpose it
        "fov": fov,
        "focal_length": view_params["focal_length"],
        "horizontal_aperture": view_params["horizontal_aperture"],
        "view_projection_matrix": view_proj_mat,
        "resolution": {"width": view_params["width"], "height": view_params["height"]},
        "clipping_range": np.array(view_params["clipping_range"]),
    }


def get_semantic_objects_pose():
    """
    Get pose of all objects with a semantic label.
    """
    stage = lazy.omni.usd.get_context().get_stage()
    mappings = lazy.omni.syntheticdata.helpers.get_instance_mappings()
    pose = []
    for m in mappings:
        prim_path = m[1]
        prim = stage.GetPrimAtPath(prim_path)
        prim_tf = lazy.omni.usd.get_world_transform_matrix(prim)
        pose.append((str(prim_path), m[2], str(m[3]), np.array(prim_tf)))
    return pose


def create_joint(prim_path, joint_type, body0=None, body1=None, enabled=True,
                 joint_frame_in_parent_frame_pos=None, joint_frame_in_parent_frame_quat=None,
                 joint_frame_in_child_frame_pos=None, joint_frame_in_child_frame_quat=None,
                 break_force=None, break_torque=None):
    """
    Creates a joint between @body0 and @body1 of specified type @joint_type

    Args:
        prim_path (str): absolute path to where the joint will be created
        joint_type (str or JointType): type of joint to create. Valid options are:
            "FixedJoint", "Joint", "PrismaticJoint", "RevoluteJoint", "SphericalJoint"
                        (equivalently, one of JointType)
        body0 (str or None): absolute path to the first body's prim. At least @body0 or @body1 must be specified.
        body1 (str or None): absolute path to the second body's prim. At least @body0 or @body1 must be specified.
        enabled (bool): whether to enable this joint or not.
        joint_frame_in_parent_frame_pos (np.ndarray or None): relative position of the joint frame to the parent frame (body0).
        joint_frame_in_parent_frame_quat (np.ndarray or None): relative orientation of the joint frame to the parent frame (body0).
        joint_frame_in_child_frame_pos (np.ndarray or None): relative position of the joint frame to the child frame (body1).
        joint_frame_in_child_frame_quat (np.ndarray or None): relative orientation of the joint frame to the child frame (body1).
        break_force (float or None): break force for linear dofs, unit is Newton.
        break_torque (float or None): break torque for angular dofs, unit is Newton-meter.

    Returns:
        Usd.Prim: Created joint prim
    """
    # Make sure we have valid joint_type
    assert JointType.is_valid(joint_type=joint_type), \
        f"Invalid joint specified for creation: {joint_type}"

    # Make sure at least body0 or body1 is specified
    assert body0 is not None or body1 is not None, \
        f"At least either body0 or body1 must be specified when creating a joint!"

    # Create the joint
    joint = getattr(lazy.pxr.UsdPhysics, joint_type).Define(og.sim.stage, prim_path)

    # Possibly add body0, body1 targets
    if body0 is not None:
        assert lazy.omni.isaac.core.utils.prims.is_prim_path_valid(body0), f"Invalid body0 path specified: {body0}"
        joint.GetBody0Rel().SetTargets([lazy.pxr.Sdf.Path(body0)])
    if body1 is not None:
        assert lazy.omni.isaac.core.utils.prims.is_prim_path_valid(body1), f"Invalid body1 path specified: {body1}"
        joint.GetBody1Rel().SetTargets([lazy.pxr.Sdf.Path(body1)])

    # Get the prim pointed to at this path
    joint_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path)

    # Apply joint API interface
    lazy.pxr.PhysxSchema.PhysxJointAPI.Apply(joint_prim)

    # We need to step rendering once to auto-fill the local pose before overwriting it.
    # Note that for some reason, if multi_gpu is used, this line will crash if create_joint is called during on_contact
    # callback, e.g. when an attachment joint is being created due to contacts.
    og.sim.render()

    if joint_frame_in_parent_frame_pos is not None:
        joint_prim.GetAttribute("physics:localPos0").Set(lazy.pxr.Gf.Vec3f(*joint_frame_in_parent_frame_pos))
    if joint_frame_in_parent_frame_quat is not None:
        joint_prim.GetAttribute("physics:localRot0").Set(lazy.pxr.Gf.Quatf(*joint_frame_in_parent_frame_quat[[3, 0, 1, 2]]))
    if joint_frame_in_child_frame_pos is not None:
        joint_prim.GetAttribute("physics:localPos1").Set(lazy.pxr.Gf.Vec3f(*joint_frame_in_child_frame_pos))
    if joint_frame_in_child_frame_quat is not None:
        joint_prim.GetAttribute("physics:localRot1").Set(lazy.pxr.Gf.Quatf(*joint_frame_in_child_frame_quat[[3, 0, 1, 2]]))

    if break_force is not None:
        joint_prim.GetAttribute("physics:breakForce").Set(break_force)
    if break_torque is not None:
        joint_prim.GetAttribute("physics:breakTorque").Set(break_torque)

    # Possibly (un-/)enable this joint
    joint_prim.GetAttribute("physics:jointEnabled").Set(enabled)

    # We update the simulation now without stepping physics if sim is playing so we can bypass the snapping warning from PhysicsUSD
    if og.sim.is_playing():
        with suppress_omni_log(channels=["omni.physx.plugin"]):
            og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)

    # Return this joint
    return joint_prim


class RigidContactAPI:
    """
    Class containing class methods to aggregate rigid body contacts across all rigid bodies in the simulator
    """
    # Dictionary mapping rigid body prim path to corresponding index in the contact view matrix
    _PATH_TO_ROW_IDX = None
    _PATH_TO_COL_IDX = None

    # Numpy array of rigid body prim paths where its array index directly corresponds to the corresponding
    # index in the contact view matrix
    _ROW_IDX_TO_PATH = None
    _COL_IDX_TO_PATH = None

    # Contact view for generating contact matrices at each timestep
    _CONTACT_VIEW = None

    # Current aggregated contacts over all rigid bodies at the current timestep. Shape: (N, N, 3)
    _CONTACT_MATRIX = None

    # Current cache, mapping 2-tuple (prim_paths_a, prim_paths_b) to contact values
    _CONTACT_CACHE = None

    @classmethod
    def initialize_view(cls):
        """
        Initializes the rigid contact view. Note: Can only be done when sim is playing!
        """
        assert og.sim.is_playing(), "Cannot create rigid contact view while sim is not playing!"

        # Compile deterministic mapping from rigid body path to idx
        # Note that omni's ordering is based on the top-down object ordering path on the USD stage, which coincidentally
        # matches the same ordering we store objects in our registry. So the mapping we generate from our registry
        # mapping aligns with omni's ordering!
        i = 0
        cls._PATH_TO_COL_IDX = dict()
        for obj in og.sim.scene.objects:
            if obj.prim_type == PrimType.RIGID:
                for link in obj.links.values():
                    if not link.kinematic_only:
                        cls._PATH_TO_COL_IDX[link.prim_path] = i
                        i += 1

        # If there are no valid objects, clear the view and terminate early
        if i == 0:
            cls._CONTACT_VIEW = None
            return

        # Generate rigid body view, making sure to update the simulation first (without physics) so that the physx
        # backend is synchronized with any newly added objects
        # We also suppress the omni tensor plugin from giving warnings we expect
        og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)
        with suppress_omni_log(channels=["omni.physx.tensors.plugin"]):
            cls._CONTACT_VIEW = og.sim.physics_sim_view.create_rigid_contact_view(
                pattern="/World/*/*",
                filter_patterns=list(cls._PATH_TO_COL_IDX.keys()),
            )

        # Create deterministic mapping from path to row index
        cls._PATH_TO_ROW_IDX = {path: i for i, path in enumerate(cls._CONTACT_VIEW.sensor_paths)}

        # Store the reverse mappings as well. This can just be a numpy array since the mapping uses integer indices
        cls._ROW_IDX_TO_PATH = np.array(list(cls._PATH_TO_ROW_IDX.keys()))
        cls._COL_IDX_TO_PATH = np.array(list(cls._PATH_TO_COL_IDX.keys()))

        # Sanity check generated view -- this should generate square matrices of shape (N, N, 3)
        n_bodies = len(cls._PATH_TO_COL_IDX)
        assert cls._CONTACT_VIEW.filter_count == n_bodies, \
            f"Got unexpected contact view shape. Expected: (N, {n_bodies}); " \
            f"got: (N, {cls._CONTACT_VIEW.filter_count})"

    @classmethod
    def get_body_row_idx(cls, prim_path):
        """
        Returns:
            int: row idx assigned to the rigid body defined by @prim_path
        """
        return cls._PATH_TO_ROW_IDX[prim_path]

    @classmethod
    def get_body_col_idx(cls, prim_path):
        """
        Returns:
            int: col idx assigned to the rigid body defined by @prim_path
        """
        return cls._PATH_TO_COL_IDX[prim_path]

    @classmethod
    def get_row_idx_prim_path(cls, idx):
        """
        Returns:
            str: @prim_path corresponding to the row idx @idx in the contact matrix
        """
        return cls._ROW_IDX_TO_PATH[idx]

    @classmethod
    def get_col_idx_prim_path(cls, idx):
        """
        Returns:
            str: @prim_path corresponding to the column idx @idx in the contact matrix
        """
        return cls._COL_IDX_TO_PATH[idx]

    @classmethod
    def get_all_impulses(cls):
        """
        Grab all impulses at the current timestep

        Returns:
            n-array: (N, M, 3) impulse array defining current impulses between all N contact-sensor enabled rigid bodies
                in the simulator and M tracked rigid bodies
        """
        # Generate the contact matrix if it doesn't already exist
        if cls._CONTACT_MATRIX is None:
            cls._CONTACT_MATRIX = cls._CONTACT_VIEW.get_contact_force_matrix(dt=1.0)

        return cls._CONTACT_MATRIX

    @classmethod
    def get_impulses(cls, prim_paths_a, prim_paths_b):
        """
        Grabs the matrix representing all impulse forces between rigid prims from @prim_paths_a and
        rigid prims from @prim_paths_b

        Args:
            prim_paths_a (list of str): Rigid body prim path(s) with which to grab contact impulses against
                any of the rigid body prim path(s) defined by @prim_paths_b
            prim_paths_b (list of str): Rigid body prim path(s) with which to grab contact impulses against
                any of the rigid body prim path(s) defined by @prim_paths_a

        Returns:
            n-array: (N, M, 3) impulse array defining current impulses between N bodies from @prim_paths_a and M bodies
                from @prim_paths_b
        """
        # Compute subset of matrix and return
        idxs_a = [cls._PATH_TO_ROW_IDX[path] for path in prim_paths_a]
        idxs_b = [cls._PATH_TO_COL_IDX[path] for path in prim_paths_b]
        return cls.get_all_impulses()[idxs_a][:, idxs_b]

    @classmethod
    def in_contact(cls, prim_paths_a, prim_paths_b):
        """
        Check if any rigid prim from @prim_paths_a is in contact with any rigid prim from @prim_paths_b

        Args:
            prim_paths_a (list of str): Rigid body prim path(s) with which to check contact against any of the rigid
                body prim path(s) defined by @prim_paths_b
            prim_paths_b (list of str): Rigid body prim path(s) with which to check contact against any of the rigid
                body prim path(s) defined by @prim_paths_a

        Returns:
            bool: Whether any body from @prim_paths_a is in contact with any body from @prim_paths_b
        """
        # Check if the contact tuple already exists in the cache; if so, return the value
        key = (tuple(prim_paths_a), tuple(prim_paths_b))
        if key not in cls._CONTACT_CACHE:
            # In contact if any of the matrix values representing the interaction between the two groups is non-zero
            cls._CONTACT_CACHE[key] = np.any(cls.get_impulses(prim_paths_a=prim_paths_a, prim_paths_b=prim_paths_b))
        return cls._CONTACT_CACHE[key]

    @classmethod
    def clear(cls):
        """
        Clears the internal contact matrix and cache
        """
        cls._CONTACT_MATRIX = None
        cls._CONTACT_CACHE = dict()


class CollisionAPI:
    """
    Class containing class methods to facilitate collision handling, e.g. collision groups
    """
    ACTIVE_COLLISION_GROUPS = dict()

    @classmethod
    def create_collision_group(cls, col_group, filter_self_collisions=False):
        """
        Creates a new collision group with name @col_group

        Args:
            col_group (str): Name of the collision group to create
            filter_self_collisions (bool): Whether to ignore self-collisions within the group. Default is False
        """
        # Can only be done when sim is stopped
        assert og.sim.is_stopped(), "Cannot create a collision group unless og.sim is stopped!"

        # Make sure the group doesn't already exist
        assert col_group not in cls.ACTIVE_COLLISION_GROUPS, \
            f"Cannot create collision group {col_group} because it already exists!"

        # Create the group
        col_group_prim_path = f"/World/collision_groups/{col_group}"
        group = lazy.pxr.UsdPhysics.CollisionGroup.Define(og.sim.stage, col_group_prim_path)
        if filter_self_collisions:
            # Do not collide with self
            group.GetFilteredGroupsRel().AddTarget(col_group_prim_path)
        cls.ACTIVE_COLLISION_GROUPS[col_group] = group

    @classmethod
    def add_to_collision_group(cls, col_group, prim_path):
        """
        Adds the prim and all nested prims specified by @prim_path to the global collision group @col_group. If @col_group
        does not exist, then it will either be created if @create_if_not_exist is True, otherwise will raise an Error.
        Args:
            col_group (str): Name of the collision group to assign the prim at @prim_path to
            prim_path (str): Prim (and all nested prims) to assign to this @col_group
        """
        # Make sure collision group exists
        assert col_group in cls.ACTIVE_COLLISION_GROUPS, \
            f"Cannot add to collision group {col_group} because it does not exist!"

        # Add this prim to the collision group
        cls.ACTIVE_COLLISION_GROUPS[col_group].GetCollidersCollectionAPI().GetIncludesRel().AddTarget(prim_path)

    @classmethod
    def add_group_filter(cls, col_group, filter_group):
        """
        Adds a new group filter for group @col_group, filtering all collision with group @filter_group
        Args:
            col_group (str): Name of the collision group which will have a new filter group added
            filter_group (str): Name of the group that should be filtered
        """
        # Make sure the group doesn't already exist
        for group_name in (col_group, filter_group):
            assert group_name in cls.ACTIVE_COLLISION_GROUPS, \
                (f"Cannot add group filter {filter_group} to collision group {col_group} because at least one group "
                 f"does not exist!")

        # Grab the group, and add the filter
        filter_group_prim_path = f"/World/collision_groups/{filter_group}"
        group = cls.ACTIVE_COLLISION_GROUPS[col_group]
        group.GetFilteredGroupsRel().AddTarget(filter_group_prim_path)

    @classmethod
    def clear(cls):
        """
        Clears the internal state of this CollisionAPI
        """
        cls.ACTIVE_COLLISION_GROUPS = {}


def clear():
    """
    Clear state tied to singleton classes
    """
    CollisionAPI.clear()


def create_mesh_prim_with_default_xform(primitive_type, prim_path, u_patches=None, v_patches=None, stage=None):
    """
    Creates a mesh prim of the specified @primitive_type at the specified @prim_path

    Args:
        primitive_type (str): Primitive mesh type, should be one of PRIMITIVE_MESH_TYPES to be valid
        prim_path (str): Destination prim path to store the mesh prim
        u_patches (int or None): If specified, should be an integer that represents how many segments to create in the
            u-direction. E.g. 10 means 10 segments (and therefore 11 vertices) will be created.
        v_patches (int or None): If specified, should be an integer that represents how many segments to create in the
            v-direction. E.g. 10 means 10 segments (and therefore 11 vertices) will be created.
            Both u_patches and v_patches need to be specified for them to be effective.
        stage (None or Usd.Stage): If specified, stage on which the primitive mesh should be generated. If None, will
            use og.sim.stage
    """
    MESH_PRIM_TYPE_TO_EVALUATOR_MAPPING = {
        "Sphere": lazy.omni.kit.primitive.mesh.evaluators.sphere.SphereEvaluator,
        "Disk": lazy.omni.kit.primitive.mesh.evaluators.disk.DiskEvaluator,
        "Plane": lazy.omni.kit.primitive.mesh.evaluators.plane.PlaneEvaluator,
        "Cylinder": lazy.omni.kit.primitive.mesh.evaluators.cylinder.CylinderEvaluator,
        "Torus": lazy.omni.kit.primitive.mesh.evaluators.torus.TorusEvaluator,
        "Cone": lazy.omni.kit.primitive.mesh.evaluators.cone.ConeEvaluator,
        "Cube": lazy.omni.kit.primitive.mesh.evaluators.cube.CubeEvaluator,
    }

    assert primitive_type in PRIMITIVE_MESH_TYPES, "Invalid primitive mesh type: {primitive_type}"
    evaluator = MESH_PRIM_TYPE_TO_EVALUATOR_MAPPING[primitive_type]
    u_backup = lazy.carb.settings.get_settings().get(evaluator.SETTING_U_SCALE)
    v_backup = lazy.carb.settings.get_settings().get(evaluator.SETTING_V_SCALE)
    hs_backup = lazy.carb.settings.get_settings().get(evaluator.SETTING_OBJECT_HALF_SCALE)
    lazy.carb.settings.get_settings().set(evaluator.SETTING_U_SCALE, 1)
    lazy.carb.settings.get_settings().set(evaluator.SETTING_V_SCALE, 1)
    stage = og.sim.stage if stage is None else stage

    # Default half_scale (i.e. half-extent, half_height, radius) is 1.
    # TODO (eric): change it to 0.5 once the mesh generator API accepts floating-number HALF_SCALE
    #  (currently it only accepts integer-number and floors 0.5 into 0).
    lazy.carb.settings.get_settings().set(evaluator.SETTING_OBJECT_HALF_SCALE, 1)
    kwargs = dict(prim_type=primitive_type, prim_path=prim_path, stage=stage)
    if u_patches is not None and v_patches is not None:
        kwargs["u_patches"] = u_patches
        kwargs["v_patches"] = v_patches

    # Import now to avoid too-eager load of Omni classes due to inheritance
    from omnigibson.utils.deprecated_utils import CreateMeshPrimWithDefaultXformCommand
    CreateMeshPrimWithDefaultXformCommand(**kwargs).do()

    lazy.carb.settings.get_settings().set(evaluator.SETTING_U_SCALE, u_backup)
    lazy.carb.settings.get_settings().set(evaluator.SETTING_V_SCALE, v_backup)
    lazy.carb.settings.get_settings().set(evaluator.SETTING_OBJECT_HALF_SCALE, hs_backup)


def mesh_prim_to_trimesh_mesh(mesh_prim, include_normals=True, include_texcoord=True):
    """
    Generates trimesh mesh from @mesh_prim

    Args:
        mesh_prim (Usd.Prim): Mesh prim to convert into trimesh mesh
        include_normals (bool): Whether to include the normals in the resulting trimesh or not
        include_texcoord (bool): Whether to include the corresponding 2D-texture coordinates in the resulting
            trimesh or not

    Returns:
        trimesh.Trimesh: Generated trimesh mesh
    """
    face_vertex_counts = np.array(mesh_prim.GetAttribute("faceVertexCounts").Get())
    vertices = np.array(mesh_prim.GetAttribute("points").Get())
    face_indices = np.array(mesh_prim.GetAttribute("faceVertexIndices").Get())

    faces = []
    i = 0
    for count in face_vertex_counts:
        for j in range(count - 2):
            faces.append([face_indices[i], face_indices[i + j + 1], face_indices[i + j + 2]])
        i += count

    kwargs = dict(vertices=vertices, faces=faces)

    if include_normals:
        kwargs["vertex_normals"] = np.array(mesh_prim.GetAttribute("normals").Get())

    if include_texcoord:
        kwargs["visual"] = trimesh.visual.TextureVisuals(uv=np.array(mesh_prim.GetAttribute("primvars:st").Get()))

    return trimesh.Trimesh(**kwargs)


def sample_mesh_keypoints(mesh_prim, n_keypoints, n_keyfaces, seed=None):
    """
    Samples keypoints and keyfaces for mesh @mesh_prim

    Args:
        mesh_prim (Usd.Prim): Mesh prim to be sampled from
        n_keypoints (int): number of (unique) keypoints to randomly sample from @mesh_prim
        n_keyfaces (int): number of (unique) keyfaces to randomly sample from @mesh_prim
        seed (None or int): If set, sets the random seed for deterministic results

    Returns:
        2-tuple:
            - n-array: (n,) 1D int array representing the randomly sampled point idxs from @mesh_prim.
                Note that since this is without replacement, the total length of the array may be less than
                @n_keypoints
            - None or n-array: 1D int array representing the randomly sampled face idxs from @mesh_prim.
                Note that since this is without replacement, the total length of the array may be less than
                @n_keyfaces
    """
    # Set seed if deterministic
    if seed is not None:
        np.random.seed(seed)

    # Generate trimesh mesh from which to aggregate points
    tm = mesh_prim_to_trimesh_mesh(mesh_prim=mesh_prim, include_normals=False, include_texcoord=False)
    n_unique_vertices, n_unique_faces = len(tm.vertices), len(tm.faces)
    faces_flat = tm.faces.flatten()
    n_vertices = len(faces_flat)

    # Sample vertices
    unique_vertices = np.unique(faces_flat)
    assert len(unique_vertices) == n_unique_vertices
    keypoint_idx = np.random.choice(unique_vertices, size=n_keypoints, replace=False) if \
        n_unique_vertices > n_keypoints else unique_vertices

    # Sample faces
    keyface_idx = np.random.choice(n_unique_faces, size=n_keyfaces, replace=False) if \
        n_unique_faces > n_keyfaces else np.arange(n_unique_faces)

    return keypoint_idx, keyface_idx


def get_mesh_volume_and_com(mesh_prim):
    """
    Computes the volume and center of mass for @mesh_prim

    Args:
        mesh_prim (Usd.Prim): Mesh prim to compute volume and center of mass for

    Returns:
        Tuple[bool, float, np.array]: Tuple containing the (is_volume, volume, center_of_mass) in the mesh
            frame of @mesh_prim
    """
    mesh_type = mesh_prim.GetPrimTypeInfo().GetTypeName()
    assert mesh_type in GEOM_TYPES, f"Invalid mesh type: {mesh_type}"
    # Default volume and com
    volume = 0.0
    com = np.zeros(3)
    is_volume = True
    if mesh_type == "Mesh":
        # We construct a trimesh object from this mesh in order to infer its volume
        trimesh_mesh = mesh_prim_to_trimesh_mesh(mesh_prim, include_normals=False, include_texcoord=False)
        is_volume = trimesh_mesh.is_volume
        if is_volume:
            volume = trimesh_mesh.volume
            com = trimesh_mesh.center_mass
        else:
            # If the mesh is not a volume, we compute its convex hull and use that instead
            try:
                trimesh_mesh_convex = trimesh_mesh.convex_hull
                volume = trimesh_mesh_convex.volume
                com = trimesh_mesh_convex.center_mass
            except:
                # if convex hull computation fails, it usually means the mesh is degenerated. We just skip it.
                pass
    elif mesh_type == "Sphere":
        volume = 4 / 3 * np.pi * (mesh_prim.GetAttribute("radius").Get() ** 3)
    elif mesh_type == "Cube":
        volume = mesh_prim.GetAttribute("size").Get() ** 3
    elif mesh_type == "Cone":
        volume = np.pi * (mesh_prim.GetAttribute("radius").Get() ** 2) * mesh_prim.GetAttribute("height").Get() / 3
        com = np.array([0, 0, mesh_prim.GetAttribute("height").Get() / 4])
    elif mesh_type == "Cylinder":
        volume = np.pi * (mesh_prim.GetAttribute("radius").Get() ** 2) * mesh_prim.GetAttribute("height").Get()
    else:
        raise ValueError(f"Cannot compute volume for mesh of type: {mesh_type}")

    return is_volume, volume, com

def check_extent_radius_ratio(mesh_prim):
    """
    Checks if the extent radius ratio of @mesh_prim is within the acceptable range for PhysX GPU acceleration (not too oblong)

    Ref: https://github.com/NVIDIA-Omniverse/PhysX/blob/561a0df858d7e48879cdf7eeb54cfe208f660f18/physx/source/geomutils/src/convex/GuConvexMeshData.h#L183-L190

    Args:
        mesh_prim (Usd.Prim): Mesh prim to check the extent radius ratio for

    Returns:
        bool: True if the extent radius ratio is within the acceptable range, False otherwise
    """

    mesh_type = mesh_prim.GetPrimTypeInfo().GetTypeName()
    assert mesh_type in GEOM_TYPES, f"Invalid mesh type: {mesh_type}"

    if mesh_type != "Mesh":
        return True

    is_volume, _, com = get_mesh_volume_and_com(mesh_prim)

    trimesh_mesh = mesh_prim_to_trimesh_mesh(mesh_prim, include_normals=False, include_texcoord=False)
    if not is_volume:
        trimesh_mesh = trimesh_mesh.convex_hull

    max_radius = trimesh_mesh.extents.max() / 2.0
    min_radius = trimesh.proximity.closest_point(trimesh_mesh, np.array([com]))[1][0]
    ratio = max_radius / min_radius

    # PhysX requires ratio to be < 100.0. We use 95.0 to be safe.
    return ratio < 95.0

def create_primitive_mesh(prim_path, primitive_type, extents=1.0, u_patches=None, v_patches=None, stage=None):
    """
    Helper function that generates a UsdGeom.Mesh prim at specified @prim_path of type @primitive_type.

    NOTE: Generated mesh prim will, by default, have extents equaling [1, 1, 1]

    Args:
        prim_path (str): Where the loaded mesh should exist on the stage
        primitive_type (str): Type of primitive mesh to create. Should be one of:
            {"Cone", "Cube", "Cylinder", "Disk", "Plane", "Sphere", "Torus"}
        extents (float or 3-array): Specifies the extents of the generated mesh. Default is 1.0, i.e.:
            generated mesh will be in be contained in a [1,1,1] sized bounding box
        u_patches (int or None): If specified, should be an integer that represents how many segments to create in the
            u-direction. E.g. 10 means 10 segments (and therefore 11 vertices) will be created.
        v_patches (int or None): If specified, should be an integer that represents how many segments to create in the
            v-direction. E.g. 10 means 10 segments (and therefore 11 vertices) will be created.
            Both u_patches and v_patches need to be specified for them to be effective.
        stage (None or Usd.Stage): If specified, stage on which the primitive mesh should be generated. If None, will
            use og.sim.stage

    Returns:
        UsdGeom.Mesh: Generated primitive mesh as a prim on the active stage
    """
    assert_valid_key(key=primitive_type, valid_keys=PRIMITIVE_MESH_TYPES, name="primitive mesh type")
    create_mesh_prim_with_default_xform(primitive_type, prim_path, u_patches=u_patches, v_patches=v_patches, stage=stage)
    mesh = lazy.pxr.UsdGeom.Mesh.Define(og.sim.stage if stage is None else stage, prim_path)

    # Modify the points and normals attributes so that total extents is the desired
    # This means multiplying omni's default by extents * 50.0, as the native mesh generated has extents [-0.01, 0.01]
    # -- i.e.: 2cm-wide mesh
    extents = np.ones(3) * extents if isinstance(extents, float) else np.array(extents)
    for attr in (mesh.GetPointsAttr(), mesh.GetNormalsAttr()):
        vals = np.array(attr.Get()).astype(np.float64)
        attr.Set(lazy.pxr.Vt.Vec3fArray([lazy.pxr.Gf.Vec3f(*(val * extents * 50.0)) for val in vals]))
    mesh.GetExtentAttr().Set(lazy.pxr.Vt.Vec3fArray([lazy.pxr.Gf.Vec3f(*(-extents / 2.0)), lazy.pxr.Gf.Vec3f(*(extents / 2.0))]))

    return mesh


def add_asset_to_stage(asset_path, prim_path):
    """
    Adds asset file (either USD or OBJ) at @asset_path at the location @prim_path

    Args:
        asset_path (str): Absolute or relative path to the asset file to load
        prim_path (str): Where loaded asset should exist on the stage

    Returns:
        Usd.Prim: Loaded prim as a USD prim
    """
    # Make sure this is actually a supported asset type
    assert asset_path[-4:].lower() in {".usd", ".obj"}, f"Cannot load a non-USD or non-OBJ file as a USD prim!"
    asset_type = asset_path[-3:]

    # Make sure the path exists
    assert os.path.exists(asset_path), f"Cannot load {asset_type.upper()} file {asset_path} because it does not exist!"

    # Add reference to stage and grab prim
    lazy.omni.isaac.core.utils.stage.add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
    prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path)

    # Make sure prim was loaded correctly
    assert prim, f"Failed to load {asset_type.upper()} object from path: {asset_path}"

    return prim


def get_world_prim():
    """
    Returns:
        Usd.Prim: Active world prim in the current stage
    """
    return lazy.omni.isaac.core.utils.prims.get_prim_at_path("/World")
