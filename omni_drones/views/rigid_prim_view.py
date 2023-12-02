import torch
from typing import Optional, Tuple, List
from contextlib import contextmanager

from typing import List, Optional, Tuple, Union
import numpy as np
from omni.isaac.core.utils.prims import get_prim_parent, get_prim_at_path, set_prim_property, get_prim_property
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
from omni.isaac.core.prims import RigidPrimView as _RigidPrimView
import omni

from .utils import require_sim_initialized, disable_warnings

class RigidPrimView(_RigidPrimView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: str = "rigid_prim_view",
        positions: Optional[torch.Tensor] = None,
        translations: Optional[torch.Tensor] = None,
        orientations: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        visibilities: Optional[torch.Tensor] = None,
        reset_xform_properties: bool = True,
        masses: Optional[torch.Tensor] = None,
        densities: Optional[torch.Tensor] = None,
        linear_velocities: Optional[torch.Tensor] = None,
        angular_velocities: Optional[torch.Tensor] = None,
        track_contact_forces: bool = False,
        prepare_contact_sensors: bool = True,
        disable_stablization: bool = True,
        contact_filter_prim_paths_expr: Optional[List[str]] = (),
        shape: Tuple[int, ...] = (-1,),
    ) -> None:
        self.shape = shape
        super().__init__(
            prim_paths_expr,
            name,
            positions,
            translations,
            orientations,
            scales,
            visibilities,
            reset_xform_properties,
            masses,
            densities,
            linear_velocities,
            angular_velocities,
            track_contact_forces,
            prepare_contact_sensors,
            disable_stablization,
            contact_filter_prim_paths_expr,
        )

    @require_sim_initialized
    def initialize(self, physics_sim_view: omni.physics.tensors.SimulationView = None):
        super().initialize(physics_sim_view)
        self.shape = torch.arange(self.count).reshape(self.shape).shape
        return self

    def get_world_poses(
        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self._resolve_env_indices(env_indices)
        pos, rot = super().get_world_poses(indices, clone)
        return pos.unflatten(0, self.shape), rot.unflatten(0, self.shape)

    def set_world_poses(
        self,
        positions: Optional[torch.Tensor] = None,
        orientations: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
    ) -> None:
        with disable_warnings(self._physics_sim_view):
            indices = self._resolve_env_indices(env_indices)
            poses = self._physics_view.get_transforms()
            if positions is not None:
                poses[indices, :3] = positions.reshape(-1, 3)
            if orientations is not None:
                poses[indices, 3:] = orientations.reshape(-1, 4)[:, [1, 2, 3, 0]]
            self._physics_view.set_transforms(poses, indices)

    def get_velocities(
        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().get_velocities(indices, clone).unflatten(0, self.shape)

    def set_velocities(
        self, velocities: torch.Tensor, env_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().set_velocities(velocities.reshape(-1, 6), indices)

    def get_net_contact_forces(
        self,
        env_indices: Optional[torch.Tensor] = None,
        clone: bool = False,
        dt: float = 1,
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return (
            super().get_net_contact_forces(indices, clone, dt).unflatten(0, self.shape)
        )

    def get_contact_force_matrix(
        self, 
        env_indices: Optional[torch.Tensor] = None, 
        clone: bool = True, 
        dt: float = 1
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().get_contact_force_matrix(indices, clone, dt).unflatten(0, self.shape)

    def get_masses(
        self, 
        env_indices: Optional[torch.Tensor] = None, 
        clone: bool = True
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            current_values = self._backend_utils.move_data(self._physics_view.get_masses(), self._device)
            masses = current_values[indices]
            if clone:
                masses = self._backend_utils.clone_tensor(masses, device=self._device)
        else:
            masses = self._backend_utils.create_zeros_tensor([indices.shape[0]], dtype="float32", device=self._device)
            write_idx = 0
            for i in indices:
                if self._mass_apis[i.tolist()] is None:
                    if self._prims[i.tolist()].HasAPI(UsdPhysics.MassAPI):
                        self._mass_apis[i.tolist()] = UsdPhysics.MassAPI(self._prims[i.tolist()])
                    else:
                        self._mass_apis[i.tolist()] = UsdPhysics.MassAPI.Apply(self._prims[i.tolist()])
                masses[write_idx] = self._backend_utils.create_tensor_from_list(
                    self._mass_apis[i.tolist()].GetMassAttr().Get(), dtype="float32", device=self._device
                )
                write_idx += 1
        return masses.reshape(-1, *self.shape[1:], 1)
    
    def set_masses(
        self, 
        masses: torch.Tensor, 
        env_indices: Optional[torch.Tensor] = None
    ) -> None:
        indices = self._resolve_env_indices(env_indices).cpu()
        masses = masses.reshape(-1, 1)
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            data = self._backend_utils.clone_tensor(self._physics_view.get_masses(), device="cpu")
            data[indices] = self._backend_utils.move_data(masses, device="cpu")
            self._physics_view.set_masses(data, indices)
        else:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            read_idx = 0
            for i in indices:
                if self._mass_apis[i.tolist()] is None:
                    if self._prims[i.tolist()].HasAPI(UsdPhysics.MassAPI):
                        self._mass_apis[i.tolist()] = UsdPhysics.MassAPI(self._prims[i.tolist()])
                    else:
                        self._mass_apis[i.tolist()] = UsdPhysics.MassAPI.Apply(self._prims[i.tolist()])
                self._mass_apis[i.tolist()].GetMassAttr().Set(masses[read_idx].tolist())
                read_idx += 1
            return

    def get_coms(
        self, 
        env_indices: Optional[torch.Tensor] = None, 
        clone: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self._resolve_env_indices(env_indices)
        positions, orientations = super().get_coms(indices, clone)
        return positions.unflatten(0, self.shape), orientations.unflatten(0, self.shape)
    
    def set_coms(
        self, 
        positions: torch.Tensor = None, 
        # orientations: torch.Tensor = None, 
        env_indices: torch.Tensor = None
    ) -> None:
        # TODO@btx0424 fix orientations
        indices = self._resolve_env_indices(env_indices)
        return super().set_coms(positions.reshape(-1, 3), None, indices)
    
    def get_inertias(
        self, 
        env_indices: Optional[torch.Tensor]=None, 
        clone: bool=True
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().get_inertias(indices, clone).unflatten(0, self.shape)
    
    def set_inertias(
        self, 
        values: torch.Tensor, 
        env_indices: Optional[torch.Tensor]=None
    ):
        indices = self._resolve_env_indices(env_indices)
        return super().set_inertias(values.reshape(-1, 9), indices)

    def _resolve_env_indices(self, env_indices: torch.Tensor):
        if not hasattr(self, "_all_indices"):
            self._all_indices = torch.arange(self.count, device=self._device)
            self.shape = self._all_indices.reshape(self.shape).shape
        if env_indices is not None:
            indices = self._all_indices.reshape(self.shape)[env_indices].flatten()
        else:
            indices = self._all_indices
        return indices

    def squeeze_(self, dim: int = None):
        self.shape = self._all_indices.reshape(self.shape).squeeze(dim).shape
        return self
