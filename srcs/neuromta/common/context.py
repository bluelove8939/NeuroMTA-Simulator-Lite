from typing import Any

from neuromta.common.core import Core


__all__ = ['Context']


class Context:
    def __init__(self, n_owners_per_res: int = -1):
        self._n_owners_per_res = n_owners_per_res
        self._registered_owners: dict[Any, list[Core]] = {}
        
    def register_owner(self, resource: Any, owner: Core) -> int:
        if resource not in self._registered_owners:
             self._registered_owners[resource] = []
        
        if len(self._registered_owners[resource]) >= self._n_owners_per_res and self._n_owners_per_res > 0:
            raise Exception(f"[ERROR] Maximum number of owners ({self._n_owners_per_res}) for resource type {resource} exceeded.")
        
        self._registered_owners[resource].append(owner)
        
        return len(self._registered_owners[resource]) - 1  # Return the index of the newly registered owner
    
    def get_owner(self, res_type: Any, owner_id: int=0) -> Core:
        if res_type not in self._registered_owners:
            raise Exception(f"[ERROR] Resource type {res_type} is not registered in the context.")
        if owner_id < 0 or owner_id >= len(self._registered_owners[res_type]):
            raise Exception(f"[ERROR] Owner ID {owner_id} not found for resource type {res_type}.")
        
        return self._registered_owners[res_type][owner_id]
    
    def is_valid_resource_type(self, res_type: Any) -> bool:
        return res_type in self._registered_owners