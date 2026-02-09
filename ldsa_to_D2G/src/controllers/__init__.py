REGISTRY = {}

from .basic_controller import BasicMAC

REGISTRY["basic_mac"] = BasicMAC

from .ldsa_controller import LDSAMAC

REGISTRY["ldsa_mac"] = LDSAMAC