REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .Kalei_controller import Kalei_MAC
from .SNP_controller import SNP_MAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["Kalei_mac"] = Kalei_MAC
REGISTRY["SNP_mac"] = SNP_MAC
