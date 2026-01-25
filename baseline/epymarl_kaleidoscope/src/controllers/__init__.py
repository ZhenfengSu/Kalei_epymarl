REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .kalei_controller import Kalei_MAC
from .snp_controller import SNP_MAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["kalei_mac"] = Kalei_MAC
REGISTRY["snp_mac"] = SNP_MAC