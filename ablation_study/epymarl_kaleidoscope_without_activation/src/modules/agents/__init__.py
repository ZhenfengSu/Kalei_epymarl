from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .kalei_rnn_agent import Kalei_RNNAgent_1R3
from .k24_rnn_agent import K24_RNNAgent_1R3


REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
REGISTRY["kalei_rnn_1R3"] = Kalei_RNNAgent_1R3
REGISTRY["k24_rnn_1R3"] = K24_RNNAgent_1R3
