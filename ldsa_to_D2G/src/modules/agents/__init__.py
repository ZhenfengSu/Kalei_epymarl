REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .ldsa_agent import LDSAAgent
REGISTRY["ldsa"] = LDSAAgent

from .ldsa_k24_agent import K24SparseLDSAAgent
REGISTRY["ldsa_k24"] = K24SparseLDSAAgent