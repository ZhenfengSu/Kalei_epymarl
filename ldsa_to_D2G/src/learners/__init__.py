from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .ldsa_learner import LDSALearner
from .ldsa_k24_learner import LDSAK24Learner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["ldsa_learner"] = LDSALearner
REGISTRY["ldsa_k24_learner"] = LDSAK24Learner