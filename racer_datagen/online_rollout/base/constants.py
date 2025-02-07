
class WptType:
    EXPERT = "expert"
    PERTURB = "perturb"
    INTERMEDIATE = "intermediate"
    DENSE = "dense"

# AgentType = PerturbType
class PerturbType:
    HEURISTIC = "heuristic"
    RVT = "rvt"
    CMD = "cmd"

PERTURB_TYPE_LIST = [getattr(PerturbType, attr) for attr in dir(PerturbType) if not attr.startswith("__")]
PERTURB_EPISODE_PKL_LIST = ["heuristic_episode.pkl", "rvt_episode.pkl", "annotated_episode.pkl"]

class TransitionType:
    SUCCESS = "success"
    ONGOING = "ongoing"
    RECOVERABLE_FAILURE = "recoverable_failure"
    CATASTROPHIC_FAILURE = "catastrophic_failure"

class WptInfoKey:
    WPT_TYPE = "wpt_type"
    WPT_ID = "wpt_id"
    PERTURB_TYPE = "perturb_type"
    PERTURB_IDX = "perturb_idx"
    TRANSITION_TYPE = "transition_type"
    VERBOSE = "verbose"
    LANG = "lang"
    FAILURE_REASON = "failure_reason"
    SCENE_INFO = "scene_info"
    CURRENT_POSE = "current_pose"
    NEXT_POSE = "next_pose"


START_ACTION = [0.27844, -0.00816, 1.47198, 0.0, 0.99266, -0.0, 0.1209, 1.0, 0.0]

RED = '\033[31m'  # Red text
GREEN = '\033[32m'  # Green text
BLUE = '\033[34m'  # Blue text
RESET = '\033[0m'  # Reset to default color
