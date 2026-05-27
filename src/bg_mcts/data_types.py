from typing import Dict, Literal, Tuple, TypeAlias, get_args, Union

Action: TypeAlias = Literal["transform", "question", "multi_questions", "answer"]
ACTIONS: Tuple[Action, ...] = get_args(Action)

Math500ProbData: TypeAlias = Dict[str, Union[str, int]]
AIMEProbData: TypeAlias = Dict[str, Union[str, int]]
MinervamathProbData: TypeAlias = Dict[str, Union[str, int]]
