from typing import TypedDict


class WorkflowState(TypedDict):

    question: str
    context: str
    answer: str

    decision: str
    score: float
    feedback: str

    iteration: int