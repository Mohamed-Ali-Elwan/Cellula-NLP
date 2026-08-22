from generator.generator import GeneratorRunnable
from evaluator.evaluator import EvaluatorRunnable

from workflow.state import WorkflowState


class GeneratorEvaluatorWorkflow:

    def __init__(self, generator, evaluator, retriever):

        self.generator = generator
        self.evaluator = evaluator
        self.retriever = retriever

        self.max_iterations = 4
        
        
    def retrieve(self, state: WorkflowState) -> WorkflowState:

        context = self.retriever.retrieve(
            state["question"]
        )

        state["context"] = context

        return state    
    
    def generate(self, state: WorkflowState) -> WorkflowState:

        answer = self.generator.run(
            question=state["question"],
            context=state["context"],
            feedback=state["feedback"],
            session_id="generator_session",
        )

        state["answer"] = answer

        return state
    
    def evaluate(self, state: WorkflowState) -> WorkflowState:

        evaluation = self.evaluator.run(
            question=state["question"],
            answer=state["answer"],
            context=state["context"],
            session_id="evaluator_session",
        )

        state["feedback"] = evaluation

        return state
    
    def should_continue(self, state: WorkflowState):

        if state["decision"] == "accept":
            return "final"

        if state["iteration"] >= self.max_iterations:
            return "final"

        return "improve"
    
    def run(self, question: str):

        state: WorkflowState = {
            "question": question,
            "context": "",
            "answer": "",
            "feedback": "",
            "decision": "",
            "score": 0.0,
            "iteration": 0,
        }

        # Retrieve external knowledge
        state = self.retrieve(state)

        while state["iteration"] < self.max_iterations:

            # Generate answer
            state = self.generate(state)

            # Evaluate answer
            state = self.evaluate(state)

            # Check decision
            if state["decision"] == "accept":
                break

            # Prepare for next iteration
            state["iteration"] += 1

        return state