# evaluation/ragas_eval.py

'''from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy


def ragas_evaluate(question: str, answer: str, contexts_list: list):
    """
    Evaluate RAG response using RAGAS metrics.

    Metrics used:
    - faithfulness
    - answer_relevancy

    These metrics do NOT require ground-truth/reference answers.
    """

    try:
        # Ensure contexts are strings
        cleaned_contexts = [
            str(context) for context in contexts_list
        ]

        # Create dataset in RAGAS expected format
        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [cleaned_contexts]
        })

        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy
            ]
        )

        # Convert to dictionary
        scores = result.to_pandas().to_dict(orient="records")[0]

        return {
            "faithfulness": round(scores.get("faithfulness", 0), 3),
            "answer_relevancy": round(scores.get("answer_relevancy", 0), 3)
        }

    except Exception as e:
        print(f"RAGAS Evaluation Error: {e}")

        return {
            "faithfulness": 0,
            "answer_relevancy": 0,
            "error": str(e)
        }
    '''


from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from evaluation.ragas_config import ragas_llm, ragas_embeddings


def ragas_evaluate(question: str, answer: str, contexts_list: list):
    """
    Evaluate RAG response using RAGAS metrics.

    Metrics:
    - faithfulness
    - answer_relevancy

    Uses local Ollama (Mistral) via LangchainLLMWrapper.
    """

    try:
        # Ensure contexts are strings
        cleaned_contexts = [str(context) for context in contexts_list]

        # Create dataset in RAGAS format
        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [cleaned_contexts]
        })

        # Run evaluation using local Ollama model
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )

        scores = result.to_pandas().to_dict(orient="records")[0]

        return {
            "faithfulness": round(
                float(scores.get("faithfulness", 0)), 3
            ),
            "answer_relevancy": round(
                float(scores.get("answer_relevancy", 0)), 3
            )
        }

    except Exception as e:
        print(f"RAGAS Evaluation Error: {e}")

        return {
            "faithfulness": 0,
            "answer_relevancy": 0,
            "error": str(e)
        }

        