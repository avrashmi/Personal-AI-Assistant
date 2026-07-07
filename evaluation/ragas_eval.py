# evaluation/ragas_eval.py

import ragas.metrics.base as base
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from evaluation.ragas_config import ragas_llm, ragas_embeddings
import traceback


base.DEFAULT_TIMEOUT = 300

def ragas_evaluate(question: str, answer: str, contexts_list: list):
    """
    Evaluate RAG response using RAGAS metrics.

    Metrics:
    - faithfulness
    - answer_relevancy

    Uses local Ollama (Mistral) via LangchainLLMWrapper.
    """

    try:

        cleaned_contexts = [
        str(c)[:500] for c in contexts_list
        ]

        # Ensure contexts are strings
        #cleaned_contexts = [str(context) for context in contexts_list]

        # Create dataset in RAGAS format
        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [cleaned_contexts]
        })

        # IMPORTANT: increase timeout 
        faithfulness.timeout = 300
        answer_relevancy.timeout = 300


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

        #scores = result.to_pandas().to_dict(orient="records")[0]
        df = result.to_pandas()

        scores = df.iloc[0].to_dict()

          # ✅ SAFE extraction (important fix)
        return {
        "answer_relevancy": float(scores.get("answer_relevancy", 0)),
        "faithfulness": float(scores.get("faithfulness", 0))
        }

        '''return {
            faithfulness: round(
            float(scores.get("faithfulness", 0)), 3
           ),
            "answer_relevancy": round(
                float(scores.get("answer_relevancy", 0)), 3
            )
        }'''

    except Exception as e:
        print(f"RAGAS Evaluation Error: {e}")

        print("========== RAGAS ERROR ==========")
        traceback.print_exc()
        print("=================================")


        return {
           # "faithfulness": 0,
            "answer_relevancy": 0,
            "error": str(e)
        }

        