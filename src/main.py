import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline import run_pipeline
from src.schemas.verification import FinalPipelineOutput

app = FastAPI(
    title="Multi-Agent Hallucination Mitigation Engine",
    description="Research pipeline mitigating hallucination snowballing via semantic clustering and dual guardrails.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str
    interactive_hitl: bool = False


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "hallucination-mitigation-engine",
    }


@app.post("/query", response_model=FinalPipelineOutput)
def execute_query(req: QueryRequest):
    result = run_pipeline(
        raw_query=req.query, interactive_hitl=req.interactive_hitl
    )
    return result


def interactive_cli():
    print("\n" + "=" * 75)
    print("  🚀 MULTI-AGENT HALLUCINATION MITIGATION ENGINE (CLI)")
    print(
        "  Heterogeneous Semantic Clustering & Dual Guardrails Active (D: Drive)"
    )
    print("=" * 75)
    print("Type your questions below. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("User Query > ").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                print("Exiting engine. Goodbye!")
                break

            result = run_pipeline(raw_query=query, interactive_hitl=True)

            print("\n" + "-" * 75)
            print(f"📊 ROUTING DECISION : {result.routing_decision.value}")
            print(f"🎯 CONFIDENCE SCORE  : {result.overall_confidence_score:.3f}")
            print(f"⏱️ EXECUTION TIME    : {result.execution_time_seconds}s")
            print(f"🛡️ HITL TRIGGERED   : {result.hitl_triggered}")
            print(f"🌐 WEB FALLBACK USED: {result.web_fallback_used}")
            print("-" * 75)
            print(f"\n{result.final_answer}\n")
            print("=" * 75 + "\n")

        except KeyboardInterrupt:
            print("\nAborted by user.")
            break
        except Exception as e:
            print(f"\n[Error executing query]: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Hallucination Mitigation Engine"
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Launch as FastAPI web server instead of interactive CLI",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for FastAPI web server"
    )
    args = parser.parse_args()

    if args.server:
        import uvicorn

        print(
            f"Starting FastAPI server on http://localhost:{args.port} (Docs at http://localhost:{args.port}/docs)"
        )
        uvicorn.run("src.main:app", host="0.0.0.0", port=args.port, reload=False)
    else:
        interactive_cli()