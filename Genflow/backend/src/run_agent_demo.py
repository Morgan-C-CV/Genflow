import json
from pathlib import Path
from typing import List

from app.agent.orchestration import AgentOrchestrationService
from app.agent.tools import AgentToolsService
from app.agent.memory import AgentMemoryService
from app.agents.creative_agent import CreativeAgent
from app.repositories.llm_repository import LLMRepository
from app.repositories.search_repository import SearchRepository
from app.services.search_service import SearchService


ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "rounds" / "agent_demo"


def ensure_artifact_dir() -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR


def build_agent_service() -> AgentOrchestrationService:
    search_repo = SearchRepository()
    tools = AgentToolsService(
        creative_agent=CreativeAgent(),
        search_repo=search_repo,
    )
    memory = AgentMemoryService()
    return AgentOrchestrationService(
        tools_service=tools,
        memory_service=memory,
    )


def build_search_service() -> SearchService:
    return SearchService(
        search_repo=SearchRepository(),
        llm_repo=LLMRepository(),
    )


def print_plan(plan) -> None:
    print("\n[Plan]")
    print(f"- fixed_constraints: {plan.fixed_constraints}")
    print(f"- free_variables: {plan.free_variables}")
    print(f"- unclear_axes: {plan.unclear_axes}")
    print(f"- next_action: {plan.next_action}")
    print(f"- reasoning: {plan.reasoning_summary}")
    if plan.clarification_questions:
        print("- clarification_questions:")
        for question in plan.clarification_questions[:3]:
            print(f"  - {question}")


def collect_answers(questions: List[str]) -> List[str]:
    answers: List[str] = []
    for question in questions[:3]:
        answer = input(f"{question}\n> ").strip()
        answers.append(answer)
        if not answer:
            break
    return answers


def print_wall(session, df) -> None:
    if session.latest_wall is None:
        print("\n[Wall] empty")
        return

    print("\n[Candidate Wall]")
    print(session.plan.reasoning_summary if session.plan else "")
    for group_idx, group in enumerate(session.latest_wall.groups, start=1):
        label = session.latest_wall.query_labels[group_idx - 1]
        print(f"\nGroup {group_idx}: {label}")
        for item_idx, df_idx in enumerate(group, start=1):
            row = df.iloc[df_idx]
            global_slot = (group_idx - 1) * max(1, len(group)) + item_idx
            prompt = str(row.get("prompt", "")).replace("\n", " ")
            prompt = prompt[:140] + ("..." if len(prompt) > 140 else "")
            print(
                f"  [{global_slot:02d}] gallery_index={df_idx} "
                f"id={row.get('id', 'N/A')} model={row.get('model', 'N/A')} "
                f"sampler={row.get('sampler', 'N/A')}"
            )
            print(f"       {prompt}")


def choose_gallery_index(session) -> int:
    if session.latest_wall is None or not session.latest_wall.flat_indices:
        raise RuntimeError("Candidate wall is empty.")

    flat_indices = session.latest_wall.flat_indices
    while True:
        raw = input("\n请选择一个候选编号，用于生成 reference bundle 和 metadata/schema\n> ").strip()
        if not raw.isdigit():
            print("请输入数字编号。")
            continue
        choice = int(raw)
        if 1 <= choice <= len(flat_indices):
            return int(flat_indices[choice - 1])
        print(f"请输入 1 到 {len(flat_indices)} 之间的编号。")


def save_artifact(name: str, payload: dict | str) -> Path:
    path = ensure_artifact_dir() / name
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    print("GenFlow Agent Demo")
    print("Current scope: start -> clarify -> candidates -> select -> reference bundle -> metadata/schema")

    agent_service = build_agent_service()
    search_service = build_search_service()
    df = search_service.search_repo.get_all_data()

    user_intent = input("\n请输入你的创作意图\n> ").strip()
    if not user_intent:
        raise ValueError("创作意图不能为空。")

    session = agent_service.start_session(user_intent)
    print_plan(session.plan)

    while session.plan and session.plan.next_action == "ask_user":
        answers = collect_answers(session.plan.clarification_questions)
        session = agent_service.submit_clarification(session.session_id, answers)
        print_plan(session.plan)
        if not answers:
            break

    session = agent_service.generate_candidates(
        session_id=session.session_id,
        refresh=False,
        per_query_k=2,
        top_k=12,
    )
    print_wall(session, df)

    gallery_index = choose_gallery_index(session)
    reference_bundle = search_service.build_diverse_reference_bundle(gallery_index)
    metadata_json = search_service.generate_image_metadata(
        reference_bundle=reference_bundle,
        user_intent=session.clarified_intent,
    )

    session_path = save_artifact(
        f"{session.session_id}_session.json",
        {
            "session_id": session.session_id,
            "original_intent": session.original_intent,
            "clarified_intent": session.clarified_intent,
            "plan": {
                "fixed_constraints": session.plan.fixed_constraints if session.plan else {},
                "free_variables": session.plan.free_variables if session.plan else [],
                "locked_axes": session.plan.locked_axes if session.plan else [],
                "unclear_axes": session.plan.unclear_axes if session.plan else [],
                "next_action": session.plan.next_action if session.plan else "",
                "reasoning_summary": session.plan.reasoning_summary if session.plan else "",
            },
            "selected_gallery_index": gallery_index,
            "candidate_wall": {
                "groups": session.latest_wall.groups if session.latest_wall else [],
                "flat_indices": session.latest_wall.flat_indices if session.latest_wall else [],
                "query_labels": session.latest_wall.query_labels if session.latest_wall else [],
            },
        },
    )
    bundle_path = save_artifact(f"{session.session_id}_reference_bundle.json", reference_bundle)
    metadata_path = save_artifact(f"{session.session_id}_metadata.json", metadata_json)

    print("\n[Selected Gallery Seed]")
    print(f"- gallery_index: {gallery_index}")
    print("\n[Generated Metadata / Schema]")
    print(metadata_json)
    print("\n[Artifacts]")
    print(f"- session: {session_path}")
    print(f"- reference_bundle: {bundle_path}")
    print(f"- metadata: {metadata_path}")


if __name__ == "__main__":
    main()
