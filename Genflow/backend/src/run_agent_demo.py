import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List

from app.agent.memory import AgentMemoryService

if TYPE_CHECKING:
    from app.agent.orchestration import AgentOrchestrationService
    from app.agent.runtime_service import AgentRuntimeService
    from app.services.search_service import SearchService


ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "rounds" / "agent_demo"
ALLOWED_EXECUTION_MODES = {"mock", "live"}


def ensure_artifact_dir() -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR


def build_agent_service(memory: AgentMemoryService | None = None) -> "AgentOrchestrationService":
    from app.agent.orchestration import AgentOrchestrationService
    from app.agent.tools import AgentToolsService
    from app.agents.creative_agent import CreativeAgent
    from app.repositories.search_repository import SearchRepository

    search_repo = SearchRepository()
    tools = AgentToolsService(
        creative_agent=CreativeAgent(),
        search_repo=search_repo,
    )
    memory = memory or AgentMemoryService()
    return AgentOrchestrationService(
        tools_service=tools,
        memory_service=memory,
    )


def build_search_service() -> "SearchService":
    from app.repositories.llm_repository import LLMRepository
    from app.repositories.search_repository import SearchRepository
    from app.services.search_service import SearchService

    return SearchService(
        search_repo=SearchRepository(),
        llm_repo=LLMRepository(),
    )


def resolve_execution_mode(env: dict | None = None) -> str:
    env = env or os.environ
    mode = str(env.get("GENFLOW_EXECUTION_MODE", "mock")).strip().lower() or "mock"
    if mode not in ALLOWED_EXECUTION_MODES:
        allowed = ", ".join(sorted(ALLOWED_EXECUTION_MODES))
        raise ValueError(f"Invalid execution mode: {mode}. Allowed modes: {allowed}.")
    return mode


def build_execution_adapter(mode: str = "mock", backend_client=None):
    mode = str(mode).strip().lower()
    if mode not in ALLOWED_EXECUTION_MODES:
        allowed = ", ".join(sorted(ALLOWED_EXECUTION_MODES))
        raise ValueError(f"Invalid execution mode: {mode}. Allowed modes: {allowed}.")
    if mode == "live":
        from app.agent.live_execution_adapter import LiveExecutionAdapter

        return LiveExecutionAdapter(backend_client=backend_client)

    from app.agent.result_executor import ResultExecutor

    return ResultExecutor()


def build_runtime_service(execution_mode: str = "mock", backend_client=None) -> "AgentRuntimeService":
    from app.agent.feedback_parser import FeedbackParser
    from app.agent.patch_planner import PatchPlanner
    from app.agent.probe_generator import PreviewProbeGenerator
    from app.agent.repair_hypothesis import RepairHypothesisBuilder
    from app.agent.runtime_service import AgentRuntimeService
    from app.agent.verifier import Verifier

    memory = AgentMemoryService()
    orchestration = build_agent_service(memory=memory)
    search_service = build_search_service()
    return AgentRuntimeService(
        memory_service=memory,
        orchestration_service=orchestration,
        search_service=search_service,
        execution_adapter=build_execution_adapter(mode=execution_mode, backend_client=backend_client),
        feedback_parser=FeedbackParser(),
        hypothesis_builder=RepairHypothesisBuilder(),
        probe_generator=PreviewProbeGenerator(),
        patch_planner=PatchPlanner(),
        verifier=Verifier(),
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


def to_serializable(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    return value


def build_session_artifact_payload(session) -> dict:
    return to_serializable(
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
            "selected_gallery_index": session.selected_gallery_index,
            "selected_reference_ids": session.selected_reference_ids,
            "current_gallery_anchor_summary": session.current_gallery_anchor_summary,
            "candidate_wall": {
                "groups": session.latest_wall.groups if session.latest_wall else [],
                "flat_indices": session.latest_wall.flat_indices if session.latest_wall else [],
                "query_labels": session.latest_wall.query_labels if session.latest_wall else [],
            },
            "current_schema": session.current_schema,
            "current_result_payload": session.current_result_payload,
            "current_result_summary": session.current_result_summary,
            "parsed_feedback": session.parsed_feedback,
            "preserve_constraints": session.preserve_constraints,
            "dissatisfaction_axes": session.dissatisfaction_axes,
            "requested_changes": session.requested_changes,
            "current_uncertainty_estimate": session.current_uncertainty_estimate,
            "repair_hypotheses": session.repair_hypotheses,
            "preview_probe_candidates": session.preview_probe_candidates,
            "preview_probe_results": session.preview_probe_results,
            "selected_probe": session.selected_probe,
            "accepted_patch": session.accepted_patch,
            "patch_history": session.patch_history,
            "previous_result_summary": session.previous_result_summary,
            "latest_verifier_result": session.latest_verifier_result,
            "continue_recommended": session.continue_recommended,
            "verifier_confidence": session.verifier_confidence,
            "stop_reason": session.stop_reason,
        }
    )


def describe_cli_failure(exc: Exception) -> str:
    chain = []
    current = exc
    while current is not None and len(chain) < 4:
        chain.append(f"{current.__class__.__name__}: {current}")
        current = getattr(current, "__cause__", None)
    message = " -> ".join(chain)
    if "ConnectError" in message or "NameResolutionError" in message:
        return (
            "[CLI ERROR] Live backend unavailable for this smoke run.\n"
            "Use the project virtual environment and a network-enabled environment "
            "when calling the real model-backed cold-start path.\n"
            f"Root cause: {message}"
        )
    if "ModuleNotFoundError" in message:
        return (
            "[CLI ERROR] Missing runtime dependency for the current interpreter.\n"
            "Launch the demo with `.venv/bin/python run_agent_demo.py` and ensure "
            "required packages are installed in that environment.\n"
            f"Root cause: {message}"
        )
    if "NotImplementedError" in message:
        return (
            "[CLI ERROR] Live execution mode is reserved but not wired yet.\n"
            "Set `GENFLOW_EXECUTION_MODE=mock` for the current demo path, or finish "
            "the live backend adapter before using `live`.\n"
            f"Root cause: {message}"
        )
    return f"[CLI ERROR] Demo startup failed.\nRoot cause: {message}"


def main() -> None:
    try:
        execution_mode = resolve_execution_mode()
        print("GenFlow Agent Demo")
        print("Current scope: start -> clarify -> candidates -> select -> reference bundle -> metadata/schema -> initial result -> feedback -> hypotheses -> preview -> commit -> verify")
        print(f"Execution mode: {execution_mode}")

        runtime_service = build_runtime_service(execution_mode=execution_mode)
        df = runtime_service.search_service.search_repo.get_all_data()

        user_intent = input("\n请输入你的创作意图\n> ").strip()
        if not user_intent:
            raise ValueError("创作意图不能为空。")

        session = runtime_service.start_episode(user_intent)
        print_plan(session.plan)

        while session.plan and session.plan.next_action == "ask_user":
            answers = collect_answers(session.plan.clarification_questions)
            session = runtime_service.clarify_episode(session.session_id, answers)
            print_plan(session.plan)
            if not answers:
                break

        session = runtime_service.generate_initial_candidates(
            session_id=session.session_id,
            refresh=False,
            per_query_k=2,
            top_k=12,
        )
        print_wall(session, df)

        gallery_index = choose_gallery_index(session)
        session = runtime_service.select_initial_reference(session.session_id, gallery_index)
        session = runtime_service.generate_initial_schema(session.session_id)
        session = runtime_service.produce_initial_result(session.session_id)

        bundle_path = save_artifact(f"{session.session_id}_reference_bundle.json", session.selected_reference_bundle)
        metadata_path = save_artifact(f"{session.session_id}_metadata.json", session.current_schema_raw)
        schema_path = save_artifact(f"{session.session_id}_normalized_schema.json", to_serializable(session.current_schema))
        initial_result_path = save_artifact(
            f"{session.session_id}_initial_result.json",
            to_serializable(
                {
                    "payload": session.current_result_payload,
                    "summary": session.current_result_summary,
                }
            ),
        )

        print("\n[Selected Gallery Seed]")
        print(f"- gallery_index: {session.selected_gallery_index}")
        print(f"- anchor_summary: {session.current_gallery_anchor_summary}")
        print("\n[Generated Metadata / Schema]")
        print(session.current_schema_raw)
        print("\n[Normalized Schema]")
        print(json.dumps(to_serializable(session.current_schema), ensure_ascii=False, indent=2))
        print("\n[Initial Result]")
        print(json.dumps(to_serializable({
            "payload": session.current_result_payload,
            "summary": session.current_result_summary,
        }), ensure_ascii=False, indent=2))
        feedback_text = input("\n请输入你对当前结果的反馈\n> ").strip()
        if feedback_text:
            session = runtime_service.submit_feedback(session.session_id, feedback_text)
            session = runtime_service.build_repair_hypotheses(session.session_id)
            print("\n[Parsed Feedback]")
            print(json.dumps(to_serializable(session.parsed_feedback), ensure_ascii=False, indent=2))
            print("\n[Repair Hypotheses]")
            print(json.dumps(to_serializable(session.repair_hypotheses), ensure_ascii=False, indent=2))
            session = runtime_service.generate_local_probes(session.session_id)
            print("\n[Preview Probes]")
            print(json.dumps(to_serializable(session.preview_probe_candidates), ensure_ascii=False, indent=2))
            preview_probes_path = None
            preview_result_path = None
            if session.preview_probe_candidates:
                first_probe = session.preview_probe_candidates[0]
                session = runtime_service.preview_probe(session.session_id, first_probe.probe_id)
                session = runtime_service.select_probe(session.session_id, first_probe.probe_id)
                latest_preview = session.preview_probe_results[-1]
                print("\n[Preview Result]")
                print(json.dumps(to_serializable(latest_preview), ensure_ascii=False, indent=2))
                preview_probes_path = save_artifact(
                    f"{session.session_id}_preview_probes.json",
                    to_serializable(session.preview_probe_candidates),
                )
                preview_result_path = save_artifact(
                    f"{session.session_id}_preview_result.json",
                    to_serializable(latest_preview),
                )
                session = runtime_service.commit_patch(session.session_id)
                session = runtime_service.execute_patch(session.session_id)
                session = runtime_service.verify_latest_result(session.session_id)
                continue_decision = runtime_service.should_continue(session.session_id)
                print("\n[Committed Patch]")
                print(json.dumps(to_serializable(session.accepted_patch), ensure_ascii=False, indent=2))
                print("\n[Updated Result]")
                print(json.dumps(to_serializable({
                    "payload": session.current_result_payload,
                    "summary": session.current_result_summary,
                }), ensure_ascii=False, indent=2))
                print("\n[Verifier Decision]")
                print(json.dumps(to_serializable(session.latest_verifier_result), ensure_ascii=False, indent=2))
                print(f"\n[Loop Decision]\ncontinue={continue_decision}")
                committed_patch_path = save_artifact(
                    f"{session.session_id}_committed_patch.json",
                    to_serializable(session.accepted_patch),
                )
                committed_result_path = save_artifact(
                    f"{session.session_id}_updated_result.json",
                    to_serializable(
                        {
                            "payload": session.current_result_payload,
                            "summary": session.current_result_summary,
                        }
                    ),
                )
                verifier_result_path = save_artifact(
                    f"{session.session_id}_verifier_result.json",
                    to_serializable(session.latest_verifier_result),
                )
            save_artifact(f"{session.session_id}_parsed_feedback.json", to_serializable(session.parsed_feedback))
            save_artifact(f"{session.session_id}_repair_hypotheses.json", to_serializable(session.repair_hypotheses))
        session_path = save_artifact(
            f"{session.session_id}_session.json",
            build_session_artifact_payload(session),
        )
        print("\n[Artifacts]")
        print(f"- session: {session_path}")
        print(f"- reference_bundle: {bundle_path}")
        print(f"- metadata: {metadata_path}")
        print(f"- normalized_schema: {schema_path}")
        print(f"- initial_result: {initial_result_path}")
        if feedback_text and preview_probes_path is not None:
            print(f"- preview_probes: {preview_probes_path}")
        if feedback_text and preview_result_path is not None:
            print(f"- preview_result: {preview_result_path}")
        if feedback_text and 'committed_patch_path' in locals():
            print(f"- committed_patch: {committed_patch_path}")
        if feedback_text and 'committed_result_path' in locals():
            print(f"- updated_result: {committed_result_path}")
        if feedback_text and 'verifier_result_path' in locals():
            print(f"- verifier_result: {verifier_result_path}")
    except Exception as exc:
        print(describe_cli_failure(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
