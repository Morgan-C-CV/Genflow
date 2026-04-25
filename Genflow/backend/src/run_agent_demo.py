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
ALLOWED_DEMO_MODES = {"interactive", "local_live_smoke", "policy_runner_demo"}


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


def resolve_demo_mode(env: dict | None = None) -> str:
    env = env or os.environ
    mode = str(env.get("GENFLOW_DEMO_MODE", "interactive")).strip().lower() or "interactive"
    if mode not in ALLOWED_DEMO_MODES:
        allowed = ", ".join(sorted(ALLOWED_DEMO_MODES))
        raise ValueError(f"Invalid demo mode: {mode}. Allowed modes: {allowed}.")
    return mode


def resolve_policy_runner_max_steps(env: dict | None = None) -> int:
    env = env or os.environ
    raw = str(env.get("GENFLOW_POLICY_RUNNER_MAX_STEPS", "6")).strip() or "6"
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ValueError(f"Invalid GENFLOW_POLICY_RUNNER_MAX_STEPS: {raw}.") from exc


def resolve_live_backend_config(env: dict | None = None):
    from app.agent.live_backend_config import resolve_live_backend_config as _resolve_live_backend_config

    return _resolve_live_backend_config(env=env)


def build_workflow_facade(config):
    if config.backend_kind == "workflow_shell" and config.workflow_profile == "local":
        from app.agent.local_workflow_facade import LocalWorkflowFacade

        return LocalWorkflowFacade()
    return None


def build_workflow_backend_transport(config, workflow_facade=None):
    from app.agent.workflow_backend_transport import WorkflowBackendTransport

    facade = workflow_facade if workflow_facade is not None else build_workflow_facade(config)
    return WorkflowBackendTransport(config, workflow_facade=facade)


def build_live_backend_client(config, workflow_facade=None):
    from app.agent.default_live_backend_client import DefaultLiveBackendClient
    from app.agent.live_backend_errors import LiveBackendNotConfiguredError

    if not config.enabled:
        raise LiveBackendNotConfiguredError("Live backend substrate is not configured.")
    transport = build_workflow_backend_transport(config, workflow_facade=workflow_facade)
    return DefaultLiveBackendClient(transport=transport)


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


def build_live_runtime_components_from_env(env: dict | None = None) -> dict:
    execution_mode = resolve_execution_mode(env=env)
    if execution_mode != "live":
        raise ValueError("Canonical live smoke helper requires GENFLOW_EXECUTION_MODE=live.")

    live_backend_config = resolve_live_backend_config(env=env)
    backend_client = build_live_backend_client(live_backend_config)
    execution_adapter = build_execution_adapter(mode=execution_mode, backend_client=backend_client)
    return {
        "execution_mode": execution_mode,
        "live_backend_config": live_backend_config,
        "backend_client": backend_client,
        "execution_adapter": execution_adapter,
    }


def run_local_live_smoke(env: dict | None = None, schema=None, reference_bundle: dict | None = None) -> dict:
    from app.agent.runtime_models import NormalizedSchema

    components = build_live_runtime_components_from_env(env=env)
    execution_adapter = components["execution_adapter"]
    schema = schema or NormalizedSchema(
        prompt="a cinematic portrait",
        negative_prompt="blurry",
        model="sdxl-base",
        sampler="DPM++ 2M",
        style=["cinematic"],
        lora=["portrait-helper"],
    )
    reference_bundle = reference_bundle or {
        "query_index": 7,
        "counts": {"best": 1},
        "references": [{"id": 101}, {"id": 202}],
    }
    payload, summary = execution_adapter.produce_initial_result(
        schema=schema,
        reference_bundle=reference_bundle,
    )
    return {
        **components,
        "schema": schema,
        "reference_bundle": reference_bundle,
        "result_payload": payload,
        "result_summary": summary,
    }


def run_local_live_multi_stage_smoke(
    env: dict | None = None,
    schema=None,
    reference_bundle: dict | None = None,
    preview_probe=None,
    committed_patch=None,
) -> dict:
    from app.agent.runtime_models import CommittedPatch, NormalizedSchema, PreviewProbe

    components = build_live_runtime_components_from_env(env=env)
    execution_adapter = components["execution_adapter"]
    schema = schema or NormalizedSchema(
        prompt="a cinematic portrait",
        negative_prompt="blurry",
        model="sdxl-base",
        sampler="DPM++ 2M",
        style=["cinematic"],
        lora=["portrait-helper"],
    )
    reference_bundle = reference_bundle or {
        "query_index": 7,
        "counts": {"best": 1},
        "references": [{"id": 101}, {"id": 202}],
    }
    preview_probe = preview_probe or PreviewProbe(
        probe_id="probe-local-001",
        summary="Push style intensity while preserving composition.",
        target_axes=["style"],
        preserve_axes=["composition"],
        preview_execution_spec={"patch_family": "resource_shift"},
        source_kind="schema_variation",
    )
    committed_patch = committed_patch or CommittedPatch(
        patch_id="patch-local-001",
        target_fields=["style", "model"],
        target_axes=["style"],
        preserve_axes=["composition"],
        changes={"style": ["cinematic", "vivid"], "model": "sdxl-base"},
        rationale="Commit the stronger style direction from preview.",
    )

    initial_payload, initial_summary = execution_adapter.produce_initial_result(
        schema=schema,
        reference_bundle=reference_bundle,
    )
    preview_result = execution_adapter.execute_preview_probe(
        schema=schema,
        probe=preview_probe,
    )
    committed_payload, committed_summary = execution_adapter.execute_committed_patch(
        schema=schema,
        patch=committed_patch,
    )

    return {
        **components,
        "schema": schema,
        "reference_bundle": reference_bundle,
        "preview_probe": preview_probe,
        "committed_patch": committed_patch,
        "initial_result": {
            "payload": initial_payload,
            "summary": initial_summary,
        },
        "preview_result": preview_result,
        "committed_result": {
            "payload": committed_payload,
            "summary": committed_summary,
        },
    }


def format_local_live_smoke_summary(smoke: dict) -> str:
    config = smoke["live_backend_config"]
    payload = smoke["result_payload"]
    summary = smoke["result_summary"]
    lines = [
        "GenFlow Local Live Smoke",
        f"execution_mode: {smoke['execution_mode']}",
        f"backend_kind: {config.backend_kind}",
        f"workflow_profile: {config.workflow_profile or 'none'}",
        f"result_id: {payload.result_id}",
        f"result_type: {payload.result_type}",
        f"summary_text: {summary.summary_text}",
    ]
    return "\n".join(lines)


def run_policy_runner_demo(
    env: dict | None = None,
    user_intent: str = "make a portrait",
    gallery_index: int = 7,
    feedback_text: str = "Keep the composition, but improve style.",
) -> dict:
    execution_mode = resolve_execution_mode(env=env)
    backend_client = None
    if execution_mode == "live":
        live_backend_config = resolve_live_backend_config(env=env)
        backend_client = build_live_backend_client(live_backend_config)
    runtime_service = build_runtime_service(execution_mode=execution_mode, backend_client=backend_client)
    session = runtime_service.start_episode(user_intent)
    session = runtime_service.generate_initial_candidates(
        session_id=session.session_id,
        refresh=False,
        per_query_k=2,
        top_k=12,
    )
    session = runtime_service.select_initial_reference(session.session_id, gallery_index)
    session = runtime_service.generate_initial_schema(session.session_id)
    session = runtime_service.produce_initial_result(session.session_id)
    session = runtime_service.submit_feedback(session.session_id, feedback_text)
    max_steps = resolve_policy_runner_max_steps(env=env)
    run_result = runtime_service.run_policy_steps(session.session_id, max_steps=max_steps)
    return {
        "demo_mode": "policy_runner_demo",
        "execution_mode": execution_mode,
        "session_id": run_result.final_session.session_id,
        "max_steps": max_steps,
        "run_result": run_result,
    }


def format_policy_runner_demo_summary(result: dict) -> str:
    run_result = result["run_result"]
    final_session = run_result.final_session
    actions = [step.action for step in run_result.steps]
    lines = [
        "GenFlow Policy Runner Demo",
        f"execution_mode: {result['execution_mode']}",
        f"session_id: {result['session_id']}",
        f"max_steps: {result['max_steps']}",
        f"actions_taken: {', '.join(actions) if actions else 'none'}",
        f"stopped: {run_result.stopped}",
        f"stop_reason: {run_result.stop_reason or 'none'}",
        f"selected_probe: {final_session.selected_probe.probe_id or 'none'}",
        f"accepted_patch: {final_session.accepted_patch.patch_id or 'none'}",
        f"verifier_summary: {final_session.latest_verifier_result.summary or 'none'}",
    ]
    return "\n".join(lines)


def run_demo_mode(env: dict | None = None) -> dict:
    demo_mode = resolve_demo_mode(env=env)
    if demo_mode == "local_live_smoke":
        smoke = run_local_live_smoke(env=env)
        return {
            "demo_mode": demo_mode,
            "summary_text": format_local_live_smoke_summary(smoke),
            "payload": smoke,
        }
    if demo_mode == "policy_runner_demo":
        policy_demo = run_policy_runner_demo(env=env)
        return {
            "demo_mode": demo_mode,
            "summary_text": format_policy_runner_demo_summary(policy_demo),
            "payload": policy_demo,
        }
    return {
        "demo_mode": demo_mode,
        "summary_text": "",
        "payload": None,
    }


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
            "refinement_benchmark_set": session.refinement_benchmark_set,
            "refinement_benchmark_summary": session.refinement_benchmark_summary,
            "benchmark_comparison_summary": session.benchmark_comparison_summary,
            "workflow_id": session.workflow_id,
            "workflow_identity": session.workflow_identity,
            "workflow_state": session.workflow_state,
            "editable_scopes": session.editable_scopes,
            "protected_scopes": session.protected_scopes,
            "last_execution_config": session.last_execution_config,
            "workflow_metadata": session.workflow_metadata,
            "workflow_graph_placeholder": session.workflow_graph_placeholder,
            "workflow_topology_hints": session.workflow_topology_hints,
            "workflow_topology_entry_node_ids": session.workflow_topology_entry_node_ids,
            "workflow_topology_exit_node_ids": session.workflow_topology_exit_node_ids,
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
            "top_schema_patch_candidate": session.top_schema_patch_candidate,
            "preferred_commit_source": session.preferred_commit_source,
            "commit_execution_mode": session.commit_execution_mode,
            "commit_execution_authority": session.commit_execution_authority,
            "commit_execution_implementation_mode": session.commit_execution_implementation_mode,
            "request_backend_execution_mode": (
                "graph_primary_backend_execution"
                if session.commit_execution_implementation_mode == "graph_primary_execution"
                else "schema_compatible_backend_execution"
            ),
            "backend_accepted_execution_mode": session.latest_execution_source_evidence.backend_accepted_execution_mode,
            "backend_realized_execution_mode": session.latest_execution_source_evidence.backend_realized_execution_mode,
            "request_primary_plan_kind": (
                "graph_primary" if session.commit_execution_authority == "graph_authoritative" else "schema_primary"
            ),
            "current_workflow_graph_patch": session.current_workflow_graph_patch,
            "selected_workflow_graph_patch": session.selected_workflow_graph_patch,
            "workflow_graph_patch_candidates": session.workflow_graph_patch_candidates,
            "top_workflow_graph_patch_candidate": session.top_workflow_graph_patch_candidate,
            "selected_graph_native_patch_candidate": session.selected_graph_native_patch_candidate,
            "latest_execution_source_evidence": session.latest_execution_source_evidence,
            "patch_history": session.patch_history,
            "previous_result_summary": session.previous_result_summary,
            "latest_verifier_result": session.latest_verifier_result,
            "latest_verifier_signal_summary": session.latest_verifier_signal_summary,
            "latest_verifier_repair_recommendation": session.latest_verifier_repair_recommendation,
            "continue_recommended": session.continue_recommended,
            "verifier_confidence": session.verifier_confidence,
            "stop_reason": session.stop_reason,
        }
    )


def describe_cli_failure(exc: Exception) -> str:
    from app.agent.live_backend_errors import (
        LiveBackendNotConfiguredError,
        LiveBackendNotImplementedError,
        LiveBackendUnavailableError,
    )

    chain = []
    current = exc
    while current is not None and len(chain) < 4:
        chain.append(f"{current.__class__.__name__}: {current}")
        current = getattr(current, "__cause__", None)
    message = " -> ".join(chain)
    if isinstance(exc, LiveBackendNotConfiguredError):
        return (
            "[CLI ERROR] Live execution mode selected, but no live substrate is configured.\n"
            "Set `GENFLOW_LIVE_BACKEND_KIND` and related live backend env vars, or use `GENFLOW_EXECUTION_MODE=mock`.\n"
            f"Root cause: {message}"
        )
    if isinstance(exc, LiveBackendNotImplementedError):
        return (
            "[CLI ERROR] Live substrate is configured, but dispatch is not implemented yet.\n"
            "Keep using `GENFLOW_EXECUTION_MODE=mock` until the workflow backend transport is wired.\n"
            f"Root cause: {message}"
        )
    if isinstance(exc, LiveBackendUnavailableError):
        return (
            "[CLI ERROR] Live backend substrate is unavailable for this run.\n"
            "Verify the live backend config or use `GENFLOW_EXECUTION_MODE=mock`.\n"
            f"Root cause: {message}"
        )
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
        demo_mode = resolve_demo_mode()
        if demo_mode in {"local_live_smoke", "policy_runner_demo"}:
            result = run_demo_mode()
            print(result["summary_text"])
            return

        execution_mode = resolve_execution_mode()
        backend_client = None
        if execution_mode == "live":
            live_backend_config = resolve_live_backend_config()
            backend_client = build_live_backend_client(live_backend_config)
        print("GenFlow Agent Demo")
        print("Current scope: start -> clarify -> candidates -> select -> reference bundle -> metadata/schema -> initial result -> feedback -> hypotheses -> preview -> commit -> verify")
        print(f"Execution mode: {execution_mode}")

        runtime_service = build_runtime_service(execution_mode=execution_mode, backend_client=backend_client)
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
            if session.selected_probe.probe_id:
                print(f"\n[Default Selected Probe]\n{session.selected_probe.probe_id}")
            preview_probes_path = None
            preview_result_path = None
            if session.preview_probe_candidates:
                session = runtime_service.preview_selected_probe(session.session_id)
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
