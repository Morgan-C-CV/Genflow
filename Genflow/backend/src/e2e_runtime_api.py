"""End-to-end smoke test for the Genflow runtime API.

Exercises the same path the frontend drives:
  episodes -> (clarify) -> candidates -> select -> schema -> result
  -> workflow -> workflow/push -> gallery image

Run with the backend venv:
    ../.venv/bin/python e2e_runtime_api.py
"""

import json
import sys

import requests

BASE = "http://127.0.0.1:8000/api/v1/runtime"
TIMEOUT = 900

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}{f' :: {detail}' if detail else ''}")
    if not condition:
        failures.append(label)


def post(path, payload=None):
    response = requests.post(f"{BASE}{path}", json=payload or {}, timeout=TIMEOUT)
    return response.status_code, response.json() if response.content else {}


def run_episode(intent, label):
    print(f"\n=== {label} ===")
    print(f"intent: {intent}")

    status, body = post("/episodes", {"user_intent": intent})
    check("POST /episodes -> 200", status == 200, f"HTTP {status}")
    if status != 200:
        print("   detail:", str(body)[:400])
        return
    session_id = body["session"]["session_id"]
    plan = body["plan"]
    print(f"  session: {session_id}")
    print(f"  next_action: {plan['next_action']}")
    print(f"  reasoning: {plan['reasoning_summary'][:150]}")
    check("plan has axes", bool(plan["locked_axes"] or plan["unclear_axes"]))

    # --- clarification dialogue loop (mirrors the frontend) ---------------
    rounds = 0
    while plan["next_action"] == "ask_user" and plan["clarification_questions"] and rounds < 2:
        rounds += 1
        print(f"  clarify round {rounds}: {plan['clarification_questions'][:2]}")
        answers = [f"clarification answer {i + 1}" for i in range(len(plan["clarification_questions"]))]
        status, body = post(f"/episodes/{session_id}/clarify", {"answers": answers})
        check(f"POST /clarify -> 200 (round {rounds})", status == 200, f"HTTP {status}")
        if status != 200:
            print("   detail:", str(body)[:400])
            return
        plan = body["plan"]
        print(f"    next_action -> {plan['next_action']}, closed={body['session']['clarification_closed']}")

    if plan["next_action"] == "ask_user":
        # The agent may keep asking by design; declining is what closes clarification.
        status, body = post(f"/episodes/{session_id}/clarify", {"answers": []})
        check("POST /clarify (decline) -> 200", status == 200, f"HTTP {status}")
        if status != 200:
            print("   detail:", str(body)[:400])
            return
        plan = body["plan"]
        check("session marked clarification_closed", body["session"]["clarification_closed"])
        check("next_action leaves ask_user after declining", plan["next_action"] != "ask_user", plan["next_action"])

    # --- candidates -------------------------------------------------------
    status, body = post(f"/episodes/{session_id}/candidates", {"refresh": False})
    check("POST /candidates -> 200", status == 200, f"HTTP {status}")
    if status != 200:
        print("   detail:", str(body)[:400])
        return
    wall = body["wall"]
    check("wall has candidates", len(wall["candidates"]) > 0, f"{len(wall['candidates'])} candidates")
    check("candidates carry image urls", all(c["image_url"] for c in wall["candidates"]))
    check("wall has query labels", len(wall["query_labels"]) > 0, f"{len(wall['query_labels'])} groups")
    print("  labels:", wall["query_labels"][:3])

    target = wall["candidates"][0]

    # --- gallery image ----------------------------------------------------
    response = requests.get(f"http://127.0.0.1:8000{target['image_url']}", timeout=180)
    check("GET gallery image -> 200", response.status_code == 200, f"HTTP {response.status_code}")
    check(
        "gallery image content-type is an image",
        response.headers.get("content-type", "").startswith("image/"),
        response.headers.get("content-type", ""),
    )
    check("gallery image non-empty", len(response.content) > 1000, f"{len(response.content)} bytes")

    # --- select / schema / result ----------------------------------------
    status, body = post(f"/episodes/{session_id}/select", {"gallery_index": target["gallery_index"]})
    check("POST /select -> 200", status == 200, f"HTTP {status}")
    if status != 200:
        print("   detail:", str(body)[:400])
        return
    check("reference bundle built", len(body["references"]) > 0, f"{len(body['references'])} refs")

    status, body = post(f"/episodes/{session_id}/schema")
    check("POST /schema -> 200", status == 200, f"HTTP {status}")
    if status != 200:
        print("   detail:", str(body)[:400])
        return
    schema = body["normalized"]
    print(f"  model={schema['model']!r} sampler={schema['sampler']!r} steps={schema['steps']} cfg={schema['cfgscale']}")
    check("schema prompt non-empty", bool(schema["prompt"].strip()))
    check("schema model present", bool(schema["model"].strip()))
    check("schema sampler present", bool(schema["sampler"].strip()))

    status, body = post(f"/episodes/{session_id}/result")
    check("POST /result -> 200", status == 200, f"HTTP {status}")
    if status == 200:
        check("result payload produced", bool(body["payload"].get("result_id")))

    # --- workflow ---------------------------------------------------------
    options = {"width": 1024, "height": 1024, "batch_size": 1, "seed": None, "filename_prefix": "Genflow"}
    status, body = post(f"/episodes/{session_id}/workflow", options)
    check("POST /workflow -> 200", status == 200, f"HTTP {status}")
    if status != 200:
        print("   detail:", str(body)[:400])
        return

    check(
        "build reports structured remediation",
        isinstance(body.get("remediation"), list)
        and any(item["kind"] == "checkpoint" for item in body["remediation"]),
        str([item["kind"] for item in body.get("remediation", [])]),
    )
    for item in body.get("remediation", [])[:3]:
        print(f"  remediation[{item['kind']}]: {item['message'][:110]}")

    api_graph = body["api_graph"]
    ui_workflow = body["ui_workflow"]
    print(f"  api nodes: {len(api_graph)} | ui nodes: {len(ui_workflow.get('nodes', []))}")
    print(f"  controls: {json.dumps(body['controls'])}")
    for warning in body["warnings"]:
        print(f"  warning: {warning[:120]}")

    check("api graph is non-empty", len(api_graph) >= 7)
    check("api graph has KSampler", any(n["class_type"] == "KSampler" for n in api_graph.values()))
    check("api graph has CheckpointLoaderSimple", any(n["class_type"] == "CheckpointLoaderSimple" for n in api_graph.values()))
    check("api graph has SaveImage", any(n["class_type"] == "SaveImage" for n in api_graph.values()))
    check("CLIPTextEncode text has no <lora: tags", not any("<lora:" in str(n["inputs"].get("text", "")) for n in api_graph.values()))

    # Every API link must point at an existing node + valid slot.
    link_ok = True
    for node in api_graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                if str(value[0]) not in api_graph:
                    link_ok = False
    check("api graph link targets exist", link_ok)

    if ui_workflow:
        nodes = ui_workflow["nodes"]
        links = ui_workflow["links"]
        ui_link_ids = {link[0] for link in links}
        out_links = [lid for node in nodes for slot in node["outputs"] for lid in slot["links"]]
        in_links = [slot["link"] for node in nodes for slot in node["inputs"] if slot["link"] is not None]
        check("ui workflow links are consistent", sorted(out_links) == sorted(in_links) == sorted(ui_link_ids))
        check("ui workflow every input linked", all(slot["link"] is not None for node in nodes for slot in node["inputs"]))
        check("ui workflow widgets sized", all(len(node["widgets_values"]) > 0 for node in nodes if node["type"] in {"KSampler", "CheckpointLoaderSimple", "CLIPTextEncode"}))
        ksampler = next((n for n in nodes if n["type"] == "KSampler"), None)
        if ksampler:
            check("ui KSampler widgets include control_after_generate", ksampler["widgets_values"][1] == "randomize", str(ksampler["widgets_values"]))

    # --- push -------------------------------------------------------------
    status, body = post(f"/episodes/{session_id}/workflow/push", options)
    check("POST /workflow/push -> 200", status == 200, f"HTTP {status}")
    if status == 200:
        if body["checkpoint_resolved"]:
            check("push accepted by ComfyUI", body["pushed"], body.get("error", "")[:200])
            if body["pushed"]:
                print(f"  prompt_id: {body['prompt_id']}")
                result = requests.get(f"{BASE}/workflow/result/{body['prompt_id']}", timeout=120).json()
                print(f"  result status: {result.get('status')} images: {len(result['images'])}")
        else:
            check("push blocked with actionable message when no checkpoint", bool(body["error"]), body["error"][:160])
            check("push did not claim success", body["pushed"] is False)
            check(
                "error names the offending node and input",
                "ckpt_name" in body["error"] and "CheckpointLoaderSimple" in body["error"],
                body["error"][:140],
            )
            check(
                "error is not the opaque ComfyUI wrapper",
                "prompt_outputs_failed_validation" not in body["error"],
            )
            check(
                "failure remediation stays actionable",
                any(item["kind"] == "checkpoint" for item in body.get("remediation", [])),
                str([item["kind"] for item in body.get("remediation", [])]),
            )


if __name__ == "__main__":
    # A vague intent exercises the clarification dialogue path.
    run_episode("画点什么好看的", "VAGUE INTENT (expect clarification)")
    # A specific intent should skip clarification.
    run_episode(
        "两个巨大的歼星舰在深空相撞，爆炸与碎片，电影级打光，超写实",
        "SPECIFIC INTENT",
    )

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for item in failures:
            print("  -", item)
        sys.exit(1)
    print("ALL CHECKS PASSED")
