from __future__ import annotations

import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# LinkODT local AI Director protocol
# -----------------------------------------------------------------------------
# Qwen never controls VTK directly. It translates natural language into this
# deliberately bounded protocol. Visualization Studio validates every action
# locally before changing the camera/timeline.

ALLOWED_ACTIONS = {
    # Backward-compatible low-level camera verbs
    "orbit_yaw", "orbit_pitch", "zoom", "pan", "roll", "preset",
    "focus_roi", "frame_roi", "focus_point", "hold",
    # Cinematic orbit / rotate verbs
    "orbit_left", "orbit_right", "orbit_up", "orbit_down", "turntable",
    "roll_left", "roll_right",
    # Look / tilt in place (camera position stays fixed)
    "look_left", "look_right", "tilt_up", "tilt_down",
    # Camera translation / lens-like moves
    "push_in", "pull_out", "truck_left", "truck_right",
    "pedestal_up", "pedestal_down", "dolly_forward", "dolly_backward",
    # Targeting and framing
    "zoom_to_roi", "focus_object", "frame_object", "frame_all",
    "look_at_point", "set_camera_position", "move_world",
    # Lens / projection
    "set_fov", "set_projection",
    # Director-friendly semantic macros
    "lay_flat_view", "upright_view", "detail_scan_horizontal", "detail_scan_vertical",
}

ALLOWED_PRESETS = {"", "iso", "front", "back", "left", "right", "top", "bottom"}
ALLOWED_PROJECTIONS = {"", "perspective", "orthographic"}
ALLOWED_TRANSITIONS = {
    "smooth", "linear", "orbit", "hold", "cut", "ease_in", "ease_out", "ease_in_out"
}

ACTION_HELP = r"""
You are translating natural-language directing instructions into LinkODT camera actions.
Use ONLY the actions below. Never invent a verb.

CAMERA ORBIT / ROTATION
- orbit_left: orbit around current focal point to the viewer's left. Use degrees > 0.
- orbit_right: orbit right around current focal point. Use degrees > 0.
- orbit_up: orbit upward around current focal point. Use degrees > 0.
- orbit_down: orbit downward around current focal point. Use degrees > 0.
- orbit_yaw: signed horizontal orbit. degrees < 0 = left; > 0 = right.
- orbit_pitch: signed vertical orbit. degrees > 0 = up; < 0 = down.
- turntable: orbit horizontally around focal point through a large angle; use degrees, normally 180-360.
- roll_left / roll_right: roll around the viewing axis; use degrees > 0.
- roll: signed roll; degrees < 0 = left, > 0 = right.

LOOK / TILT WITHOUT MOVING CAMERA POSITION
- look_left / look_right: yaw the viewing direction in place; use degrees > 0.
- tilt_up / tilt_down: pitch the viewing direction in place; use degrees > 0.

MOVE / ZOOM
- push_in: move visually closer to the current target. factor > 1; default 1.4.
- pull_out: move visually farther away. factor > 1; default 1.4.
- zoom: generic dolly factor. factor > 1 = in; 0 < factor < 1 = out.
- truck_left / truck_right: translate camera + target horizontally in screen plane. amount > 0.
- pedestal_up / pedestal_down: translate camera + target vertically in screen plane. amount > 0.
- pan: generic view-plane translation; dx left/right, dy down/up; moderate values are around -1..1.
- dolly_forward / dolly_backward: translate camera toward/away from focal point by amount scene-radius units.
- move_world: translate both camera and target in world XYZ by dx,dy,dz scene-radius units.

TARGET / ROI / OBJECT
- focus_roi: look at named ROI but keep camera position.
- frame_roi: center and fit named ROI to view.
- zoom_to_roi: focus named ROI, then push in by factor.
- focus_point / look_at_point: look at explicit world x,y,z.
- focus_object: look at the center of a named loaded object.
- frame_object: center and fit a named loaded object.
- frame_all: fit all visible objects.

ABSOLUTE CAMERA / LENS
- preset: iso/front/back/left/right/top/bottom.
- lay_flat_view: make the specimen look flat / lying down / 平躺 / 平放 / 俯视展示. This means a top-down camera view looking along the Z axis.
- upright_view: restore an upright / standing / 正立展示 view. This means a front camera view.
- set_camera_position: place camera at absolute world x,y,z, keeping current target.
- set_fov: perspective field of view in degrees; use fov.
- set_projection: projection = perspective or orthographic.

DIRECTOR DETAIL SCANS
- detail_scan_horizontal: after a close-up, translate the camera+target laterally across the detail. amount > 0. Use when the user says 平移看细节 / 横向扫过 / scan across details.
- detail_scan_vertical: after a close-up, translate vertically across the detail. amount > 0. Use when the user says 上下扫过细节 / vertical detail scan.

NATURAL-LANGUAGE DIRECTOR RULES
- “转一圈 / 绕一圈 / 旋转一周 / full turn / one full revolution” ALWAYS means turntable with degrees=360.
- “半圈 / half turn” means turntable 180. “四分之一圈 / quarter turn” means 90.
- A generic “左转/右转/绕过去/rotate/orbit” with no angle should normally use 90 degrees, not a tiny move.
- “稍微 / slightly” may use 30-45 degrees.
- “平躺 / 平放 / lay flat” means lay_flat_view.
- “拉近后平移看细节 / zoom in then scan the detail” should be a SEQUENCE: focus/frame target -> push_in or zoom_to_roi -> detail_scan_horizontal/vertical.
- For a close-up around a named/selected ROI, focus_roi or zoom_to_roi should happen BEFORE orbit/scan moves so the camera rotates/translates around that detail.

TIMING
- hold: keep current shot unchanged for duration seconds.

Each action MUST have duration and transition.
Transitions: smooth, linear, orbit, hold, cut, ease_in, ease_out, ease_in_out.
Use "orbit" for orbit/turntable actions, "hold" for hold, "cut" only when user explicitly asks for a hard cut.
If the user gives no duration, choose a natural cinematic value, normally 0.6-2.0 s.
If no orbit/tilt angle is given, use 90 degrees for a normal directorial move; use 30-45 only when the user says slightly. If no zoom/push factor is given, use 1.4.
Prefer a sequence of simple actions over one ambiguous action.
If the user says "以选中的点/ROI为中心", "selected point", or equivalent: focus_roi(selected ROI) first, then perform the requested motion.
A 360-degree turntable should be one turntable action; LinkODT will internally subdivide it into smooth intermediate keyframes.
""".strip()

# A flat schema is intentional: llama.cpp JSON grammar is more reliable with a
# compact, predictable object shape. Unused fields are returned as neutral values.
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "actions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 120,
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
                    "duration": {"type": "number"},
                    "transition": {"type": "string", "enum": sorted(ALLOWED_TRANSITIONS)},
                    "degrees": {"type": "number"},
                    "factor": {"type": "number"},
                    "amount": {"type": "number"},
                    "dx": {"type": "number"},
                    "dy": {"type": "number"},
                    "dz": {"type": "number"},
                    "roi": {"type": "string"},
                    "object": {"type": "string"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                    "preset": {"type": "string", "enum": sorted(ALLOWED_PRESETS)},
                    "projection": {"type": "string", "enum": sorted(ALLOWED_PROJECTIONS)},
                    "fov": {"type": "number"},
                    "note": {"type": "string"},
                },
                "required": [
                    "action", "duration", "transition", "degrees", "factor", "amount",
                    "dx", "dy", "dz", "roi", "object", "x", "y", "z",
                    "preset", "projection", "fov", "note"
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "actions"],
    "additionalProperties": False,
}

_MODEL_CACHE: dict[tuple[str, int, int, int], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _roi_text(rois: list[dict[str, Any]], selected_roi: str | None) -> str:
    if not rois:
        return "No 3D ROI is currently defined."
    rows = []
    for r in rois:
        marker = " [SELECTED]" if selected_roi and r.get("name") == selected_roi else ""
        center = r.get("center", [0, 0, 0])
        rows.append(
            f"- {r.get('name','ROI')}{marker}: center=({center[0]:.6g}, {center[1]:.6g}, {center[2]:.6g}), "
            f"radius={float(r.get('radius',0)):.6g}"
        )
    return "\n".join(rows)


def _object_text(objects: list[dict[str, Any]] | None) -> str:
    if not objects:
        return "No loaded object names were supplied."
    return "\n".join(f"- {str(o.get('name', 'object'))}" for o in objects)


def build_system_prompt(
    rois: list[dict[str, Any]],
    selected_roi: str | None,
    camera: dict[str, Any],
    objects: list[dict[str, Any]] | None = None,
) -> str:
    return f"""
You are the LOCAL camera-director translator inside LinkODT Visualization Studio.
The human speaks naturally in Chinese or English. Translate the request into a constrained JSON camera plan.
You never control VTK directly. You only select from LinkODT's supported action vocabulary.
Do not add biological claims. Do not invent ROI names, object names, coordinates, or unsupported actions.
Keep scientific 3D structures readable and camera motion smooth unless the user explicitly requests a dramatic move or cut.

{ACTION_HELP}

AVAILABLE 3D ROIs
{_roi_text(rois, selected_roi)}

LOADED OBJECTS
{_object_text(objects)}

CURRENT CAMERA
position={camera.get('position')}
focal_point={camera.get('focal_point')}
view_up={camera.get('view_up')}
view_angle={camera.get('view_angle')}
parallel_projection={camera.get('parallel_projection')}

Selected ROI name: {selected_roi!r}
If the user refers to "selected ROI", "selected point", "这个点", "选中的点", use that selected ROI when it exists.
If a requested ROI/object is unavailable, do not fabricate it. Use the closest safe camera-only action and say why in note.
Output valid JSON only. /no_think
""".strip()


def _load_llama(model_path: str, n_ctx: int, n_gpu_layers: int, n_threads: int):
    try:
        from llama_cpp import Llama
    except Exception as exc:
        raise RuntimeError(
            "Local Qwen requires llama-cpp-python. Install it in the Visualization environment first. "
            "See INSTALL_LOCAL_QWEN.md."
        ) from exc

    p = Path(model_path).expanduser().resolve()
    if not p.exists() or p.suffix.lower() != ".gguf":
        raise FileNotFoundError(f"GGUF model not found: {p}")

    threads = int(n_threads)
    if threads <= 0:
        threads = max(1, (os.cpu_count() or 4) - 1)
    gpu_layers = int(n_gpu_layers)
    # llama.cpp / llama-cpp-python use -1 for all possible layers.
    key = (str(p), int(n_ctx), gpu_layers, threads)
    with _MODEL_CACHE_LOCK:
        llm = _MODEL_CACHE.get(key)
        if llm is None:
            llm = Llama(
                model_path=str(p),
                n_ctx=max(2048, int(n_ctx)),
                n_gpu_layers=gpu_layers,
                n_threads=threads,
                n_threads_batch=threads,
                verbose=False,
            )
            _MODEL_CACHE.clear()  # keep at most one large GGUF resident
            _MODEL_CACHE[key] = llm
        return llm


def unload_local_model() -> None:
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    # Qwen3 older checkpoints may emit thinking blocks if /no_think is ignored.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    raise RuntimeError(f"Local Qwen did not return valid JSON. First output: {text[:800]}")


def request_local_qwen_plan(
    user_prompt: str,
    rois: list[dict[str, Any]],
    selected_roi: str | None,
    camera: dict[str, Any],
    *,
    model_path: str,
    objects: list[dict[str, Any]] | None = None,
    n_ctx: int = 8192,
    n_gpu_layers: int = -1,
    n_threads: int = 0,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> dict[str, Any]:
    if not str(user_prompt).strip():
        raise ValueError("Describe the shot you want first.")
    if not str(model_path).strip():
        raise ValueError("Choose a local Qwen .gguf model first.")

    llm = _load_llama(model_path, n_ctx, n_gpu_layers, n_threads)
    messages = [
        {"role": "system", "content": build_system_prompt(rois, selected_roi, camera, objects)},
        {"role": "user", "content": str(user_prompt).strip() + "\n/no_think"},
    ]

    kwargs = dict(
        messages=messages,
        temperature=float(max(0.0, min(temperature, 1.5))),
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        max_tokens=max(256, min(int(max_tokens), 8192)),
        repeat_penalty=1.05,
    )

    # llama-cpp-python supports JSON Schema mode. If the installed build is old,
    # fall back to JSON object mode, then validate locally.
    try:
        response = llm.create_chat_completion(
            **kwargs,
            response_format={"type": "json_object", "schema": PLAN_SCHEMA},
        )
    except Exception:
        messages[0]["content"] += "\nReturn a JSON object with root keys title and actions. Return JSON only."
        kwargs2 = dict(kwargs)
        kwargs2["messages"] = messages
        response = llm.create_chat_completion(
            **kwargs2,
            response_format={"type": "json_object"},
        )

    content = response["choices"][0]["message"].get("content", "")
    plan = _extract_json(content)
    return validate_plan(
        plan,
        {str(r.get("name")) for r in rois},
        {str(o.get("name")) for o in (objects or [])},
    )


# Backward-compatible alias for older GUI imports. It is intentionally local.
request_qwen_plan = request_local_qwen_plan


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    return v if math.isfinite(v) else float(default)


def validate_plan(
    plan: dict[str, Any],
    roi_names: set[str] | None = None,
    object_names: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("Camera plan must be a JSON object.")
    actions = plan.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("Camera plan contains no actions.")
    if len(actions) > 120:
        raise ValueError("Camera plan is too long (>120 actions).")

    roi_names = roi_names or set()
    object_names = object_names or set()
    clean = []
    for i, row in enumerate(actions, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Action {i} is not an object.")
        action = str(row.get("action", "")).strip()
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"Unsupported AI Director action: {action!r}")

        duration = max(0.02, min(_finite(row.get("duration"), 1.0), 60.0))
        transition = str(row.get("transition", "smooth") or "smooth").strip().lower()
        if transition not in ALLOWED_TRANSITIONS:
            transition = "smooth"

        item = {
            "action": action,
            "duration": duration,
            "transition": transition,
            "degrees": _finite(row.get("degrees"), 0.0),
            "factor": _finite(row.get("factor"), 0.0),
            "amount": _finite(row.get("amount"), 0.0),
            "dx": _finite(row.get("dx"), 0.0),
            "dy": _finite(row.get("dy"), 0.0),
            "dz": _finite(row.get("dz"), 0.0),
            "roi": str(row.get("roi", "") or "").strip(),
            "object": str(row.get("object", "") or "").strip(),
            "x": _finite(row.get("x"), 0.0),
            "y": _finite(row.get("y"), 0.0),
            "z": _finite(row.get("z"), 0.0),
            "preset": str(row.get("preset", "") or "").strip().lower(),
            "projection": str(row.get("projection", "") or "").strip().lower(),
            "fov": _finite(row.get("fov"), 0.0),
            "note": str(row.get("note", "") or "").strip(),
        }

        if action in {
            "orbit_left", "orbit_right", "orbit_up", "orbit_down", "roll_left", "roll_right",
            "look_left", "look_right", "tilt_up", "tilt_down"
        } and abs(item["degrees"]) < 1e-9:
            item["degrees"] = 90.0
        if action == "turntable" and abs(item["degrees"]) < 1e-9:
            item["degrees"] = 360.0
        if action in {"orbit_yaw", "orbit_pitch", "roll"}:
            item["degrees"] = max(-720.0, min(item["degrees"], 720.0))
        else:
            item["degrees"] = max(-1440.0, min(item["degrees"], 1440.0))

        if action in {"zoom", "push_in", "pull_out", "zoom_to_roi"}:
            if item["factor"] <= 0:
                item["factor"] = 1.4
            item["factor"] = max(0.1, min(item["factor"], 10.0))
        if action in {
            "truck_left", "truck_right", "pedestal_up", "pedestal_down",
            "dolly_forward", "dolly_backward", "detail_scan_horizontal", "detail_scan_vertical"
        } and abs(item["amount"]) < 1e-12:
            item["amount"] = 0.8
        item["amount"] = max(-20.0, min(item["amount"], 20.0))
        item["dx"] = max(-20.0, min(item["dx"], 20.0))
        item["dy"] = max(-20.0, min(item["dy"], 20.0))
        item["dz"] = max(-20.0, min(item["dz"], 20.0))

        if action in {"focus_roi", "frame_roi", "zoom_to_roi"}:
            if not item["roi"] or item["roi"] not in roi_names:
                raise ValueError(f"Action {i} references unknown ROI: {item['roi']!r}")
        if action in {"focus_object", "frame_object"}:
            if not item["object"] or item["object"] not in object_names:
                raise ValueError(f"Action {i} references unknown object: {item['object']!r}")
        if action == "preset" and item["preset"] not in ALLOWED_PRESETS - {""}:
            raise ValueError(f"Action {i} has unsupported preset: {item['preset']!r}")
        if action == "set_projection" and item["projection"] not in ALLOWED_PROJECTIONS - {""}:
            raise ValueError(f"Action {i} has unsupported projection: {item['projection']!r}")
        if action == "set_fov":
            item["fov"] = max(5.0, min(item["fov"] or 30.0, 120.0))

        # Make transition semantics safe and predictable.
        if action == "hold":
            item["transition"] = "hold"
        elif action in {"orbit_left", "orbit_right", "orbit_up", "orbit_down", "orbit_yaw", "orbit_pitch", "turntable"}:
            if item["transition"] not in {"cut", "hold"}:
                item["transition"] = "orbit"
        clean.append(item)

    return {"title": str(plan.get("title", "AI camera plan") or "AI camera plan"), "actions": clean}


def summarize_plan(plan: dict[str, Any]) -> str:
    lines = [str(plan.get("title", "AI camera plan"))]
    elapsed = 0.0
    for i, a in enumerate(plan.get("actions", []), start=1):
        elapsed += float(a.get("duration", 0.0))
        action = a["action"]
        if action in {
            "orbit_yaw", "orbit_pitch", "orbit_left", "orbit_right", "orbit_up", "orbit_down",
            "turntable", "roll", "roll_left", "roll_right", "look_left", "look_right", "tilt_up", "tilt_down"
        }:
            detail = f"{a['degrees']:+g}°"
        elif action in {"zoom", "push_in", "pull_out", "zoom_to_roi"}:
            detail = f"×{a['factor']:.3g}" + (f" @ {a['roi']}" if a.get("roi") else "")
        elif action == "pan":
            detail = f"dx={a['dx']:+.3g}, dy={a['dy']:+.3g}"
        elif action in {"truck_left", "truck_right", "pedestal_up", "pedestal_down", "dolly_forward", "dolly_backward", "detail_scan_horizontal", "detail_scan_vertical"}:
            detail = f"amount={a['amount']:.3g}"
        elif action == "move_world":
            detail = f"d=({a['dx']:+.3g},{a['dy']:+.3g},{a['dz']:+.3g})"
        elif action in {"focus_roi", "frame_roi"}:
            detail = a["roi"]
        elif action in {"focus_object", "frame_object"}:
            detail = a["object"]
        elif action in {"focus_point", "look_at_point", "set_camera_position"}:
            detail = f"({a['x']:.3g}, {a['y']:.3g}, {a['z']:.3g})"
        elif action == "preset":
            detail = a["preset"]
        elif action == "set_projection":
            detail = a["projection"]
        elif action == "set_fov":
            detail = f"{a['fov']:.3g}°"
        else:
            detail = ""
        lines.append(
            f"{i:02d}. {action:<18} {detail:<30} {a['duration']:.2f}s  [{a.get('transition','smooth')}]"
            + (f"  — {a['note']}" if a.get("note") else "")
        )
    lines.append(f"Total duration: {elapsed:.2f} s")
    return "\n".join(lines)
