"""Voice stack plugin — local wake-word → STT → LLM → TTS → HA media_player.

P1: Real engine implementations replacing P0 stubs.

Registered tools:
- voice_status   — show engine availability and pipeline state
- voice_enable   — enable continuous voice mode with HA media_player
- voice_disable  — disable voice mode
- voice_speak    — TTS-only: speak text through the configured TTS engine
- voice_listen   — one-shot: listen for a command, transcribe, and return
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy engine import (engines have heavy deps — only load when needed)
# ---------------------------------------------------------------------------

_pipeline: Optional[Any] = None  # VoicePipeline instance
_pipeline_lock = threading.Lock()
_voice_ready = threading.Event()

_wake_word_engine: Optional[Any] = None
_stt_engine: Optional[Any] = None
_tts_engine: Optional[Any] = None


def _get_config() -> Dict[str, Any]:
    """Load voice stack config from plugin.yaml or env."""
    return {
        "wake_word": {
            "engine": os.getenv("HERMES_WAKE_WORD_ENGINE", "porcupine"),
            "keyword": os.getenv("HERMES_WAKE_WORD", "computer"),
        },
        "stt": {
            "engine": os.getenv("HERMES_STT_ENGINE", "faster-whisper"),
            "model_size": os.getenv("HERMES_STT_MODEL", "tiny"),
        },
        "tts": {
            "engine": os.getenv("HERMES_TTS_ENGINE", "edge"),
            "voice": os.getenv("HERMES_TTS_VOICE", "en-US-AriaNeural"),
        },
        "media_player_entity": os.getenv("HERMES_MEDIA_PLAYER", ""),
        "max_record_duration": float(os.getenv("HERMES_RECORD_DURATION", "10")),
        "silence_timeout": float(os.getenv("HERMES_SILENCE_TIMEOUT", "2.0")),
        "confidence_threshold": float(os.getenv("HERMES_STT_CONFIDENCE", "0.70")),
    }


def _init_engines() -> bool:
    """Initialise TTS + STT engines from config. Returns True if both ready."""
    global _tts_engine, _stt_engine, _wake_word_engine
    config = _get_config()

    # TTS
    from .engines.tts import create_tts_engine
    try:
        _tts_engine = create_tts_engine(
            engine_type=config["tts"]["engine"],
            voice=config["tts"]["voice"],
        )
    except Exception as exc:
        logger.warning("TTS engine init failed: %s", exc)
        _tts_engine = None

    # STT
    from .engines.stt import create_stt_engine
    try:
        _stt_engine = create_stt_engine(
            engine_type=config["stt"]["engine"],
            model_size=config["stt"].get("model_size", "tiny"),
        )
    except Exception as exc:
        logger.warning("STT engine init failed: %s", exc)
        _stt_engine = None

    # Wake Word (optional — voice mode works without it via voice_listen)
    from .engines.wake_word import create_wake_word_engine
    try:
        _wake_word_engine = create_wake_word_engine(
            engine_type=config["wake_word"]["engine"],
            keywords=[config["wake_word"]["keyword"]],
        )
    except Exception as exc:
        logger.warning("Wake word engine init failed: %s", exc)
        _wake_word_engine = None

    tts_ok = _tts_engine is not None and _tts_engine.available()
    stt_ok = _stt_engine is not None and _stt_engine.available()
    logger.info("Voice engines: TTS=%s STT=%s WakeWord=%s", tts_ok, stt_ok, _wake_word_engine is not None)

    if tts_ok and stt_ok:
        _voice_ready.set()
    return tts_ok and stt_ok


def _check_voice_available() -> bool:
    """Check if voice engines are available (check_fn for tools)."""
    return _voice_ready.is_set()


def _ensure_voice_ready() -> bool:
    """Lazy-init engines on first tool call."""
    if not _voice_ready.is_set():
        return _init_engines()
    return True


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _handle_voice_status(args: dict, **kw) -> str:
    """Show Voice Stack engine states and pipeline status."""
    _ensure_voice_ready()

    def _engine_status(engine, name: str) -> Dict[str, Any]:
        if engine is None:
            return {"status": "not_configured"}
        try:
            avail = engine.available()
        except Exception:
            avail = False
        return {"status": "ready" if avail else "unavailable"}

    engines = {
        "wake_word": _engine_status(_wake_word_engine, "wake_word"),
        "stt": _engine_status(_stt_engine, "stt"),
        "tts": _engine_status(_tts_engine, "tts"),
    }

    # Get voice list for TTS
    tts_voices: list = []
    if _tts_engine and _tts_engine.available():
        try:
            tts_voices = _tts_engine.list_voices()
        except Exception:
            pass

    pipeline_state: Dict[str, Any] = {}
    if _pipeline:
        pipeline_state = _pipeline.state.to_dict()

    result: Dict[str, Any] = {
        "engines": engines,
        "tts_voices": tts_voices[:10],  # Limit to 10
        "pipeline": pipeline_state,
        "ready": _voice_ready.is_set(),
    }
    return json.dumps(result, default=str)


def _handle_voice_enable(args: dict, **kw) -> str:
    """Enable continuous voice mode with wake word and HA media_player."""
    global _pipeline

    _ensure_voice_ready()
    if not _voice_ready.is_set():
        return json.dumps({
            "ok": False,
            "error": "Voice engines not available. Check voice_status for details.",
        })

    with _pipeline_lock:
        if _pipeline and _pipeline.state.enabled:
            return json.dumps({"ok": True, "message": "Voice mode already enabled."})

        config = _get_config()
        media_player = args.get("media_player_entity") or config["media_player_entity"] or None

        from .pipeline import VoicePipeline

        # Define the callback that sends user text to Hermes
        def _voice_callback(text: str) -> str:
            """Called when STT produces text. This is where Hermes processes it."""
            logger.info("Voice callback received: %s", text)
            try:
                from tools.homeassistant_tool import _async_list_entities, _run_async
                _run_async(_async_list_entities())
            except Exception:
                pass
            return (
                f"I heard: {text}. "
                "Voice processing is active — Hermes is listening."
            )

        _pipeline = VoicePipeline(
            callback=_voice_callback,
            wake_word_engine=_wake_word_engine,
            stt_engine=_stt_engine,
            tts_engine=_tts_engine,
            media_player_entity=media_player,
            max_record_duration=config["max_record_duration"],
            silence_timeout=config["silence_timeout"],
            confidence_threshold=config["confidence_threshold"],
        )

        if not _pipeline.available:
            _pipeline = None
            return json.dumps({"ok": False, "error": "Engines not all available."})

        started = _pipeline.start()
        if not started:
            _pipeline = None
            return json.dumps({"ok": False, "error": "Voice pipeline failed to start."})
    return json.dumps({"ok": True, "message": "Voice mode enabled. Wake word active."})



def _handle_voice_disable(args: dict, **kw) -> str:
    """Disable voice mode."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline:
            _pipeline.stop()
            _pipeline = None
            return json.dumps({"ok": True, "message": "Voice mode disabled."})
    return json.dumps({"ok": True, "message": "Voice mode was not active."})


def _handle_voice_speak(args: dict, **kw) -> str:
    """Speak text through TTS engine + playback."""
    _ensure_voice_ready()
    if not _tts_engine or not _tts_engine.available():
        return json.dumps({"ok": False, "error": "TTS engine not available."})

    text = args.get("text", "")
    if not text:
        return json.dumps({"ok": False, "error": "No text provided."})

    try:
        from .pipeline import play_audio_local, play_audio_ha

        audio_path = _tts_engine.synthesize(text)
        media_player = args.get("media_player_entity") or _get_config().get("media_player_entity", "")
        if media_player:
            ok = play_audio_ha(audio_path, media_player)
        else:
            ok = play_audio_local(audio_path)

        return json.dumps({"ok": ok, "audio_path": audio_path})
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)})


def _handle_voice_listen(args: dict, **kw) -> str:
    """One-shot: record audio, transcribe, and return text.

    This is a simpler alternative to continuous voice mode.
    Useful for testing STT or for push-to-talk workflows.
    """
    _ensure_voice_ready()
    if not _stt_engine or not _stt_engine.available():
        return json.dumps({"ok": False, "error": "STT engine not available.", "error_category": "engine_unavailable"})

    config = _get_config()
    try:
        duration = float(args.get("duration", config["max_record_duration"]))
    except (TypeError, ValueError):
        return json.dumps({"ok": False, "error": "duration must be numeric", "error_category": "invalid_duration"})
    if duration <= 0 or duration > 60:
        return json.dumps({"ok": False, "error": "duration must be between 0 and 60 seconds", "error_category": "invalid_duration"})
    language = args.get("language", None)

    import tempfile
    from .pipeline import record_audio

    cache_dir = Path.home() / ".hermes" / "voice_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return json.dumps({"ok": False, "error": str(exc), "error_category": "cache_unavailable"})

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(cache_dir)) as tmp:
        audio_path = tmp.name

    try:
        recorded = record_audio(audio_path, duration=duration)
        if not recorded:
            return json.dumps({"ok": False, "error": "No speech detected.", "error_category": "no_speech"})
        try:
            result = _stt_engine.transcribe_with_confidence(audio_path, language=language)
        except Exception as exc:
            return json.dumps({"ok": False, "error": str(exc), "error_category": "transcription_failed"})
        return json.dumps({
            "ok": True,
            "text": result.get("text", ""),
            "confidence": round(result.get("confidence", 1.0), 3),
            "language": result.get("language", "unknown"),
        })
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc), "error_category": "recording_failed"})
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass


def _handle_voice_prompt(args: dict, **kw) -> str:
    """Return the voice-optimised system prompt with current HA context."""
    from .pipeline import build_voice_system_prompt

    areas = None
    entities = None
    try:
        from tools.homeassistant_tool import _async_list_entities, _run_async
        res = _run_async(_async_list_entities())
        entities = res.get("entities", [])[:30]
    except Exception:
        pass

    prompt = build_voice_system_prompt(areas=areas, entities=entities)
    return json.dumps({"ok": True, "prompt": prompt})


# Cache of recent conversation turns keyed by conversation_id: list of messages
_CONVERSATION_HISTORIES: Dict[str, List[Dict[str, Any]]] = {}
_CONVERSATION_HISTORY_MAX_TURNS = 10  # retain last 10 messages (5 round trips)
_CONVERSATION_HISTORY_TTL_SECONDS = 300  # expire history after 5 minutes of inactivity
_CONVERSATION_LAST_ACTIVITY: Dict[str, float] = {}


def _get_history(conversation_id: Optional[str]) -> List[Dict[str, Any]]:
    if not conversation_id:
        return []
    now = time.monotonic()
    last = _CONVERSATION_LAST_ACTIVITY.get(conversation_id, 0.0)
    if (now - last) > _CONVERSATION_HISTORY_TTL_SECONDS:
        _CONVERSATION_HISTORIES.pop(conversation_id, None)
        _CONVERSATION_LAST_ACTIVITY.pop(conversation_id, None)
        return []
    return list(_CONVERSATION_HISTORIES.get(conversation_id, []))


def _append_history(conversation_id: Optional[str], user_text: str, assistant_text: str) -> None:
    if not conversation_id or not user_text or not assistant_text:
        return
    now = time.monotonic()
    history = _CONVERSATION_HISTORIES.setdefault(conversation_id, [])
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_text})
    # Keep bounded
    if len(history) > _CONVERSATION_HISTORY_MAX_TURNS:
        history = history[-_CONVERSATION_HISTORY_MAX_TURNS:]
        _CONVERSATION_HISTORIES[conversation_id] = history
    _CONVERSATION_LAST_ACTIVITY[conversation_id] = now


# HA's conversation agent gives up after 30 s (see custom_components/hermes/__init__.py
# in hermes-voice-ha-integration). Keep the whole LLM loop under that so HA always gets
# a typed assist_response instead of "Hermes is not responding".
_ASSIST_DEADLINE_SECONDS = float(os.getenv("HERMES_ASSIST_DEADLINE", "26"))
_ASSIST_LLM_RETRIES = 1  # one retry for transient errors (e.g. a malformed tool call the server rejects)
# Hermes toolsets offered to Assist queries besides the HA tools (comma-separated; "" = none), e.g. MCP servers'
# runtime toolsets "mcp-<server>". Default: the qBArm robot arm (mcp_servers.qbarm in config.yaml).
_ASSIST_EXTRA_TOOLSETS = [t.strip() for t in os.getenv("HERMES_ASSIST_EXTRA_TOOLSETS", "mcp-qbarm").split(",")
                          if t.strip()]
# the qBArm tools that take `wait` (used only if a call still comes through Hermes's tool_call bridge)
_QBARM_WAIT_TOOLS = {"pick", "place", "hand_over", "take_from_hand", "go_home", "go_to", "start_robot", "stop_robot"}
# MCP utility wrappers left out of the voice tool list (the model only needs the server's own tools)
_ASSIST_SKIP_SUFFIXES = ("_list_resources", "_read_resource", "_list_prompts", "_get_prompt")


def _assist_extra_tools() -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    """The extra toolsets' tools for Assist: (schemas, dispatch, names). Calls go through Hermes's own dispatcher
    (hooks, approvals) in a worker thread; a tool with a `wait` parameter gets wait=false, so a long action (a robot
    arm's pick) answers at once instead of blowing HA's deadline."""
    if not _ASSIST_EXTRA_TOOLSETS:
        return [], {}, set()
    try:
        from model_tools import get_tool_definitions, handle_function_call
        # the tools themselves, not Hermes's tool-search bridge (tool_search / tool_describe / tool_call): the
        # bridge costs voice extra model rounds against HA's deadline and hides the tools' parameters
        defs = get_tool_definitions(enabled_toolsets=_ASSIST_EXTRA_TOOLSETS, quiet_mode=True,
                                    skip_tool_search_assembly=True)
    except Exception as exc:
        logger.warning("Assist: could not load toolsets %s: %s", _ASSIST_EXTRA_TOOLSETS, exc)
        return [], {}, set()
    schemas, dispatch, names = [], {}, set()
    for d in defs:
        fn = d.get("function", {})
        name = fn.get("name", "")
        if not name or name.endswith(_ASSIST_SKIP_SUFFIXES):
            continue
        schemas.append(d)
        names.add(name)
        has_wait = "wait" in ((fn.get("parameters") or {}).get("properties") or {})

        def call(args, _name=name, _has_wait=has_wait):
            args = dict(args or {})
            if _has_wait:
                args["wait"] = False
            elif _name == "tool_call" and isinstance(args.get("arguments"), dict):   # (bridge, just in case)
                args["arguments"] = {**args["arguments"], "wait": False} if args.get("name", "").startswith(
                    "mcp__qbarm__") and args.get("name", "").split("__")[-1] in _QBARM_WAIT_TOOLS else args["arguments"]
            logger.info("Assist tool call %s %s", _name, args)
            return asyncio.to_thread(handle_function_call, _name, args, task_id="assist-query")
        dispatch[name] = call
    logger.info("Assist extra tools from %s: %s", _ASSIST_EXTRA_TOOLSETS,
                ", ".join(f"{n}{'(wait)' if 'wait' in ((d.get('function', {}).get('parameters') or {}).get('properties') or {}) else ''}"
                          for n, d in zip([d.get('function', {}).get('name') for d in schemas], schemas)))
    return schemas, dispatch, names

_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_STRAY_THINK_RE = re.compile(r"</?think>")


def _clean_spoken_text(text: str) -> str:
    """Strip reasoning tags a local model may leak into the spoken reply."""
    text = _THINK_TAG_RE.sub("", text or "")
    text = _STRAY_THINK_RE.sub("", text)
    return text.strip()


# Follow-up listening ("continue conversation"): when the reply needs an answer, HA
# tells the satellite to reopen the mic after TTS without a wake word.
#   HERMES_VOICE_CONTINUE = auto   -> [LISTEN] marker OR reply ending in "?"   (default)
#                           marker -> only the explicit [LISTEN] marker
#                           off    -> never
_CONTINUE_MODE = os.getenv("HERMES_VOICE_CONTINUE", "auto").strip().lower()
_LISTEN_MARKER_RE = re.compile(r"\s*\[\s*LISTEN\s*\]\s*", re.IGNORECASE)


def _split_continue_marker(text: str) -> tuple[str, bool]:
    """Return (spoken_text_without_marker, continue_conversation)."""
    text = text or ""
    has_marker = bool(_LISTEN_MARKER_RE.search(text))
    spoken = _LISTEN_MARKER_RE.sub(" ", text).strip()
    if _CONTINUE_MODE == "off":
        return spoken, False
    if has_marker:
        return spoken, True
    if _CONTINUE_MODE == "auto":
        return spoken, spoken.rstrip().endswith("?")
    return spoken, False


async def _handle_assist_query_with_llm(ctx: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Run the LLM tool loop under HA's response deadline."""
    loop_task = asyncio.ensure_future(_assist_query_llm_loop(ctx, payload))
    try:
        return await asyncio.wait_for(loop_task, timeout=_ASSIST_DEADLINE_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("Assist query exceeded %.0fs deadline; returning timeout reply", _ASSIST_DEADLINE_SECONDS)
        return {
            "ok": False,
            "text": "Sorry, that took too long. Please try again.",
            "conversation_id": payload.get("conversation_id"),
            "error": "assist_query deadline exceeded",
        }


async def _assist_query_llm_loop(ctx: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Turn an HA Assist query into a Hermes LLM response with HA tool access.

    This handles the HA-side ``assist_query`` message introduced by the
    conversation platform. It equips Home Assistant tools and contextual
    entity state so Assist voice queries can inspect and control devices.
    """
    text = str(payload.get("text") or "").strip()
    language = str(payload.get("language") or "en")
    conversation_id = payload.get("conversation_id")

    # 1. Gather context & build system prompt
    areas = None
    entities = None
    try:
        from tools.homeassistant_tool import _async_list_entities, _async_get_state, _async_call_service, _async_list_services, _get_config as _get_ha_config
        url, token = _get_ha_config()
        if token:
            list_res = await _async_list_entities()
            entities = list_res.get("entities", [])[:60]
    except Exception as exc:
        logger.debug("Failed to prefetch entities for assist query prompt: %s", exc)

    try:
        from .pipeline import build_voice_system_prompt
        system_prompt = build_voice_system_prompt(areas=areas, entities=entities)
    except Exception:
        system_prompt = (
            "You are Hermes, responding through Home Assistant Assist. "
            "Reply naturally and concisely for text-to-speech. "
            "Use available Home Assistant tools to inspect or control devices."
        )

    # 2. Prepare tools from tools.homeassistant_tool (uses scoped secrets)
    ha_tools = []
    try:
        from tools.homeassistant_tool import (
            HA_LIST_ENTITIES_SCHEMA,
            HA_GET_STATE_SCHEMA,
            HA_CALL_SERVICE_SCHEMA,
            HA_LIST_SERVICES_SCHEMA,
            _async_list_entities,
            _async_get_state,
            _async_call_service,
            _async_list_services,
        )
        ha_tools = [
            {"type": "function", "function": HA_LIST_ENTITIES_SCHEMA},
            {"type": "function", "function": HA_GET_STATE_SCHEMA},
            {"type": "function", "function": HA_CALL_SERVICE_SCHEMA},
            {"type": "function", "function": HA_LIST_SERVICES_SCHEMA},
        ]
        tool_dispatch = {
            "ha_list_entities": lambda a: _async_list_entities(domain=a.get("domain"), area=a.get("area")),
            "ha_get_state": lambda a: _async_get_state(entity_id=a.get("entity_id", "")),
            "ha_call_service": lambda a: _async_call_service(
                domain=a.get("domain", ""),
                service=a.get("service", ""),
                entity_id=a.get("entity_id"),
                data=json.loads(a["data"]) if isinstance(a.get("data"), str) and a["data"].strip() else a.get("data"),
            ),
            "ha_list_services": lambda a: _async_list_services(domain=a.get("domain")),
        }
    except Exception as exc:
        logger.warning("Could not load HA tools for assist query: %s", exc)
        tool_dispatch = {}

    # 2b. Extra Hermes toolsets (MCP servers, e.g. the qBArm robot arm)
    extra_tools, extra_dispatch, extra_names = _assist_extra_tools()
    if extra_tools:
        ha_tools = ha_tools + extra_tools
        tool_dispatch = {**tool_dispatch, **extra_dispatch}
        if any("qbarm" in n for n in extra_names):
            system_prompt += (
                "\n\nYou can also control the qBArm robot arm on the user's desk with the mcp__qbarm__* tools "
                "(look at objects, pick, place, hand over, take from the hand, jog, home, claw, stop). Long actions "
                "start and run on while you answer: say in one short sentence what you started (\"Picking up the "
                "tape roll now\"). If the user says stop, call the stop tool at once. Directions are the user's "
                "(left = their left, towards me = closer to them).")

    # Load previous conversation turns if present
    prior_history = _get_history(conversation_id)

    messages = [{"role": "system", "content": system_prompt}]
    if prior_history:
        messages.extend(prior_history)
    messages.append({"role": "user", "content": f"Language: {language}\nUser request: {text}"})

    # 3. Call LLM with tool loop
    from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
    import inspect
    max_turns = 4  # e.g. look -> pick -> spoken answer
    final_text = ""
    provider = None
    model = None

    for _ in range(max_turns):
        call_kw: dict[str, Any] = {
            "task": "assist_query",
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 512,
        }
        if ha_tools:
            call_kw["tools"] = ha_tools

        # Retry on the same (configured) route rather than falling back to ctx.llm: the
        # generic fallback drops the HA tools and can wander through slow cloud providers,
        # which blows past HA's deadline.
        resp = None
        for attempt in range(_ASSIST_LLM_RETRIES + 1):
            try:
                resp = await async_call_llm(**call_kw)
                break
            except Exception as exc:
                if attempt < _ASSIST_LLM_RETRIES:
                    logger.warning("Assist query LLM call failed (attempt %d), retrying: %s", attempt + 1, exc)
                    continue
                logger.error("Assist query LLM call failed: %s", exc)
                return {
                    "ok": False,
                    "text": "Sorry, I couldn't reach the language model right now.",
                    "conversation_id": conversation_id,
                    "error": str(exc),
                }

        provider = getattr(resp, "provider", None)
        model = getattr(resp, "model", None)
        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        msg = getattr(choice, "message", None)
        tool_calls = getattr(msg, "tool_calls", None) if msg else None

        if tool_calls:
            # Append assistant message with tool calls
            msg_dict = {"role": "assistant", "content": getattr(msg, "content", "") or ""}
            raw_tc = []
            for tc in tool_calls:
                fn_name = getattr(tc.function, "name", "")
                fn_args = getattr(tc.function, "arguments", "{}")
                raw_tc.append({
                    "id": getattr(tc, "id", f"call_{fn_name}"),
                    "type": "function",
                    "function": {"name": fn_name, "arguments": fn_args},
                })
            msg_dict["tool_calls"] = raw_tc
            messages.append(msg_dict)

            # Execute tool calls
            for tc in raw_tc:
                fn_name = tc["function"]["name"]
                fn_args_str = tc["function"]["arguments"]
                try:
                    fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
                except Exception:
                    fn_args = {}

                handler = tool_dispatch.get(fn_name)
                if handler:
                    try:
                        res = handler(fn_args)
                        if inspect.isawaitable(res):
                            res = await res
                        result_str = json.dumps(res) if not isinstance(res, str) else res
                    except Exception as err:
                        result_str = json.dumps({"error": str(err)})
                else:
                    result_str = json.dumps({"error": f"Unknown tool: {fn_name}"})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })
        else:
            final_text = _clean_spoken_text(extract_content_or_reasoning(resp))
            if final_text:
                break

    if not final_text:
        # If loop ended on tool calls without producing a final text turn, ask the LLM for a spoken summary
        try:
            summary_res = await async_call_llm(
                task="assist_query",
                messages=messages + [{"role": "user", "content": "Briefly state what you found or did in one spoken sentence."}],
                temperature=0.2,
                max_tokens=256,
            )
            final_text = _clean_spoken_text(extract_content_or_reasoning(summary_res))
        except Exception:
            pass

    if not final_text:
        final_text = "I processed your request."

    final_text, continue_conversation = _split_continue_marker(final_text)

    # Save turns to conversation history cache for multi-turn dialogue context
    _append_history(conversation_id, text, final_text)

    return {
        "ok": True,
        "text": final_text,
        "conversation_id": conversation_id,
        "provider": provider,
        "model": model,
        # Forwarded verbatim in assist_response; the HA integration maps it onto
        # ConversationResult(continue_conversation=...) so the satellite keeps listening.
        "continue_conversation": continue_conversation,
    }


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

VOICE_STATUS_SCHEMA = {
    "name": "voice_status",
    "description": (
        "Report the current state of the Voice Stack engines "
        "(wake word, STT, TTS, media_player) and pipeline."
    ),
    "parameters": {"type": "object", "properties": {}},
}

VOICE_ENABLE_SCHEMA = {
    "name": "voice_enable",
    "description": (
        "Enable continuous voice mode — wake word detection, STT, "
        "Hermes LLM round-trip, and TTS playback through HA media_player. "
        "The pipeline runs in the background until disabled."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "media_player_entity": {
                "type": "string",
                "description": "HA media_player entity for TTS output (e.g. media_player.kitchen_speaker). "
                               "If omitted, uses HERMES_MEDIA_PLAYER env var or local speakers.",
            },
        },
    },
}

VOICE_DISABLE_SCHEMA = {
    "name": "voice_disable",
    "description": "Disable continuous voice mode.",
    "parameters": {"type": "object", "properties": {}},
}

VOICE_SPEAK_SCHEMA = {
    "name": "voice_speak",
    "description": (
        "Speak text through the configured TTS engine and output to "
        "the configured media player or local speakers."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to speak.",
            },
            "media_player_entity": {
                "type": "string",
                "description": "Optional HA media_player to output to.",
            },
        },
        "required": ["text"],
    },
}

VOICE_LISTEN_SCHEMA = {
    "name": "voice_listen",
    "description": (
        "One-shot listen: record audio from the microphone, transcribe it, "
        "and return the text. Useful for testing or push-to-talk workflows."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "duration": {
                "type": "number",
                "description": "Maximum recording duration in seconds (default: 10).",
            },
            "language": {
                "type": "string",
                "description": "Language code (e.g. 'en', 'fr'). Pass to STT engine.",
            },
        },
    },
}

VOICE_PROMPT_SCHEMA = {
    "name": "voice_prompt",
    "description": (
        "Return the voice-optimised system prompt with current Home Assistant "
        "context injected. Use this when Hermes is about to enter a voice "
        "interaction to ensure concise, natural spoken responses."
    ),
    "parameters": {"type": "object", "properties": {}},
}


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

_TOOLS = (
    ("voice_status",  VOICE_STATUS_SCHEMA,  _handle_voice_status,  "🎙️"),
    ("voice_enable",  VOICE_ENABLE_SCHEMA,  _handle_voice_enable,  "🔊"),
    ("voice_disable", VOICE_DISABLE_SCHEMA, _handle_voice_disable, "🔇"),
    ("voice_speak",   VOICE_SPEAK_SCHEMA,   _handle_voice_speak,   "🗣️"),
    ("voice_listen",  VOICE_LISTEN_SCHEMA,  _handle_voice_listen,  "👂"),
    ("voice_prompt",  VOICE_PROMPT_SCHEMA,  _handle_voice_prompt,  "📋"),
)


def register(ctx) -> None:
    """Register Voice Stack tools with Hermes.

    Registration is unconditional — tools that require unavailable engines
    return descriptive errors rather than being hidden, so users can see
    what's missing via voice_status.
    """
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="voice_stack",
            schema=schema,
            handler=handler,
            emoji=emoji,
        )
    # Start the HA-facing WebSocket receiver used by the Home Assistant
    # custom integration at /api/hermes/ws. It is fire-and-forget: when the
    # port is already occupied or aiohttp is unavailable, the warning is logged
    # and normal tool registration still succeeds.
    try:
        from .ws_receiver import set_assist_query_handler, start_ws_receiver
        set_assist_query_handler(lambda payload: _handle_assist_query_with_llm(ctx, payload))
        start_ws_receiver()
    except Exception as exc:
        logger.warning("Hermes HA WebSocket receiver did not start: %s", exc)

    # Run availability check in background so voice_status is accurate
    _init_engines()
