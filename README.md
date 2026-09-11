# ha-voice-hermes-plugin

Hermes plugin connecting Home Assistant Voice Assist with Hermes Agent.

## Features
- **Assist Pipeline Integration**: Receives `assist_query` events over WebSocket (`/api/hermes/ws`) from Home Assistant.
- **Device Control & Inspection**: Equips Hermes with Home Assistant tools (`ha_list_entities`, `ha_get_state`, `ha_call_service`, `ha_list_services`) to query and control smart home entities.
- **Conversational Multi-Turn Context**: Maintains conversational memory across turns for each `conversation_id`, allowing natural follow-ups (*"which are these lights?"*, *"turn them off"*).
- **Spoken Text Summaries**: Voice-optimized system prompt and response formatting tailored for text-to-speech.

## Structure
- `plugins/voice_stack/`: Voice pipeline, WebSocket receiver (`/api/hermes/ws`), and Assist query handler.
- `plugins/home_assistant/`: Tools and helpers for device control, entity cache, and service dispatch.

## Installation
Clone or copy into your Hermes plugins directory:
```bash
cp -r plugins/voice_stack ~/.hermes/plugins/
cp -r plugins/home_assistant ~/.hermes/plugins/
```

Ensure both plugins are enabled in `~/.hermes/config.yaml`:
```yaml
plugins:
  enabled:
    - home_assistant
    - voice_stack
```

Restart the Hermes gateway:
```bash
systemctl --user restart hermes-gateway.service
```
