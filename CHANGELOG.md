## [0.15.0]
- Added new UI schema widgets for `tabs`/`tab` (including initial tab selection and active background styling) plus a `visible` boolean widget property for conditional rendering.
- Updated Python LiveKit integration dependencies to include `livekit==1.0.20`.

## [0.14.0]
- Breaking change: toolkit extension hooks were simplified to a synchronous `get_toolkit_builders()` API and tool selection now uses per-toolkit configuration objects (not just tool names)
- `LLMTaskRunner` now supports per-client and per-room rules, plus dynamically injected required toolkits at call time
- `TaskRunner.ask` now supports optional binary attachments; `LLMTaskRunner` can unpack tar attachments and pass images/files into the LLM conversation context
- `AgentsClient.ask` now returns `TextResponse` when the agent responds with plain text (instead of always treating answers as JSON)
- Added a CLI `task-runner` command to run/join LLM task runners with configurable rules, schemas, toolkits, and optional remote LLM delegation

## [0.13.0]
- Added `initial_json` and explicit schema support when opening MeshDocuments, enabling schema-first document initialization
- Added binary attachment support when invoking agent tools so tool calls can include raw payload data
- Breaking change: toolkit construction is now async and receives the active room client, enabling toolkits that introspect room state during build
- Added database schema inspection and JSON Schema mappings for data types to support tool input validation and generation
- Introduced database toolkits (list/inspect/search/insert/update/delete) and integrated optional per-table enablement into the chatbot/mailbot/helpers CLI flows

## [0.12.0]
- Reduce worker-queue logging verbosity to avoid logging full message payloads

## [0.11.0]
- Stability

## [0.10.1]
- Stability

## [0.10.0]
- Stability

## [0.9.3]
- Stability

## [0.9.2]
- Stability

## [0.9.1]
- Stability

## [0.9.0]
- Stability

## [0.8.4]
- Stability

## [0.8.3]
- Stability

## [0.8.2]
- Stability

## [0.8.1]
- Stability

## [0.8.0]
- Stability

## [0.7.2]
- Stability

## [0.7.1]
- Stability
