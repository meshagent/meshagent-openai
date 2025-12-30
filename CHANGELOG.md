## [0.17.0]
- Added scheduled tasks support to the Python accounts client (create/update/list/delete scheduled tasks) with typed models
- Added mailbox CRUD helpers to the Python accounts client and improved error handling with typed HTTP exceptions (404/403/409/400/5xx)
- Added `RequiredTable` requirement type plus helper to create required tables, indexes, and optimize them automatically
- Added database namespace support for database toolkit operations (inspect/search/insert/update/delete in a namespace)
- Enhanced worker and mail agents (per-message tool selection, optional remote toolkit exposure for queue task submission, reply-all/cc support)
- Updated Python dependency: `supabase-auth` from `~2.12.3` to `~2.22.3`

## [0.16.0]
- Add optional `namespace` support across database client operations (list/inspect/create/drop/index/etc.) to target namespaced tables
- Update dependencies `livekit-api` to `~1.1` (from `>=1.0`) and `livekit-agents`/`livekit-plugins-openai`/`livekit-plugins-silero`/`livekit-plugins-turn-detector` to `~1.3` (from `~1.2`)

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
