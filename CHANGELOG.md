## [0.28.10]
- Stability

## [0.28.9]
- Stability

## [0.28.8]
- Stability

## [0.28.7]
- Stability

## [0.28.6]
- Stability

## [0.28.5]
- Stability

## [0.28.4]
- Stability

## [0.28.3]
- Stability

## [0.28.2]
- Stability

## [0.28.1]
- Stability

## [0.28.0]
- BREAKING: AgentChatContext was replaced by AgentSessionContext; create_chat_context/init_chat_context moved to create_session/init_session, TaskContext and ChatThreadContext now expose .session, and LLMAdapter.next no longer accepts a tool_adapter argument.
- BREAKING: The PromptAgent helper was removed from the Python agents package.
- Deprecated init_chat_context in favor of init_session, emitting DeprecationWarning for legacy overrides.
- Added async session lifecycle management (start/close) and async context manager support for sessions, task contexts, and thread adapters for safer cleanup.
- Added OpenAI Responses websocket mode (default) with persistent sockets, configurable ping/timeout, and a new OpenAIResponsesSessionContext (old class name kept as an alias).
- Added tool-call stream close status codes and InvalidToolDataException; invalid tool data closes with status 1007 and abnormal closes surface as errors.
- Added thread status modes and steering: thread status now includes text + mode, "steer" messages are supported via on_thread_steer, and Codex can steer active turns with cancellable tasks.

## [0.27.2]
- Stability

## [0.27.1]
- Stability

## [0.27.0]
- Added project route/domain APIs to the Python client and CLI, covering create/update/get/list/delete operations with per-route annotations.
- Added mailbox annotations support across Python client and CLI mailbox workflows.
- Added secret-based container environment variable support in service specs (`SecretValue`).
- Added a new webserver CLI workflow for serving static/Python routes and packaging/deploying them as services.
- Added a new `meshagent-codex` integration package plus Codex CLI workflows for chatbot/task-runner/worker agents.
- Added Codex approval/sandbox policy controls, skill directory support, and container runtime defaults with mount controls.
- Added multi-rules-file loading and expanded agent CLI options (`--skill-dir`, `--shell-image`, `--shell-image-mount`, `--require-time`).
- Added normalized agent event ingestion in thread adapters and persisted `event` elements in thread schema.
- Updated OpenAI/Anthropic tool adapters to convert HTML responses to Markdown and reduce duplicate tool lifecycle stream events.
- Updated shell/container execution paths to pass argv arrays instead of shell-joined command strings.
- Updated third-party Python dependencies: added `html-to-markdown~=2.24.3`.
- Breaking change: annotation constants moved from `meshagent.webhook.*` to `meshagent.request.*`.
- Breaking change: CLI secret usage changed from `meshagent secrets --secret-id ...` to `meshagent secret --id ...`.

## [0.26.0]
- Stability

## [0.25.9]
- Stability

## [0.25.8]
- Stability

## [0.25.7]
- Stability

## [0.25.6]
- Stability

## [0.25.5]
- Stability

## [0.25.4]
- Stability

## [0.25.3]
- Stability

## [0.25.2]
- BREAKING: Agent metadata field renamed from labels to annotations across Python agents and examples.
- Thread history now limits appended messages to a context window and exposes search/count/range tools for conversation history outside that window.
- Tool decorator now supports bound instance/class methods as tools.
- Anthropic adapters omit null/unset fields in serialized payloads and log tool call errors for easier diagnostics.
- Schema document grep now requires keyword arguments for options like ignore_case.

## [0.25.1]
- Added Anthropic web search and web fetch toolkits, including beta header injection and request middleware support.
- Added a container-based shell tool to run commands in persistent containers with configurable image, mounts, and environment.
- Expanded the web fetch tool to return text, JSON, or file responses with HTML-to-Markdown conversion and content-type handling.
- Breaking: CLI commands now reject OpenAI-only tool flags when using Claude models (image generation, local shell, apply patch, computer use).
- CLI MCP bridge adds streamable HTTP connections plus custom headers and secret-backed headers; OAuth2 secret set now accepts text/base64 input and identity-scoped secrets.
- Dependency updates: mcp to ~1.26.0; html-to-markdown to ~2.24.3.

## [0.25.0]
- Added SQL column-schema parsing and CLI support for SQL-like `--columns` when creating tables or adding columns.
- Breaking: SQL query requests now use a single `params` map for typed bindings instead of `parameters`/`param_values`.
- Added `published`/`public` port fields in service specs for externally routed services.
- Secrets set now supports `for_identity` to set secrets on behalf of another identity.
- Added a Slack events HTTP bot package with dependencies including `pyjwt` 2.10.
- Breaking: the CLI `exec` command was removed.
- ThreadAdapter message writing now uses `write_text_message` and accepts participant name strings.

## [0.24.5]
- Stability

## [0.24.4]
- Stability

## [0.24.3]
- Stability

## [0.24.2]
- Stability

## [0.24.1]
- Stability

## [0.24.0]
- Breaking: removed `AgentsClient.ask` and `list_agents` from the Python SDK.
- Breaking: `AgentCallContext` renamed to `TaskContext`, planning module and Pydantic agent utilities removed, and discovery toolkit no longer lists agents.
- Feature: TaskRunner refactor adds RunTaskTool/RemoteToolkit support plus a `run()` helper for direct execution.
- Feature: task-runner CLI adds `run` and an `allow_model_selection` toggle for LLM task runners; legacy agent ask/list CLI commands removed.

## [0.23.0]
- Breaking: service template APIs now expect YAML template strings and ServiceTemplateSpec.to_service_spec() no longer accepts values; use ServiceTemplateSpec.from_yaml(..., values) for Jinja rendering
- Added Jinja/YAML template parsing and ServiceSpec.from_yaml for loading service specs from YAML
- Added file storage mounts and token role fields in service/container specs
- Added render_template client method plus new User/UserRoomGrant models and a none project role

## [0.22.2]
- Stability

## [0.22.1]
- Stability

## [0.22.0]
- Added meshagent-anthropic with Anthropic Messages adapter, MCP connector toolkit support, and an OpenAI-Responses-compatible stream adapter (depends on anthropic>=0.25,<1.0).
- Breaking: agent naming now derives from participant name (Agent.name deprecated; TaskRunner/LLMRunner/Worker/VoiceBot constructors no longer require name; Voicebot alias removed; MailWorker renamed to MailBot with queue default).
- Breaking: SecretsClient methods renamed to list_secrets/delete_secret and expanded with request_secret/provide_secret/get_secret/set_secret/delete_requested_secret flows.
- Breaking: Meshagent client create_service/update_service now return ServiceSpec objects; service-template create/update helpers added for project and room services.
- OpenAI Responses adapter adds context window tracking, compaction via responses.compact, input-token counting, usage storage, max_output_tokens control, and shell tool env injection.
- RoomClient can auto-initialize from MESHAGENT_ROOM/MESHAGENT_TOKEN; websocket URL helper added.
- Schema documents add grep/tag queries and ChatBotClient; chat reply routing now targets the requesting participant reliably.
- Database toolkit now expects update values as a list of column updates and defaults to advanced search/delete tools.
- Dependency addition: prompt-toolkit~=3.0.52 added to CLI 'all' extras.

## [0.21.0]
- Breaking: the Image model no longer exposes manifest/template metadata in image listings.
- Add token-backed environment variables in service specs so Python clients can inject participant tokens instead of static values.
- Add `on_demand` and `writable_root_fs` flags on container specs to control per-request services and filesystem mutability.
- Breaking: the agent schedule annotation key is corrected to `meshagent.agent.schedule`; update any existing annotations using the old spelling.
- Add a Shell agent type and a shell command annotation for service metadata.

## [0.20.6]
- Stability

## [0.20.5]
- Stability

## [0.20.4]
- Stability

## [0.20.3]
- Stability

## [0.20.2]
- Stability

## [0.20.1]
- Stability

## [0.20.0]
- Breaking: mailbox create/update requests must now include a `public` flag (SDK defaults to `False` when omitted in method calls)
- Mailbox response models include a `public` field
- Breaking: service specs now require either `external` or `container` to be set
- External service specs allow omitting the base URL
- Service template variables include optional `annotations` metadata
- CLI mailbox commands support `--public` and include the `public` value in listings
- Mailbot whitelist parsing accepts comma-separated values
- Fixed JSON schema generation for database delete/search tools

## [0.19.5]
- Stability

## [0.19.4]
- Stability

## [0.19.3]
- Stability

## [0.19.2]
- Add boolean data type support plus `nullable`/`metadata` on schema types and generated JSON Schema.
- BREAKING: OpenAI proxy client creation now takes an optional `http_client` (request logging is configured via a separate logging client helper).
- Shell tool now reuses a long-lived container with a writable root filesystem, runs commands via `bash -lc`, and defaults to the `python:3.13` image.
- Add `log_llm_requests` support to enable OpenAI request/response logging.

## [0.19.1]
- Add optional metadata to agent chat contexts and propagate it through message-stream LLM delegation, including recording thread participant lists
- Add an option for the mailbot CLI to delegate LLM interactions to a remote participant instead of using the local LLM adapter

## [0.19.0]
- Add a reusable transcript logger/transcriber agent that writes conversation segments to transcript documents from live conversation events or user-turn completion
- Add optional voicebot transcription via a provided transcript path, wrapping the voice agent to persist user/assistant speech to a transcript document
- Refactor meeting transcription to use the shared transcript logger with per-participant sessions and improved session lifecycle cleanup
- Breaking change: starting a new email thread no longer accepts attachments; attachments are now handled by a dedicated “new thread with attachments” tool that downloads files from room storage before sending
- Simplify CLI agent room-rules loading and ensure worker message toolkits include inherited toolkits

## [0.18.2]
- Stability

## [0.18.1]
- Updated OpenAI Python SDK dependency to `openai~=2.14.0` (from `~2.6.0`).
- Breaking: OpenAI Responses adapter no longer sends tool definitions with requests, disabling tool/function calling via the Responses API.
- CLI deploy commands now report “Deployed service” on successful deploys.
- Shell toolkit/tool builders now pass the configured shell image via the `image` field.

## [0.18.0]
- Added local TCP port-forwarding helper that bridges to the remote tunnel WebSocket
- Added a CLI `port forward` command to expose container ports locally
- Added `writable_root_fs` support when running containers
- Added `host_port` support for service port specs
- Added `ApiScope.tunnels` support in participant tokens (including `agent_default(tunnels=...)`)
- Added container-based Playwright “computer use” and enabled computer-use toolkits for chatbot/worker/mailbot flows
- Removed `scrapybara` from the computers package dependencies
- OpenAI proxy client can now optionally log requests/responses with redacted authorization headers

## [0.17.1]
- Prevented worker toolkit lifecycle collisions when running alongside other toolkits by isolating the worker’s remote toolkit handling.
- Improved the error message when attempting to start a single-room agent more than once.

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
- `AgentsClient.ask` now returns `TextChunk` when the agent responds with plain text (instead of always treating answers as JSON)
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
