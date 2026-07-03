## [0.45.7]
- Tightened container-shell input validation and managed-container parsing so malformed `commands`, `container_id`, `env`, and timeout fields fail fast with clearer errors.
- Changed container mount merging to stop carrying project mounts into the merged shell mount spec.
- Improved tool hosting error mapping and remote-tool message validation so invalid payloads surface as invalid-request errors.
- Simplified the image pack/build flow and changed default image tagging so non-BuildKit images use the `-nydus` suffix while BuildKit and Playwright defaults keep the plain version tag.
- Adjusted computer automation to preserve blocked-URL handling correctly and fixed async mouse drag execution in Docker-backed computers.
- Made memory-toolkit and remote graph loading more resilient when using remote storage roots and room-local API endpoints.

## [0.45.6]
- Added strict dataset prompt rendering with escaped-brace support and simple field placeholders, replacing generic `str.format` expansion.
- Tightened OpenAI adapter handling so code-interpreter results must be structured objects with required string fields, and invalid payloads now error out earlier.
- Aligned computer/browser tooling with Python behavior for Browserbase setup, Docker geometry handling, and optional rule initialization.
- Synchronized Python package dependencies and release metadata across the API, agents, CLI, OpenAI, tools, computers, codex, and related companion packages.

## [0.45.5]
- Shell command analysis now handles mixed absolute and dynamic write targets more safely, keeping generated file previews and path grouping accurate.

## [0.45.4]
- Stability

## [0.45.3]
- Added `whoami`, typed `ServiceAccount.email`, and service-account `view` filtering.
- Hardened shared HTTP session creation with certifi-backed TLS, optional host aliases, and an extra CA bundle.
- Exposed service runtime startup errors and event histories through room service-state responses.
- Added `pull_secret` support for service container specs and enforced id-only secret payloads, breaking older named-secret service specs.
- Updated agent chat and thread handling to consume direct payload envelopes, clean outbound metadata, and match the newer completion/thread message shape.
- Expanded OpenAI, Anthropic, CLI, and tool parity so web fetch/grep, HTML-to-Markdown, regex handling, data URLs, Docker subprocesses, and MCP/web tool behavior more closely match Python semantics.
- Removed deprecated compatibility aliases around thread adapters and managed-agent exports.

## [0.45.2]
- Added `whoami`, typed `ServiceAccount.email`, and service-account `view` filtering, and hardened shared HTTP session creation with certifi-backed TLS, optional host aliases, and an extra CA bundle.
- Exposed service runtime startup errors and event histories through room service state responses.
- Added `pull_secret` to service container specs and enforced id-only secret payloads, which is a breaking change for older named-secret service specs.
- Updated agent chat and thread handling to consume direct payload envelopes, clean outbound metadata, and align completions/thread behavior with the newer message shape.
- Expanded OpenAI, Anthropic, CLI, and tool parity so web fetch/grep, HTML-to-Markdown, regex handling, data URLs, Docker subprocesses, and MCP/web tool behavior more closely match Python semantics.
- Removed deprecated compatibility aliases around thread adapters and managed-agent exports, and updated related tests and package wiring accordingly.

## [0.45.1]
- `meshagent.api` now supports `view` on service-account listing, and the shared HTTP helper uses certifi-backed TLS with optional host aliases and an extra CA bundle.
- The CLI image and deploy helpers now preserve digest-pinned published refs, canonicalize image mounts, and use personal-room paging for room-connect flows.
- Web, dataset, script, and computer tooling were tightened to match Python semantics more closely for filename inference, charset decoding, HTML-to-Markdown conversion, regex behavior, environment-variable handling, and Docker subprocess execution.
- OpenAI, Anthropic, and agent abstractions gained parity updates for threaded channels, completions/response streaming, and raw-output handling.

## [0.45.0]
- Added `whoami` support plus typed `ServiceAccount.email` and `Whoami` responses in the API client.
- Exposed service runtime startup errors and lifecycle events through room service state responses.
- Added `pull_secret` to container specs and tightened secret validation to id-only payloads, breaking legacy secret objects.
- Expanded the room-services CLI with a `describe` command and richer list output that includes runtime events, and updated image deploy cleanup to run after deploy completion.
- Removed the `thread_adapter` alias, require `thread_storage`, and switched queue targeting to `path` instead of `thread_id`, which is a breaking change for existing agent integrations.
- Improved dataset tooling and OpenAI code-interpreter handling, including Python-compatible row formatting, binary row support, and correct log/file aggregation.

## [0.44.13]
- Stability

## [0.44.12]
- Added generated project support flows across the SDK Python CLI and API, including visible-room selection, deploy helpers, and broader create-template coverage.
- Expanded generated Python app templates with a new task queue dashboard app, simpler install/dev scripts, and safer contact-form and mailbox delivery behavior.
- Added session log and metrics CLI commands, plus support helpers for `SingleRoomAgent` and toolkit annotation preservation in room listings.
- Fixed OpenAI shell-output restoration and corrected audio transcription pricing and reasoning-token accounting.
- Improved template and deployment robustness with stale deploy-value handling and other create-flow hardening.

## [0.44.11]
- Stability

## [0.44.10]
- Stability

## [0.44.9]
- Stability

## [0.44.8]
- Stability

## [0.44.7]
- Realtime websocket sessions now expand nested namespace toolkits into individual function tools, improving support for live tool advertising in OpenAI realtime flows.

## [0.44.6]
- Breaking: Codex setup now uses a single default-provider flow and removes the older multi-profile conflict/update screens.
- Codex setup now detects all MeshAgent-managed Codex configs instead of only the active project or API host, so stale configs can be updated or removed during setup.

## [0.44.5]
- OpenAI completions now emit structured assistant text and tool-call lifecycle events, including arguments and results, so consumers can reconstruct streamed assistant and tool activity.
- Reasoning-end events are now preserved even when no active reasoning buffer exists, as long as metadata is available, which restores reasoning dataset replay.
- OpenAI response handling now includes all structured output items instead of only message and compaction items, improving replay coverage for additional response types.

## [0.44.4]
- Breaking: `AgentSessionContext` no longer exposes `previous_messages` or `previous_response_id`; restore flows now operate from the current `messages` payload instead of maintaining a separate history buffer.
- OpenAI Responses restore now preserves encrypted reasoning metadata, normalizes legacy reasoning items, and forces stateless requests to use `store=False` while replaying the current context.
- Agent event handling and dataset thread storage now keep reasoning provider/model metadata so empty reasoning items with encrypted content can still be restored without losing the payload.
- Managed agents now share the `llm` backend abstraction, which changes thread creation and resume behavior and lets thread naming consider the selected provider/model plus audio attachments.
- Updated `typer` to `~=0.26.6`.

## [0.44.3]
- Stability

## [0.44.2]
- Base chat clients now expose an async event stream for consuming emitted agent payloads alongside listener callbacks.
- Messaging chat clients now distinguish first connect from reconnect, track participant add/remove events, and reopen open sessions when the agent returns.
- Restored session context now resolves stored room file URLs into inline attachments, including PDF and image handling for OpenAI- and Anthropic-style message content.
- Fresh turns no longer trigger redundant storage restoration when there is no prior thread history.

## [0.44.1]
- Stability

## [0.44.0]
- Python agent processes now support multiple thread storages, thread watch/unwatch flows, and storage-aware list/view routing for thread lifecycle operations.
- Thread startup now preserves better thread naming and metadata, including fallback names derived from message content and attachments when no explicit name is provided.
- Managed-agent server code now creates thread metadata earlier and persists thread naming information consistently through websocket-driven lifecycles.
- Single-room and Codex orchestration now avoid unnecessary toolkit-discovery waits and improve startup latency and initialization behavior.

## [0.43.4]
- Public room toolkit metadata now preserves annotations end-to-end, and the SDK exports tool-search metadata for consumers that need to discover searchable tools.
- Tool listings now round-trip annotations alongside tools and participant IDs, so extra toolkit metadata is no longer lost when clients read or write it.
- Agent and room message handling now keeps `created_at` timestamps through streamed deltas and live thread updates, improving ordering and replay consistency.
- Responses integration now supports tool namespaces and search across the Python agent tooling stack.

## [0.43.3]
- Stability

## [0.43.2]
- Added backend-aware fields across agent messages and chat/session APIs in `meshagent-agents`, enabling multi-backend conversations, model changes, and room/thread opening flows.
- Breaking: `meshagent-codex` was reorganized around a dedicated process, supervisor, and thread-storage stack, so the old internal process/chat wiring moved.
- Removed the mandatory Codex binary wheel dependency and vendored the OpenAI Codex client into `meshagent-codex`.
- Added thread inspection and thread-storage diagnostics for Codex sessions, along with no-room process mode and improved inline attachment handling in the CLI.
- Added IAP websocket support in `meshagent-api`, including nullable tokens, `withIAP()`, and Authorization-header based connections.

## [0.43.1]
- Added IAP room websocket support in the SDK chat and channel stack.
- Reworked the CLI around a unified process backend and Codex integration, replacing the older Codex-specific launch path and changing thread loading semantics.
- Added multi-backend support, the Codex thread-storage repository and diagnostics, and managed thread-storage fixes so threads can be loaded, renamed, and deleted through the new repository flow.
- Added TUI image attachment support with `textual-image[textual]~=0.12.0`.
- Removed the hard dependency on the Codex binary wheel.

## [0.43.0]
- `meshagent-agents` now supports backend-aware multi-backend supervision, with backend metadata in thread/model messages, attachment-aware prompts, and backend-aware thread/model/realtime-audio operations.
- `meshagent-codex` was split into a dedicated process, supervisor, and thread-storage stack and now vendors `openai_codex`, with a new `meshagent-openai==0.42.2` dependency.
- `meshagent-cli` now uses Typer consistently, adds lazy command loading, image/PDF/file paste-and-drop handling for `ask`, thread-sidebar controls, and improved `doctor`, `create`, and deploy-room workflows, including new room-workspace and meeting templates.
- `meshagent-openai` now preserves image-generation call inputs and emits structured image-generation results, and the LLM proxy and agent server websocket auth path now accept the `meshagent-agent.` token prefix.
- Third-party dependency updates include `textual-image[textual]~=0.12.0` for inline image rendering.

## [0.42.2]
- Added `wait_for_exit_status` and richer container/build models, exposing image IDs, runtime stats, published build image digests, and detailed exit status information while keeping the existing integer exit-code helper.
- The deploy flow now resolves completed builds to published digests, rewrites deploy plans to use the resolved image reference, and cleans up replaced built images after successful deploys.
- Room client shutdown is now cancellation-safe, so protocol teardown completes even if exit is cancelled mid-close.
- CLI API URL resolution now prefers `MESHAGENT_API_URL` before the persisted active URL, matching the explicit environment override users expect.

## [0.42.1]
- Deploy liveness checks now treat `401` and `403` responses as live, improving detection for protected endpoints.
- Deploy log streaming now cancels background log and progress tasks cleanly on exit, avoiding hangs during shutdown.
- Deploy TUI now supports copying selected text or the full deploy log buffer to the clipboard.

## [0.42.0]
- Added project lookup by key.
- Service and route specs now support container templates, `host_port`, and `stripPrefix`; route-path serialization omits `stripPrefix` when it is false.
- Room/container responses now expose structured port mappings instead of bare integers, which is a breaking shape change for container listings.
- Room creation and room-service helpers now carry annotations and permissions through the API.
- Container creation and room-client helpers now accept the `template` option.

## [0.41.10]
- Breaking change: several resource lookup commands now use `get` instead of `show`, including API keys, feeds, LLM loggers, mailboxes, registries, routes, services, sessions, storage, and subscriptions.
- Added `get` commands for projects, dataset branches, and webhooks, plus a `get` alias for memory inspection.

## [0.41.9]
- Deployment config models now carry an optional server version, and `meshagent config get version` can surface it.
- `meshagent` now performs a best-effort startup version check, warns when the installed client is older than the server, and `meshagent version` reports both client and server versions instead of the previous raw version string, so scripts that parse the old output need to update.

## [0.41.8]
- Stability

## [0.41.7]
- Deployment config models now carry optional server version metadata, allowing API consumers to read the server version from config responses.
- CLI config lookup now supports returning the deployment version, and the version command now prints both client and server versions.
- The CLI startup path now performs a best-effort server version check and emits a one-time warning when the CLI is behind the server.

## [0.41.6]
- Added deploy-ready project scaffolding and the Python contact-form starter, including deploy/dev/install scripts and generated deployment links.
- Improved CLI ask/deploy flows with room selection, domain entry, service lookup, and template-variable prompts.
- Extended the Python API client and cloud router to resolve services by name in both project and room scopes.
- Added attachment-aware signed download URLs across the Python storage stack so files can be served inline or as forced downloads.

## [0.41.5]
- Stability

## [0.41.4]
- `ChatThreadSession` now exposes thread-start, turn-steer, and interrupt workflows, along with richer pending-input state and active-turn tracking for acceptance, application, and rejection events.
- Container and service models now support a `template` value (`agent` or `none`), and container runs can opt into that template to receive the standard agent runtime environment and mount defaults.

## [0.41.3]
- Stability

## [0.41.2]
- `meshagent create` now uses clearer stable focus IDs and labels, adds an Anthropic chatbot option, and prints grouped next steps plus agent-toolkit deploy guidance for backend-agent templates.
- `meshagent rooms list` now defaults to rooms the current user can access, with `--all` to switch back to listing every room in the project.
- Deploy-room prompting now derives the Pages suffix from the configured API host, pre-fills a room-based subdomain, and validates subdomain-only input before constructing the final public domain.
- The CLI chat and process runtime now centralize turn-toolkit assembly and thread-list tooling through the supervisor, while websocket chat sessions keep web participants aligned with the base participant identity for on-behalf-of access.

## [0.41.1]
- Python feed subscription APIs and CLI commands now support an optional `filename_datetime_format`, and listing shows the stored value.
- The create workflow now prints a `cd` hint for new subfolders and blocks reusing an already occupied nested folder.
- Image deploys now preserve Dockerfile default environment values and clear the newly built image from the room cache after a successful build.

## [0.41.0]
- Managed-agent support now includes thread listing, thread create/update/delete events, attachment names, and sender-name trust for chat input.
- Websocket process support now uses `/messages`, adds `jwt`/`iap`/`none` auth modes, and supports websocket-based `process use` sessions.
- Route handling now uses the spec-based route model and supports room or agent backends.
- The CLI gained new agent/process/route flows, removed the `codex` command, and added `ascii-magic~=2.3`, `pillow~=11.3.0`, and `msgpack~=1.1`.
- Managed-agent storage and shell toolkits were removed from the public managed-agent surface.
- OpenAI, Anthropic, browser, computer, and toolkit helpers were updated to work with the new managed-agent and client-toolkit plumbing.
- Fixed thread storage, chat replay, and process shutdown races.

## [0.40.3]
- Added managed-agent spec and API models covering allowed models, toolkits, secrets, MCP servers, thread isolation, agent/room grants, and agent session listing.
- Route APIs now use `RouteSpec` with room or agent backends and preserve compatibility with legacy route payloads.
- Chat and channel code now supports websocket transport, participant connect/disconnect events, sender-name propagation, and attachment-aware thread start/load flows.
- Added a new `create` scaffolder with Dart, .NET, JavaScript, Python, React, and TypeScript templates, replacing the old `init`/Codex entrypoints.
- Added CLI dependencies on `ascii-magic~=2.3` and `pillow~=11.3.0`.

## [0.40.2]
- Stability

## [0.40.1]
- Stability

## [0.40.0]
- Breaking: Renamed OpenAI LLM adapter `next(...)` implementations to `create_response(...)`, including Responses, Chat Completions, and Realtime adapters.

## [0.39.9]
- Added/expanded `meshagent init` and `meshagent doctor` CLI workflows in the Python SDK, including TUI init improvements.
- Expanded `meshagent doctor` to provide richer, toolchain-aware diagnostics (Python/TypeScript/.NET), including stronger deployment/runtime guidance and missing toolchain detection.
- Implemented dataset table rename support and SDK dataset toolkit support for renamed dataset handling.
- Improved dataset path restoration and dataset-backed conversation handling in the SDK.
- Implemented dataset thread storage in the SDK, including dataset thread storage/watch plumbing for dataset-scoped conversation threads.
- Added SDK wiring for error reporting and transaction reconciliation-related CLI behavior.

## [0.39.8]
- Added `rename_table` support to the Python datasets client API (`DatasetsClient.rename_table`) for renaming dataset tables with optional namespace/branch
- Updated image dataset schema to store the image data column as `large_binary` (instead of `binary`) for newly created datasets
- Updated scrapy/dataset schema handling to use `large_string` for large compressed text fields (including image `src`/`alt`)
- CLI: ask-style TUI now supports a configurable assistant label name
- CLI: `meshagent process use` now routes through a room chat-channel session and streams text deltas into the ask-style TUI

## [0.39.7]
- Documentation cleanup: removed stale archived Python example agents/services/webserver routes.
- Documentation cleanup: removed several Python service example entrypoints (browser, document author, presentation author, voice, voice proofreader, voice tools).

## [0.39.6]
- CLI help docs generation was rewritten to recursively render command documentation for lazy-loaded Click/Typer command trees, with more robust hidden/deprecated filtering and deterministic command-block generation.
- CLI help reference generation now normalizes command output to produce stable reference content.
- Skill package validation now permits missing top-level help command references for `webserver`.

## [0.39.5]
- Added Scrapy crawler HTML/content stripping configuration via new `strip` and `strip_order` inputs (including support for stripping `scripts`, `css`, `whitespace`, `clean`, and `image-data-urls`)
- Changed default behavior for `content_format="html"` to strip `scripts` and inline image data URLs while preserving the rest of the HTML (and updated `--clean` CLI usage to map onto the new stripping configuration)
- Broke Scrapy dataset output schema by removing `links`, `link_urls`, `image_urls`, and reducing `images` to `src`/`alt` only; inline image data URLs are excluded from extracted images
- Changed index creation defaults: automatic creation no longer includes inverted/label indexes for removed link/image URL columns; `text` inverted index creation is now opt-in via `index_columns=("text",)`
- Updated generated dataset schema to apply ZSTD compression metadata to large string fields (including `text` and image fields)

## [0.39.4]
- Breaking: Python scheduled-task client and spec models now use a `ScheduledTaskSpec` contract (including queue/container targeting) instead of separate queue/schedule/payload parameters.
- Added Python scheduled-task run listing support with models/pages for runs and their status/attempt/timestamp fields.
- Updated scheduled-task client methods to support `room_id` filtering and the new spec-based request/response shapes.
- Added/updated CLI scheduled-task create/update flows to load the `ScheduledTaskSpec` from a YAML file and included new run-related CLI functionality.
- Removed generated CLI dataset functionality (including the previously available SQL-exec command).
- Added `croniter~=6.0` as a dependency to support cron parsing for scheduled tasks.

## [0.39.3]
- Added `meshagent-commoncrawl` package with Common Crawl import support (progress reporting, dataset record extraction/import utilities, and tests); includes dependencies such as `pyarrow~=21.0.0` and `warcio~=1.7`.
- Added `meshagent-scrapy` package with Scrapy-based dataset import support (scrapy import utilities, examples, and tests); includes dependencies such as `scrapy~=2.13`, `trafilatura~=2.0`, and `pyarrow~=21.0.0`.
- Updated OpenAI Responses adapter error handling to detect out-of-credits/`insufficient_quota` conditions and return a clearer non-retryable 402 response; also improved websocket error payload message extraction.
- Updated `meshagent-cli` default model selections from `gpt-5.4` to `gpt-5.5` across ask/chatbot/codex/task runner/mailbot/worker CLI flows.
- Updated `meshagent-cli` and `meshagent-python` packaging extras to include `meshagent-commoncrawl` and `meshagent-scrapy` (including dedicated `commoncrawl`/`scrapy` extras).
- Added/updated tests for the new OpenAI out-of-credits handling and for commoncrawl/scrapy importer functionality.

## [0.39.2]
- Aligned meshagent-sdk Python package dependency pins to internal `meshagent-*` packages `0.39.1` (from `0.39.0`) including `meshagent-api`, `meshagent-agents`, `meshagent-tools`, `meshagent-openai`, `meshagent-anthropic`, `meshagent-llm-proxy`, and related components.

## [0.39.1]
- Added paged response models and `*_page` methods to the Python Meshagent client for users/rooms/mailboxes/routes/feeds/OAuth clients/scheduled tasks (each supporting `count`/`offset`/`filter` and returning `total`).
- Updated existing Python list/get methods to use paged requests by default (default page-size behavior changed) and to accept paging/filter parameters.
- Updated Python CLI list commands (feeds, mailboxes, rooms, routes, scheduled tasks) to add `--filter`, `--count`, and `--offset` flags and to pass them through to the new paged API methods.
- Updated room-list CLI option handling to use `--count` for paging (with the previous `--limit` behavior adjusted/hidden).

## [0.39.0]
- Updated Python CLI networking to use certifi-backed shared client sessions for consistent TLS behavior during fetches.
- Updated WebSocket/response adapter HTTP session handling to use meshagent’s shared `new_client_session` instead of ad-hoc aiohttp sessions.
- Added dataset SQL cancellation support to the dataset toolkit execution flow (including cancel status/results).
- Expanded dataset index management capabilities (index configuration/remapping and index metadata support) end-to-end in Python tooling/clients.
- Applied “database” -> “datasets” terminology and dataset refactor updates across Python dataset/tooling clients (breaking for prior database-named usage).
- Added usage export and usage annotation plumbing for cost/billing reporting, including ClickHouse usage annotation/projection support and LLM tool usage instrumentation.
- Extended LLM proxy support with “pipes” plus custom LLM usage tracking and pricing updates (including gpt-5.5 pricing).
- Fixed async OAuth token exchange path in the Python CLI auth flow.
- Added mailbox and route domain validation helpers to harden routing/mailbox configuration.
- Added LLM agent turn lifecycle tracing instrumentation to improve observability/debugging.

## [0.38.4]
- On Windows, meshagent-cli now uses `subprocess.list2cmdline` to correctly pass arguments when launching the Claude integration.
- `meshagent image deploy` now supports `--wait/--no-wait` (wait enabled by default) to wait for deployment readiness, stream container logs, and verify route liveness when `--domain` is provided.
- meshagent-llm-proxy pricing now filters out zero-usage token values and returns no pricing when all usage values are zero.
- meshagent-cli setup wizard now only reuses an authenticated session when the requested API URL matches the active API URL.
- meshagent-cli `all` extra now includes `meshagent-otel==0.38.3`.

## [0.38.3]
- Breaking: `Image` now uses `references`/`preferred_ref` with optional metadata (timestamps/media type) instead of `tags`/`size`, and `inspect_image` returns manifests/layers/content size.
- `Meshagent.get_usage` now supports filters (users/room/provider/model/usage_type) and adds `can_use_llm_proxy`.
- CLI adds `auth token` for printing the current access token.
- CLI adds `launch` commands for Codex/Claude plus configuration helpers for proxy profiles/settings; setup wizard now handles LLM proxy access and tool setup.
- CLI container tooling now renders image lists as tables by default and adds `images inspect` with detailed metadata output.

## [0.38.2]
- Image generation tool events now emit binary image results, persist them with metadata to the images database, and redact inline image payloads in event data.
- OpenAI image generation now defaults to `gpt-image-2`, and the pricing catalog adds `gpt-image-2` plus text-embedding-3/ada token rates.
- OpenAI and Anthropic adapters now normalize extra headers and only send `Meshagent-On-Behalf-Of` when a valid name is present.
- Room container models now include a `ports` list in container responses.
- Breaking change: CLI runtime containers no longer inherit Dockerfile `ENV`; environment variables must be provided explicitly.

## [0.38.1]
- Added profile-aware CLI settings with multi-account support, per-profile API URLs, and new `auth switch`/`auth login --api-url` flows, with migration from legacy local state.
- Breaking: `meshagent image` has been replaced by top-level `meshagent build` and `meshagent deploy`, and build/deploy now take the build context as a positional PATH instead of `--pack`.
- Image build/deploy now resolve registry hosts from deployment config or API base URL, normalize shorthand image tags to project registries, and can auto-create missing repositories with permission-aware guidance.
- Registry deletion now accepts repository names or ids with clearer errors.
- Harbor now routes OpenAI adapter calls through the room’s OpenAI endpoint using room tokens and adds extensive diagnostics, including debug event telemetry, build/exec log capture, preserved agent logs, and new terminal-bench audit/report tooling.
- API URL helpers and room connections now respect profile API URL overrides and strip trailing slashes; Python REST/room examples were updated to use CLI token helpers and current messaging callback shapes.
- OpenAI responses adapter now logs unhandled reasoning events at debug level to reduce warning noise.

## [0.38.0]
- Added deployment config models and a `get_config()` API call to retrieve MeshAgent domain/registry settings.
- CLI image build/pack/deploy now validates and targets the configured registry host (from deployment config, with default fallback) instead of a hard‑coded registry host, and uses that registry for credential resolution.

## [0.37.2]
- MailBot and Worker now build room-bound toolkits at startup and cleanly stop/clear hosted toolkits on startup failures and shutdown.

## [0.37.1]
- Transcript schema and livekit transcript logging now include a `participant_role` field for each segment.
- Meshagent CLI join subcommands now preserve storage mount flags and deprecated option aliases when lazy-loaded.
- Scheduled task list/update commands now default the room to `MESHAGENT_ROOM`.
- Harbor environments now fall back to the `meshagent/shell-codex:default` image when no Dockerfile or prebuilt image is provided.
- Harbor environments can delegate the room token into container env and now prefer API-key auth over ambient room tokens.

## [0.37.0]
- Breaking: Database API now supports `json`, `uuid`, `list`, and `struct` types with typed wrappers (DatabaseJson/DatabaseStruct/DatabaseExpression/DatabaseDate/Uuid), and query results/params use structured encoding instead of JSON strings.
- Breaking: Containers build now streams build contexts (start/data chunks) with `mount_path`/`chunks` and removes `start_build`.
- Breaking: Toolkit/hosting refactor removes toolkit config/builders and `supports_context`, introduces room-bound `Toolkit` with public/hidden/client options, and adds LocalRoomTool plus new hosted-toolkit start/stop flow.
- Storage and skills updates: storage toolkits now require explicit mounts (with delete support) and skills prompt generation reads skills through storage mounts.
- Added MCP toolkit support and tool-choice selection across agents and OpenAI/Anthropic adapters, including MCP authorization handling.
- Added agent packaging build/run/deploy workflow with new CLI commands.
- Breaking: Participant tokens now require an LLM grant, include role-based default scopes (LLM/secrets), and preserve extra payload fields.
- Dependency updates across SDK packages: added `aiofiles~=24.1`, `pyyaml~=6.0.2`, and `pathspec>=1.0.3,<2`.

## [0.36.3]
- Breaking: `OpenAIResponsesAdapter`/`OpenAICompletionsAdapter` now take `base_url` in the constructor, default it from `OPENAI_BASE_URL`, and no longer derive provider clients from a `RoomClient`; `meshagent.openai.proxy.get_client` also dropped its `room` parameter.
- Storage client now supports move operations and emits `file.moved` events.
- Secrets client now supports existence checks.
- Project user add calls now omit permission fields unless explicitly set.
- CLI image deploy now supports `--env-secret` and `--meshagent-token` with `--identity` validation, plus Dockerfile `VOLUME` mount and secret existence checks.
- CLI room connect now supports `--env` and `--env-secret` with optional local token minting; API key list and room list outputs now surface active keys and table views.

## [0.36.2]
- Breaking: Removed share-connect API from the Python client (`connect_share` / RoomShareConnectionInfo).
- Added full OAuth scope constants (FULL_OAUTH_SCOPE/FULL_OAUTH_SCOPES) and CLI auth now defaults to requesting the full scope set.
- Added `meshagent room connect` to run local commands with room auth, exporting MeshAgent/OpenAI/Anthropic env vars.
- Default container image for shell/script tools and OpenAI/Slack integrations switched to `python:3.13`.

## [0.36.1]
- Breaking: room internal API port standardized to `8078` with `ROOM_INTERNAL_API_PORT`/`room_api_base_url`, and service port specs now reject reserved room ports.
- Participant container grants now include `ContainerRegistryGrant` for registry list/pull/run/write patterns.
- Containers build APIs accept `optimize_image` (default true) to enable eStargz optimization, and image runtime definitions for node/python were added.
- Room client lifecycle hardened: `wait_for_close` is cancellation-safe, `__aenter__` fails if the connection closes before ready, and `__aexit__` cancels close watchers and fails pending tool streams.
- CLI image deploy now parses packed Dockerfile metadata to infer ports, default HTTP liveness to `/`, validates reserved ports, and supports `meshagent.runtime` runtime overrides with mounted app images; pack preserves Dockerfile/.dockerignore paths.
- CLI container sessions now use the room URL returned by the account service when opening WebSocket connections.
- Skills prompt generation supports location remapping; Harbor agents embed the skills prompt under `/skills` and can optionally delegate `MESHAGENT_TOKEN` into container env.
- Default shell image across CLI/tools/OpenAI/Slack switched to `meshagent/python:default`.

## [0.36.0]
- Added AgentInputContent (text/file), agent email/heartbeat settings, service files, and config mounts to service models.
- Breaking: container API key provisioning was removed from container specs.
- Queue channels now accept structured `content`/`prompt` payloads (including `room://` file references), resolve legacy prompt files, and support thread ID templates with time tokens.
- Container shell tooling now expands config mounts by injecting runtime spec/members files, and the CLI adds `--shell-tool-config-mount` to pass these mounts.
- Service template conversions now include config and empty-dir mounts alongside project/image/file mounts.

## [0.35.8]
- Harbor agent event logs now record turn-start messages (including instructions) before dispatch.
- Python SDK examples updated to use storage upload and call-style webhook handling, with corrected entrypoint guard and cleanup of empty example stubs.

## [0.35.7]
- Breaking: Messaging stream APIs were removed from the Python SDK (stream callbacks, MessageStream types, and `create_stream`); use streaming toolkits instead.
- Breaking: MessageStreamLLMAdapter and ChatKit messaging-stream responses were removed; use streaming toolkits instead.
- OpenAI responses adapter now logs request payloads for websocket errors (especially 4xx) and surfaces nested error messages for debugging.

## [0.35.6]
- Breaking: document runtime no longer falls back to STPyV8; environments must provide the CRDT backend.
- Containers client adds room-storage OCI archive loading with a new import result type; CLI image loading now targets room-storage paths and loads directly into the room.
- Storage upload streaming now uses server-provided `chunk_size` pull headers to drive client chunking.
- CLI now lazily loads subcommands, adds `api-key show`/`api-key env` and interactive setup, and emits OCI zstd layers when packing images.
- New MeshAgent Harbor package provides Harbor-compatible agent/environment classes for running tasks via MeshAgent rooms/containers.
- Shell tooling now supports stopping/deleting cached containers and reuses shell toolkits across turns for process agents.
- Observability improvements add descriptive toolkit/tool span names for tool execution and invocation.
- API key parsing errors now include a key prefix for easier debugging.
- Dependency updates: `pathspec` now `>=1.0.3,<2`, added `zstandard~=0.25.0`, and new Harbor package depends on `harbor>=0.3.0`, `pytest~=8.4`, `pytest-asyncio~=0.26`.

## [0.35.5]
- Shell tooling now supports local process execution and runtime selection between OpenAI, container, or process shells, with unified execution outputs.
- Image deploy updates request-validation and published-port annotations when toggling public/private, using cookie-based validation for private services.

## [0.35.4]
- Added `ApiScope.user_default` to create participant tokens with default user grants.
- Added a managed container toolkit (start/list/stop/run) with strict JSON schemas and structured results for container shell execution.
- CLI now supports `--require-advanced-shell` to expose the managed container toolkit with working-dir support and optional token delegation.
- Breaking: `image build --deploy/--domain` was replaced by `image deploy`, which updates services with mounts (room/project/image/empty-dir), env vars, env-token scopes, and optional domain routing.
- Breaking: `containers exec` removed interactive TTY mode (`--tty` is no longer available).
- CLI help output now hides selected top-level commands, and generated CLI docs omit hidden groups.

## [0.35.3]
- Stability

## [0.35.2]
- Outbound SMTP now includes a resolved local hostname (from `SMTP_LOCAL_HOSTNAME`, system hostname, or `localhost`) to improve mail delivery when the host name is missing.

## [0.35.1]
- Breaking: container build APIs now return a build id and add build lifecycle endpoints (`start_build`, `list_builds`, `cancel_build`, `delete_build`, `get_build_logs`), with optional context-archive fields for packed build contexts.
- Breaking: image builds moved to the new `meshagent image` CLI, which adds `pack` support for OCI archives, room storage uploads, and optional service/route deployment; CLI log output now strips CRI prefixes.
- Chat agent should-reply logic now tolerates unstructured responses; Stagehand availability now requires the SEA binary before reporting ready.

## [0.35.0]
- Meshagent client adds managed secrets with project/room CRUD (base64 payloads), get/list/delete APIs, and external OAuth registration CRUD; list_secrets now resolves managed secret data.
- Agent runtime now formats live turn messages with sender + ISO timestamps via thread adapter hooks, and file-attachment messages include sender context.
- Chat agent shutdown now closes message channels and waits for thread tasks before teardown.
- Diagnostics improved by filtering unsupported computer action arguments and logging tool-call failures with exception details.

## [0.34.0]
- Added a queue listing command that returns room queues in table or JSON format.
- Route upserts now store the originating service ID in route annotations, avoiding unnecessary updates.

## [0.33.3]
- Stability

## [0.33.2]
- Stability

## [0.33.1]
- Stability

## [0.33.0]
- Stability

## [0.32.0]
- Stability

## [0.31.3]
- Stability

## [0.31.2]
- Stability

## [0.31.1]
- Stability

## [0.31.0]
- Stability

## [0.30.1]
- Stability

## [0.30.0]
- Breaking: tool invocation migrated to toolkit-based `room.invoke_tool` with streaming tool-call inputs/outputs, and storage writes now use upload/download streams instead of handle-based APIs.
- Added RoomClient clients for containers, services, and memory plus expanded storage (stat, download URLs, streaming uploads/downloads) and database/search streaming with index management.
- Sync client now streams document state/updates; database and storage APIs accept streaming inputs for large payloads.
- Agents and workers gained threaded task runner indexing, configurable threading/initial-message summarization, and updated responses/completions thread adapters with usage tracking.
- Computer toolkit now uses the `computer` tool type with configurable dimensions/starting URL, optional goto tool, and storage-backed screenshots.
- CLI improvements for storage/containers/database/memory commands (folder-aware `ls`/`cp`/`show`, streaming upload/download, new TUI setup).
- Dependency update: `openai` to `~2.25.0`.

## [0.29.4]
- Stability

## [0.29.3]
- Stability

## [0.29.2]
- Stability

## [0.29.1]
- Stability

## [0.29.0]
- Stability

## [0.28.16]
- Stability

## [0.28.15]
- Stability

## [0.28.14]
- Stability

## [0.28.13]
- Stability

## [0.28.12]
- Stability

## [0.28.11]
- Stability

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
- Shell tool now reuses a long-lived container with a writable root filesystem, runs commands via `bash -lc`, and defaults to the `meshagent/python:default` image.
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
