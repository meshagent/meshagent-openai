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
