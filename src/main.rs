#![feature(
    str_as_str,
    error_generic_member_access,
    custom_inner_attributes,
    proc_macro_hygiene,
    const_option_ops,
    try_blocks,
    min_specialization,
)]

mod comfy;
mod mask;
mod mcp;
mod metadata;
mod setup;
mod tools;

use comfy::ComfyClient;
use mcp::{
    CallToolResult, ContentItem, InitializeResult, JsonRpcRequest, JsonRpcResponse,
    ServerCapabilities, ServerInfo, ToolsCapability, ToolsListResult,
};
use serde_json::Value;
use std::io::Write;
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::{debug, error, info};

fn main() -> anyhow::Result<()> {
    match std::env::args().nth(1).as_deref() {
        Some("--register") => return setup::register_mcp(),
        Some("--unregister") => return setup::unregister_mcp(),
        Some("--version") | Some("-V") => {
            println!("comfy-ui-mcp {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        _ => {}
    }
    serve()
}

#[tokio::main]
async fn serve() -> anyhow::Result<()> {
    // Load .env from the binary's directory or current dir
    let _ = dotenvy::dotenv();

    // Tracing → stderr only, never stdout
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "comfy_ui_mcp=info".parse().unwrap()),
        )
        .init();

    let comfy_url =
        std::env::var("COMFYUI_URL").unwrap_or_else(|_| "http://127.0.0.1:8188".into());
    info!(
        version = env!("CARGO_PKG_VERSION"),
        comfyui_url = %comfy_url,
        "comfy-ui-mcp starting (stdio JSON-RPC; upstream ComfyUI at comfyui_url)"
    );

    let client = ComfyClient::new(&comfy_url);

    // Single writer task drains the channel and serializes all stdout writes.
    // Multiple handler tasks can produce responses concurrently without
    // interleaving bytes into the JSON-RPC stream.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<JsonRpcResponse>();
    let writer = tokio::spawn(async move {
        while let Some(resp) = rx.recv().await {
            send_response(&resp);
        }
    });

    let stdin = tokio::io::stdin();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        debug!(raw = %line, "Received");

        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = JsonRpcResponse::error(None, -32700, format!("Parse error: {e}"));
                let _ = tx.send(resp);
                continue;
            }
        };

        // Spawn the handler so a long-running tool call (e.g. a multi-minute
        // generation) doesn't block the stdin read loop. Without this, the
        // host can time out and close our stdio, killing the process.
        let client = client.clone();
        let tx = tx.clone();
        tokio::spawn(async move {
            if let Some(resp) = handle_request(&client, &request).await {
                let _ = tx.send(resp);
            }
        });
    }

    info!("stdin closed, shutting down");
    // Drop the original sender so the writer task can drain remaining
    // responses from still-running handlers and then exit cleanly.
    drop(tx);
    let _ = writer.await;
    Ok(())
}

async fn handle_request(client: &ComfyClient, req: &JsonRpcRequest) -> Option<JsonRpcResponse> {
    let id = req.id.clone();

    match req.method.as_str() {
        // ── Lifecycle ──────────────────────────────────────────────────
        "initialize" => {
            let result = InitializeResult {
                protocol_version: "2024-11-05".into(),
                capabilities: ServerCapabilities {
                    tools: ToolsCapability {},
                },
                server_info: ServerInfo {
                    name: "comfy-ui-mcp".into(),
                    version: env!("CARGO_PKG_VERSION").into(),
                },
            };
            Some(JsonRpcResponse::success(
                id,
                serde_json::to_value(result).unwrap(),
            ))
        }

        // Notification — no response
        "notifications/initialized" => {
            info!("Client initialized");
            None
        }

        // ── Tools ──────────────────────────────────────────────────────
        "tools/list" => {
            let result = ToolsListResult {
                tools: tools::tool_definitions(),
            };
            Some(JsonRpcResponse::success(
                id,
                serde_json::to_value(result).unwrap(),
            ))
        }

        "tools/call" => {
            let params = req.params.as_ref().cloned().unwrap_or(Value::Null);
            let tool_name = params
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or(Value::Object(Default::default()));

            info!(tool = %tool_name, "Tool call");

            match tools::handle_tool_call(client, tool_name, &arguments).await {
                Ok(result) => Some(JsonRpcResponse::success(
                    id,
                    serde_json::to_value(result).unwrap(),
                )),
                Err(e) => {
                    error!(tool = %tool_name, error = %e, "Tool call failed");
                    let error_result = CallToolResult {
                        content: vec![ContentItem::Text {
                            text: format!("Error: {e:#}"),
                        }],
                        is_error: true,
                    };
                    Some(JsonRpcResponse::success(
                        id,
                        serde_json::to_value(error_result).unwrap(),
                    ))
                }
            }
        }

        // ── Unknown ────────────────────────────────────────────────────
        method => {
            debug!(method, "Unknown method");
            // Notifications (no id) get no response
            if id.is_none() || id.as_ref() == Some(&Value::Null) {
                None
            } else {
                Some(JsonRpcResponse::error(
                    id,
                    -32601,
                    format!("Method not found: {method}"),
                ))
            }
        }
    }
}

fn send_response(resp: &JsonRpcResponse) {
    let json = match serde_json::to_string(resp) {
        Ok(s) => s,
        Err(e) => {
            error!(error = %e, "Failed to serialize response");
            return;
        }
    };
    let mut stdout = std::io::stdout().lock();
    if let Err(e) = writeln!(stdout, "{json}") {
        // Broken pipe (host closed our stdio) or similar. Drop the response
        // rather than panicking — we'll exit naturally on the next stdin read.
        error!(error = %e, "Failed to write to stdout");
        return;
    }
    if let Err(e) = stdout.flush() {
        error!(error = %e, "Failed to flush stdout");
        return;
    }
    debug!(json = %json, "Sent");
}
