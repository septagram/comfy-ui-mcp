use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde_json::Value;
use std::time::Duration;
use tracing::{debug, info};

pub struct ComfyClient {
    client: Client,
    base_url: String,
}

impl ComfyClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// List available checkpoint models.
    pub async fn list_checkpoints(&self) -> Result<Vec<String>> {
        let url = format!("{}/models/checkpoints", self.base_url);
        let resp: Vec<String> = self.client.get(&url).send().await?.json().await?;
        Ok(resp)
    }

    /// Get system stats (GPU, VRAM, versions).
    pub async fn system_stats(&self) -> Result<Value> {
        let url = format!("{}/system_stats", self.base_url);
        let resp: Value = self.client.get(&url).send().await?.json().await?;
        Ok(resp)
    }

    /// Submit a workflow prompt and return the prompt_id.
    pub async fn submit_prompt(&self, workflow: &Value) -> Result<String> {
        let url = format!("{}/prompt", self.base_url);
        let body = serde_json::json!({ "prompt": workflow });
        let resp: Value = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .json()
            .await?;

        resp["prompt_id"]
            .as_str()
            .map(String::from)
            .context("No prompt_id in response")
    }

    /// Poll history until the prompt completes. Returns the output node's images info.
    pub async fn wait_for_completion(
        &self,
        prompt_id: &str,
        timeout: Duration,
    ) -> Result<Vec<ImageOutput>> {
        let url = format!("{}/history/{}", self.base_url, prompt_id);
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > timeout {
                bail!("Timed out waiting for image generation after {:?}", timeout);
            }

            let resp: Value = self.client.get(&url).send().await?.json().await?;

            if let Some(entry) = resp.get(prompt_id) {
                // Check for error
                if let Some(status) = entry.get("status") {
                    if status.get("status_str").and_then(|s| s.as_str()) == Some("error") {
                        let msgs = status.get("messages").cloned().unwrap_or(Value::Null);
                        bail!("ComfyUI execution error: {msgs}");
                    }
                }

                // Look for image outputs in any node
                if let Some(outputs) = entry.get("outputs").and_then(|o| o.as_object()) {
                    for (_node_id, node_output) in outputs {
                        if let Some(images) = node_output.get("images").and_then(|i| i.as_array())
                        {
                            let result: Vec<ImageOutput> = images
                                .iter()
                                .filter_map(|img| {
                                    Some(ImageOutput {
                                        filename: img.get("filename")?.as_str()?.to_string(),
                                        subfolder: img
                                            .get("subfolder")
                                            .and_then(|s| s.as_str())
                                            .unwrap_or("")
                                            .to_string(),
                                        type_: img
                                            .get("type")
                                            .and_then(|s| s.as_str())
                                            .unwrap_or("output")
                                            .to_string(),
                                    })
                                })
                                .collect();

                            if !result.is_empty() {
                                info!(count = result.len(), "Generation complete");
                                return Ok(result);
                            }
                        }
                    }
                }
            }

            debug!("Still waiting for prompt {prompt_id}...");
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    /// Fetch the raw image bytes from ComfyUI.
    pub async fn fetch_image(&self, output: &ImageOutput) -> Result<Vec<u8>> {
        let url = format!(
            "{}/view?filename={}&subfolder={}&type={}",
            self.base_url,
            urlencoding::encode(&output.filename),
            urlencoding::encode(&output.subfolder),
            urlencoding::encode(&output.type_),
        );
        let bytes = self.client.get(&url).send().await?.bytes().await?;
        Ok(bytes.to_vec())
    }
}

#[derive(Debug)]
pub struct ImageOutput {
    pub filename: String,
    pub subfolder: String,
    pub type_: String,
}

/// Build a txt2img workflow from the given parameters.
pub fn build_txt2img_workflow(
    checkpoint: &str,
    prompt: &str,
    negative_prompt: &str,
    width: u32,
    height: u32,
    steps: u32,
    cfg: f64,
    sampler: &str,
    seed: i64,
) -> Value {
    serde_json::json!({
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": checkpoint
            }
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["1", 1]
            }
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["1", 1]
            }
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": "normal",
                "denoise": 1.0
            }
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2]
            }
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["6", 0],
                "filename_prefix": "mcp_gen"
            }
        }
    })
}
