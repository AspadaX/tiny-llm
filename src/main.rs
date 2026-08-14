mod algorithms;
mod benchmark;
mod inference;
mod kv_cache;
mod simd;
mod tensors;
mod tui;

use std::{
    f32,
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};

use anyhow::{Result, anyhow};
use safetensors::SafeTensors;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    benchmark::{CompletionReason, InferenceStats, TensorFlowStep},
    inference::{compute_input_tensor, predict_next_token_with_kv_cache},
    kv_cache::KVCache,
    tensors::TinyTensor,
    tui::{InferenceDebugState, TerminalSession, should_quit},
};

/// This is entirely loaded from `config.json`
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelConfigurations {
    #[serde(rename = "_name_or_path")]
    #[serde(default)]
    pub name_or_path: String,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub pad_token_id: u32,
    #[serde(default)]
    pub hidden_act: String,
    #[serde(default)]
    pub intermediate_size: usize,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub torch_dtype: String,
    #[serde(default)]
    pub transformers_version: String,
    #[serde(default)]
    pub use_cache: bool,

    /*
     * Share weights
     */
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /*
     * Generation
     */
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default, deserialize_with = "deserialize_eos_token_id")]
    pub eos_token_id: Vec<u32>,

    /*
     * Normalization
     */
    #[serde(default)]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: Option<Value>,
    #[serde(default)]
    pub max_position_embeddings: usize,

    /*
     * Attention
     */
    #[serde(default)]
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: usize,

    /*
     * Shapes
     */
    #[serde(default)]
    pub hidden_size: usize,
    #[serde(default)]
    pub initializer_range: f32,
    #[serde(default)]
    pub vocab_size: usize,
}

// Custom deserializer that accepts both `0` and `[0]`
fn deserialize_eos_token_id<'de, D>(deserializer: D) -> Result<Vec<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    let value: Value = Deserialize::deserialize(deserializer)?;
    match value {
        Value::Null => Ok(Vec::new()),
        Value::Number(n) => Ok(vec![n.as_u64().unwrap() as u32]),
        Value::Array(arr) => {
            let mut token_ids = Vec::new();

            for n in arr {
                token_ids.push(n.as_u64().unwrap() as u32);
            }

            Ok(token_ids)
        }
        _ => Ok(Vec::new()),
    }
}

fn default_rope_theta() -> f32 {
    10000.0 // Default for LLaMA 1/2/SmolLM/TinyLlama. LLaMA 3 uses 500000.0
}

impl ModelConfigurations {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }
}

pub struct TransformerBlock {
    /*
     * Attention
     */
    pub q_projection: TinyTensor,
    pub k_projection: TinyTensor,
    pub v_projection: TinyTensor,
    pub o_projection: TinyTensor,

    /*
     * MLP
     */
    pub gate_projection: TinyTensor,
    pub up_projection: TinyTensor,
    pub down_projection: TinyTensor,

    /*
     * Normalization
     */
    pub input_layernorm: TinyTensor,
    pub post_attention_norm: TinyTensor,
}

impl TransformerBlock {
    pub fn load_transformer_blocks(safetensors: &SafeTensors, layer_index: usize) -> Result<Self> {
        let layer_prefix = format!("model.layers.{}", layer_index);

        Ok(TransformerBlock {
            // Attention
            q_projection: TinyTensor::load_weight(
                safetensors,
                &format!("{}.self_attn.q_proj.weight", layer_prefix),
            )?,
            k_projection: TinyTensor::load_weight(
                safetensors,
                &format!("{}.self_attn.k_proj.weight", layer_prefix),
            )?,
            v_projection: TinyTensor::load_weight(
                safetensors,
                &format!("{}.self_attn.v_proj.weight", layer_prefix),
            )?,
            o_projection: TinyTensor::load_weight(
                safetensors,
                &format!("{}.self_attn.o_proj.weight", layer_prefix),
            )?,

            // MLP (SwiGLU)
            gate_projection: TinyTensor::load_weight(
                safetensors,
                &format!("{}.mlp.gate_proj.weight", layer_prefix),
            )?,
            up_projection: TinyTensor::load_weight(
                safetensors,
                &format!("{}.mlp.up_proj.weight", layer_prefix),
            )?,
            down_projection: TinyTensor::load_weight(
                safetensors,
                &format!("{}.mlp.down_proj.weight", layer_prefix),
            )?,

            // Norms
            input_layernorm: TinyTensor::load_weight(
                safetensors,
                &format!("{}.input_layernorm.weight", layer_prefix),
            )?,
            post_attention_norm: TinyTensor::load_weight(
                safetensors,
                &format!("{}.post_attention_layernorm.weight", layer_prefix),
            )?,
        })
    }
}

pub struct LlamaModel {
    /// This usually shares the weight with `embedding_tokens`
    pub lm_head: TinyTensor,
    /// This usually shares the weight with `lm_head`
    pub embedding_tokens: Option<TinyTensor>,
    pub norm: TinyTensor,
    pub layers: Vec<TransformerBlock>,
}

impl LlamaModel {
    pub fn load_from_configurations(
        configurations: &ModelConfigurations,
        safetensors: &SafeTensors,
    ) -> Result<Self> {
        let lm_head: TinyTensor = TinyTensor::load_weight(safetensors, "lm_head.weight")?;
        let embedding_tokens: Option<TinyTensor> = if configurations.tie_word_embeddings {
            None
        } else {
            Some(TinyTensor::load_weight(
                safetensors,
                "model.embed_tokens.weight",
            )?)
        };
        let norm = TinyTensor::load_weight(safetensors, "model.norm.weight")?;

        let mut layers = Vec::new();

        for layer_index in 0..configurations.num_hidden_layers {
            layers.push(TransformerBlock::load_transformer_blocks(
                safetensors,
                layer_index,
            )?);
        }

        Ok(Self {
            lm_head,
            norm,
            embedding_tokens,
            layers,
        })
    }
}

struct PredictionResult {
    next_token: u32,
    debug_state: InferenceDebugState,
    visualization_overhead: std::time::Duration,
}

fn tensor_shape(tensor: &TinyTensor) -> Vec<usize> {
    tensor.get_shape().to_vec()
}

fn record_tensor_flow_step(
    debug_state: &mut InferenceDebugState,
    layer_index: Option<usize>,
    step_name: impl Into<String>,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    started_at: Instant,
) {
    debug_state.tensor_flow_steps.push(TensorFlowStep {
        layer_index,
        step_name: step_name.into(),
        input_shape,
        output_shape,
        elapsed: started_at.elapsed(),
    });
}

/*
 * Start of the main loop
 */

fn parse_arguments() -> Result<(String, String)> {
    let mut model_dir: String = String::new();
    let mut prompt: String = String::new();

    for (index, argument) in std::env::args().into_iter().enumerate() {
        if index == 1 {
            model_dir.push_str(&argument);
        }

        if index == 2 {
            prompt.push_str(&argument);
        }
    }

    if prompt.is_empty() || model_dir.is_empty() {
        return Err(anyhow!("You should supply both model directory and prompt"));
    }

    Ok((model_dir, prompt))
}

fn main() -> Result<()> {
    let (model_dir, prompt) = parse_arguments()?;

    // A tokenizer converts input text into a sequence of token IDs.
    // Each token ID represents a piece of the text, such as a word,
    // subword, character, or punctuation mark.
    // This numeric representation allows the model to process the text.
    //
    // For example, the sentence "Today is sunny." might be converted into
    // token IDs like [12840, 374, 27737, 13].
    let tokenizer = Tokenizer::from_file(
        PathBuf::from_str(&model_dir)
            .unwrap()
            .join("tokenizer.json"),
    )
    .unwrap();

    println!("Tokenizer loaded!");

    // A model is a collection of trained matrices, and inference is a sequence of matrix operations.
    //
    // We save these matrices to a file so others can use the model. The model stays
    // in memory until the inference engine shuts down.
    let buffer = std::fs::read(format!("{}/model-00000-of-00001.safetensors", model_dir))?;

    let safetensors = SafeTensors::deserialize(&buffer)?;
    let model_configurations = ModelConfigurations::load(format!("{}/config.json", model_dir))?;

    let llama_model = LlamaModel::load_from_configurations(&model_configurations, &safetensors)?;

    // Safetensors is no longer needed in the memory.
    // Removing it will reduce the memory footprint.
    drop(safetensors);

    println!("Model loaded!");

    let tokens = tokenizer.encode(prompt.clone(), true).unwrap();

    let mut prefill_input_token_ids = tokens.get_ids().to_vec();
    let mut generated_text = String::new();
    let mut stats = InferenceStats::new(prefill_input_token_ids.len());
    let mut terminal_session = TerminalSession::start()?;

    let mut kv_cache = KVCache::new();
    let mut generation_input_token_ids = Vec::new();

    // An LLM predicts the next token from the input text, one token at a time.
    // Since we want a full response rather than a single token, we keep generating
    // tokens until the model emits an end-of-sequence token.
    loop {
        let mut debug_state = InferenceDebugState {
            prompt: prompt.to_string(),
            generated_text: generated_text.to_string(),
            ..InferenceDebugState::default()
        };

        let has_prefilled = prefill_input_token_ids.is_empty();

        let (input_tensor, input_token_ids) = match has_prefilled {
            true => (
                compute_input_tensor(
                    &model_configurations,
                    &llama_model,
                    &generation_input_token_ids,
                    &mut debug_state,
                )?,
                std::mem::take(&mut generation_input_token_ids),
            ),
            false => {
                let prefill_input_tensor = compute_input_tensor(
                    &model_configurations,
                    &llama_model,
                    &prefill_input_token_ids,
                    &mut debug_state,
                )?;

                (
                    prefill_input_tensor,
                    std::mem::take(&mut prefill_input_token_ids),
                )
            }
        };

        let context_length = input_token_ids.len();
        let inference_started_at = Instant::now();
        let mut prediction = predict_next_token_with_kv_cache(
            &model_configurations,
            &llama_model,
            &tokenizer,
            debug_state,
            input_tensor,
            &mut kv_cache,
        )?;
        let inference_duration = inference_started_at
            .elapsed()
            .saturating_sub(prediction.visualization_overhead);
        let is_eos = model_configurations
            .eos_token_id
            .contains(&prediction.next_token);

        stats.record_prediction(
            context_length,
            inference_duration,
            !is_eos,
            &prediction.debug_state.tensor_flow_steps,
        );

        // Append the generated token to the "context"
        generation_input_token_ids.push(prediction.next_token);

        let word = tokenizer.decode(&[prediction.next_token], false).unwrap();
        prediction.debug_state.current_token_id = Some(prediction.next_token);
        prediction.debug_state.current_token_text = word.clone();

        if is_eos {
            stats.finish(CompletionReason::EndOfSequence);
            prediction.debug_state.generated_text = generated_text.clone();
            prediction.debug_state.benchmark = stats.snapshot();
            terminal_session.draw(&prediction.debug_state)?;
            break;
        }

        generated_text.push_str(&word);
        prediction.debug_state.generated_text = generated_text.clone();
        prediction.debug_state.benchmark = stats.snapshot();
        terminal_session.draw(&prediction.debug_state)?;

        if should_quit()? {
            stats.finish(CompletionReason::UserInterrupted);
            break;
        }
    }

    drop(terminal_session);
    println!("{stats}");

    Ok(())
}
