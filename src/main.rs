mod algorithms;
mod benchmark;
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
    algorithms::{
        align_to_q, compute_linear_layer, compute_multi_head_attention, compute_rms_norm,
        compute_rotary_position_embeddings, compute_swiglu, create_attention_mask,
        precompute_theta_tables,
    },
    benchmark::{CompletionReason, InferenceStats, TensorFlowStep},
    tensors::{
        TinyTensor, argmax, broadcast_add, narrow, reshape, select_index, transpose_with_dim,
        unsqueeze,
    },
    tui::{
        InferenceDebugState, TerminalSession, build_attention_heatmaps,
        collect_top_candidate_logits, should_quit,
    },
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

fn predict_next_token(
    model_configurations: &ModelConfigurations,
    llama_model: &LlamaModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    generated_text: &str,
    input_token_ids: &[u32],
) -> Result<PredictionResult, anyhow::Error> {
    let mut visualization_overhead = std::time::Duration::ZERO;
    let mut debug_state = InferenceDebugState {
        prompt: prompt.to_string(),
        generated_text: generated_text.to_string(),
        ..InferenceDebugState::default()
    };
    // Wrap the token IDs in a tensor so that the embedding table can be indexed with them.
    // At this point, the tensor shape is [num_token_ids].
    // e.g., for 10 token IDs, the shape is [10], with each element a token ID.
    let token_ids = input_token_ids
        .iter()
        .map(|&token_id| token_id as f32)
        .collect::<Vec<_>>();
    let token_ids_length = token_ids.len();
    let token_ids_tensor = TinyTensor::new_from_vec(token_ids, &[token_ids_length])?;

    let embedding_started_at = Instant::now();
    // Convert token IDs to initial hidden state via embedding table lookup.
    // After this, the tensor shape will be [num_tokens, hidden_size]
    // e.g., for 10 tokens and hidden_size = 2048, the shape is [10, 2048].
    let hidden_state = match &llama_model.embedding_tokens {
        Some(result) => select_index(&token_ids_tensor, result, 0)?,
        None => select_index(&token_ids_tensor, &llama_model.lm_head, 0)?,
    };
    record_tensor_flow_step(
        &mut debug_state,
        None,
        "token embedding",
        tensor_shape(&token_ids_tensor),
        tensor_shape(&hidden_state),
        embedding_started_at,
    );

    let unsqueeze_started_at = Instant::now();
    // Add an additional dimension to the hidden state tensor at dimension 0.
    // This is because the model needs to have the batch size information in the tensor.
    // Now the tensor will be: [batch_size, num_tokens, hidden_size],
    // for example, [10, 2048] becomes [1, 10, 2048]
    //
    // Notice the batch size is 1. This is because we are doing a single sequence inference.
    // That is, we put all input tokens into 1 batch.
    //
    // Also, here num_tokens is the sequence length of this inference batch.
    let mut hidden_state = unsqueeze(hidden_state, 0)?;
    record_tensor_flow_step(
        &mut debug_state,
        None,
        "add batch dim",
        vec![input_token_ids.len(), model_configurations.hidden_size],
        tensor_shape(&hidden_state),
        unsqueeze_started_at,
    );
    // Remember we mentioned above that the num_tokens is also the sequence length?
    // Here, we are accessing the second dimension of the hidden state to get the sequence length / number of tokens.
    // The reason for calling this one a max sequence length is because we are only doing single sequence inference,
    // so the current sequence length naturally becomes the max length.
    //
    // This needed for the attention mask and marking the token positions in positional embeddings.
    let max_sequence_length = hidden_state.get_shape()[1];
    // We will use this later to mark the token position info.
    let (cos_table, sin_table) = precompute_theta_tables(
        max_sequence_length,
        model_configurations.head_dim,
        model_configurations.rope_theta,
    )?;

    // Create an attention mask with the max sequence length we got from above.
    let attention_mask = create_attention_mask(max_sequence_length)?;

    // Pass the hidden state through each transformer layer.
    // The number of layers is defined by the model architecture in the configuration.
    // Each layer contains its own Q, K, V, O weights, plus an MLP,
    // and applies attention followed by a residual connection.
    // The final hidden state (after all layers) will be projected to logits.
    //
    // A residual connection means the input to a sublayer is added to its output (skip connection).
    // This helps with gradient flow and lets the model learn incremental transformations.
    for index in 0..model_configurations.num_hidden_layers {
        if let Some(layer) = llama_model.layers.get(index) {
            // The full‑sequence attention mask (computed above) already covers
            // every token position, so we don't need to slice it yet.
            // (During token‑by‑token generation we would narrow the mask to
            //  the current position.)

            let input_norm_started_at = Instant::now();
            // RMSNorm: Normalize each hidden vector to unit root mean square (RMS).
            // This keeps activations well‑scaled, preventing runaway values.
            // (Values typically remain within a few units, rather than exploding to 10 or 100.)
            //
            // The original Transformer used LayerNorm (mean subtraction + RMS scaling).
            // LLaMA uses RMSNorm, which drops the mean subtraction, reducing computation
            // while still providing effective normalization.
            //
            // After normalization, a learned weight vector (input_layernorm) scales each
            // hidden dimension. This weight is trained along with the model.
            //
            // Epsilon prevents division by zero when the RMS is extremely small.
            let normalized_hidden_state = compute_rms_norm(
                &hidden_state,
                &layer.input_layernorm,
                Some(model_configurations.rms_norm_eps),
            )?;
            record_tensor_flow_step(
                &mut debug_state,
                Some(index),
                "input rms norm",
                tensor_shape(&hidden_state),
                tensor_shape(&normalized_hidden_state),
                input_norm_started_at,
            );

            let q_projection_started_at = Instant::now();
            // Refer to `compute_multi_head_attention` and `compute_scaled_dot_product_attention`
            // for QKVO explanations.
            //
            // The shape of QKV weight matrices is [hidden_size, hidden_size],
            // where the first hidden_size marks the output matrix's hidden_size
            // and the second hidden_size marks the input matrix's hidden_size
            //
            // The shape of QKV after projection will become [batch_size, num_tokens, hidden_size].
            // Notice that LLaMA models usually don't include a bias.
            //
            // After the projection, it will outp
            let q = compute_linear_layer(&layer.q_projection, &normalized_hidden_state, None)?;
            record_tensor_flow_step(
                &mut debug_state,
                Some(index),
                "q projection",
                tensor_shape(&normalized_hidden_state),
                tensor_shape(&q),
                q_projection_started_at,
            );

            let k_projection_started_at = Instant::now();
            let k = compute_linear_layer(&layer.k_projection, &normalized_hidden_state, None)?;
            record_tensor_flow_step(
                &mut debug_state,
                Some(index),
                "k projection",
                tensor_shape(&normalized_hidden_state),
                tensor_shape(&k),
                k_projection_started_at,
            );

            let v_projection_started_at = Instant::now();
            let v = compute_linear_layer(&layer.v_projection, &normalized_hidden_state, None)?;
            record_tensor_flow_step(
                &mut debug_state,
                Some(index),
                "v projection",
                tensor_shape(&normalized_hidden_state),
                tensor_shape(&v),
                v_projection_started_at,
            );

            // Reshape the QKV from 3D matrices: [batch_size, num_tokens, hidden_size]
            // To: [batch, sequence_length, num_attention_heads, head_dim]
            // where the hidden_size is split into num_attention_heads and head_dim.
            //
            // num_attention_heads: The number of attention heads used when computing attentions.
            // head_dim: The size of each head.
            let q_shape_before_reshape = tensor_shape(&q);
            let q_reshape_started_at = Instant::now();
            let q = reshape(
                q,
                &[
                    1,
                    max_sequence_length,
                    model_configurations.num_attention_heads,
                    model_configurations.head_dim,
                ],
            )?;
            record_tensor_flow_step(
                &mut debug_state,
                Some(index),
                "q reshape",
                q_shape_before_reshape,
                tensor_shape(&q),
                q_reshape_started_at,
            );

            let k_shape_before_reshape = tensor_shape(&k);
            let k_reshape_started_at = Instant::now();
            let k = reshape(
                k,
                &[
                    1,
                    max_sequence_length,
                    model_configurations.num_key_value_heads,
                    model_configurations.head_dim,
                ],
            )?;
            record_tensor_flow_step(
                &mut debug_state,
                Some(index),
                "k reshape",
                k_shape_before_reshape,
                tensor_shape(&k),
                k_reshape_started_at,
            );

            let v_shape_before_reshape = tensor_shape(&v);
            let v_reshape_started_at = Instant::now();
            let v = reshape(
                v,
                &[
                    1,
                    max_sequence_length,
                    model_configurations.num_key_value_heads,
                    model_configurations.head_dim,
                ],
            )?;
            record_tensor_flow_step(
                &mut debug_state,
                Some(index),
                "v reshape",
                v_shape_before_reshape,
                tensor_shape(&v),
                v_reshape_started_at,
            );

            // Change the shape
            // from [batch_size, num_tokens, num_heads, head_dim]
            // to [batch_size, num_heads, num_tokens, head_dim]
            //
            // We basically swapped the position of num_tokens with num_heads
            // to match the shape required when computing attentions.
            let q_shape_before_transpose = tensor_shape(&q);
            let q_transpose_started_at = Instant::now();
            let q = transpose_with_dim(q, 1, 2)?;
            record_tensor_flow_step(
                &mut debug_state,
                Some(index),
                "q transpose",
                q_shape_before_transpose,
                tensor_shape(&q),
                q_transpose_started_at,
            );

            let k_shape_before_transpose = tensor_shape(&k);
            let k_transpose_started_at = Instant::now();
            let k = transpose_with_dim(k, 1, 2)?;
            record_tensor_flow_step(
                &mut debug_state,
                Some(index),
                "k transpose",
                k_shape_before_transpose,
                tensor_shape(&k),
                k_transpose_started_at,
            );

            let v_shape_before_transpose = tensor_shape(&v);
            let v_transpose_started_at = Instant::now();
            let mut v = transpose_with_dim(v, 1, 2)?;
            record_tensor_flow_step(
                &mut debug_state,
                Some(index),
                "v transpose",
                v_shape_before_transpose,
                tensor_shape(&v),
                v_transpose_started_at,
            );

            // For each token position, RoPE rotates pairs of adjacent dimensions
            // (x, y) in the head vector by an angle derived from the token index.
            // This encodes absolute position into relative attention scores,
            // so that the model is aware of the semantic difference between having
            // a word appears earlier vs later in a sentence.
            let q_shape_before_rope = tensor_shape(&q);
            let q_rope_started_at = Instant::now();
            let q = compute_rotary_position_embeddings(&q, &cos_table, &sin_table)?;
            record_tensor_flow_step(
                &mut debug_state,
                Some(index),
                "q rope",
                q_shape_before_rope,
                tensor_shape(&q),
                q_rope_started_at,
            );

            let k_shape_before_rope = tensor_shape(&k);
            let k_rope_started_at = Instant::now();
            let mut k = compute_rotary_position_embeddings(&k, &cos_table, &sin_table)?;
            record_tensor_flow_step(
                &mut debug_state,
                Some(index),
                "k rope",
                k_shape_before_rope,
                tensor_shape(&k),
                k_rope_started_at,
            );

            // Apply Groupped Attention Query, when number of KV heads does not match Q's.
            // Paper: https://arxiv.org/pdf/2305.13245
            if model_configurations.num_attention_heads != model_configurations.num_key_value_heads
            {
                let align_started_at = Instant::now();
                let k_shape_before_align = tensor_shape(&k);
                let v_shape_before_align = tensor_shape(&v);
                (k, v) = align_to_q(
                    model_configurations.num_attention_heads,
                    model_configurations.num_key_value_heads,
                    &k,
                    &v,
                )?;
                record_tensor_flow_step(
                    &mut debug_state,
                    Some(index),
                    "gqa align k",
                    k_shape_before_align,
                    tensor_shape(&k),
                    align_started_at,
                );
                record_tensor_flow_step(
                    &mut debug_state,
                    Some(index),
                    "gqa align v",
                    v_shape_before_align,
                    tensor_shape(&v),
                    align_started_at,
                );
            }

            if index == model_configurations.num_hidden_layers.saturating_sub(1) {
                let visualization_started_at = Instant::now();
                debug_state.attention_heatmaps =
                    build_attention_heatmaps(index, &q, &k, &attention_mask)?;
                visualization_overhead += visualization_started_at.elapsed();
            }

            let attention_started_at = Instant::now();
            let attention =
                compute_multi_head_attention(&q, &k, &v, &layer.o_projection, &attention_mask)?;
            record_tensor_flow_step(
                &mut debug_state,
                Some(index),
                "attention",
                tensor_shape(&q),
                tensor_shape(&attention),
                attention_started_at,
            );

            // Update hidden state with the newly calculated attention.
            // This is residual connection.
            let residual_attention_started_at = Instant::now();
            let hidden_shape_before_attention_residual = tensor_shape(&hidden_state);
            hidden_state = broadcast_add(&hidden_state, &attention)?;
            record_tensor_flow_step(
                &mut debug_state,
                Some(index),
                "attention residual",
                hidden_shape_before_attention_residual,
                tensor_shape(&hidden_state),
                residual_attention_started_at,
            );

            let post_attention_norm_started_at = Instant::now();
            let normalized_hidden_state = compute_rms_norm(
                &hidden_state,
                &layer.post_attention_norm,
                Some(model_configurations.rms_norm_eps),
            )?;
            record_tensor_flow_step(
                &mut debug_state,
                Some(index),
                "post attn norm",
                tensor_shape(&hidden_state),
                tensor_shape(&normalized_hidden_state),
                post_attention_norm_started_at,
            );

            let swiglu_started_at = Instant::now();
            let swiglu = compute_swiglu(
                &normalized_hidden_state,
                &layer.gate_projection,
                &layer.up_projection,
                &layer.down_projection,
            )?;
            record_tensor_flow_step(
                &mut debug_state,
                Some(index),
                "swiglu mlp",
                tensor_shape(&normalized_hidden_state),
                tensor_shape(&swiglu),
                swiglu_started_at,
            );

            let residual_mlp_started_at = Instant::now();
            let hidden_shape_before_mlp_residual = tensor_shape(&hidden_state);
            hidden_state = broadcast_add(&hidden_state, &swiglu)?;
            record_tensor_flow_step(
                &mut debug_state,
                Some(index),
                "mlp residual",
                hidden_shape_before_mlp_residual,
                tensor_shape(&hidden_state),
                residual_mlp_started_at,
            );
        }
    }
    let final_norm_started_at = Instant::now();
    let normalized_hidden_state = compute_rms_norm(
        &hidden_state,
        &llama_model.norm,
        Some(model_configurations.rms_norm_eps),
    )?;
    record_tensor_flow_step(
        &mut debug_state,
        None,
        "final rms norm",
        tensor_shape(&hidden_state),
        tensor_shape(&normalized_hidden_state),
        final_norm_started_at,
    );

    let slice_started_at = Instant::now();
    let sliced = narrow(
        normalized_hidden_state.clone(),
        1,
        max_sequence_length - 1,
        1,
    )?;
    record_tensor_flow_step(
        &mut debug_state,
        None,
        "last token slice",
        tensor_shape(&normalized_hidden_state),
        tensor_shape(&sliced),
        slice_started_at,
    );

    let logits_started_at = Instant::now();
    let logits = compute_linear_layer(&llama_model.lm_head, &sliced, None)?;
    record_tensor_flow_step(
        &mut debug_state,
        None,
        "lm head logits",
        tensor_shape(&sliced),
        tensor_shape(&logits),
        logits_started_at,
    );
    let visualization_started_at = Instant::now();
    debug_state.candidate_logits = collect_top_candidate_logits(&logits, tokenizer, 10)?;
    visualization_overhead += visualization_started_at.elapsed();

    let argmax = argmax(&logits, 2)?;

    let next_token = argmax.to_scalar()? as u32;

    Ok(PredictionResult {
        next_token,
        debug_state,
        visualization_overhead,
    })
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

    let mut input_token_ids = tokens.get_ids().to_vec();
    let mut generated_text = String::new();
    let mut stats = InferenceStats::new(input_token_ids.len());
    let mut terminal_session = TerminalSession::start()?;

    // An LLM predicts the next token from the input text, one token at a time.
    // Since we want a full response rather than a single token, we keep generating
    // tokens until the model emits an end-of-sequence token.
    loop {
        let context_length = input_token_ids.len();
        let inference_started_at = Instant::now();
        let mut prediction = predict_next_token(
            &model_configurations,
            &llama_model,
            &tokenizer,
            &prompt,
            &generated_text,
            &input_token_ids,
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
        input_token_ids.push(prediction.next_token);

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
