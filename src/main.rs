mod tensors;

use std::{
    f32,
    io::{self, Stdout},
    path::{Path, PathBuf},
    str::FromStr,
    time::{Duration, Instant},
};

use anyhow::{Result, anyhow};
use candle_core::{Shape, Tensor, WithDType, shape::Dim};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use half::{bf16, f16};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, Wrap},
};
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::tensors::TinyTensor;

/*
 * Start of math computations
 */

fn flatten_to_2d(x: &TinyTensor) -> Result<TinyTensor, anyhow::Error> {
    let x_shape = x.get_shape().to_owned().into_dims();
    let in_dim = x_shape.last().unwrap();
    let new_batch_size: usize = x_shape[..x_shape.len() - 1].iter().product();

    Ok(reshape(x, (new_batch_size, *in_dim))?)
}

fn bloat_back_to_original_dimension(
    weights: &TinyTensor,
    original_x: &TinyTensor,
    matrix_multiplied_x: TinyTensor,
) -> Result<TinyTensor, anyhow::Error> {
    let out_dim = weights.get_shape().dim(0)?;
    let original_shape = original_x.get_shape().dims();
    let mut new_shape = original_shape[..original_shape.len() - 1].to_vec();
    new_shape.push(out_dim);

    Ok(reshape(&matrix_multiplied_x, new_shape)?)
}

/// result = w * x + b
///
/// but bias, b, can be omitted in LLaMA models
pub fn compute_linear_layer(
    weights: &TinyTensor,
    x: &TinyTensor,
    bias: Option<&TinyTensor>,
) -> Result<TinyTensor> {
    // Flatten x into a 2D tensor
    let flattened_x = flatten_to_2d(x)?;

    // Transpose the weight,
    // [out_dim, in_dim] becomes [in_dim, out_dim]
    let transposed = transpose(weights)?;

    // Matrix multiplication
    let matrix_multiplied = matrix_multiply(&flattened_x, &transposed)?;

    // Reshape x back to the original dimension
    let x = bloat_back_to_original_dimension(weights, x, matrix_multiplied)?;

    // Add bias if any
    Ok(match bias {
        Some(bias) => matrix_add(&x, bias)?,
        None => x,
    })
}

/// Calculates the RoPE rotation angle θ.
///
/// θ = m * 10000^(-2j / d_head)
///
/// `m = token_position`, `j = pair_position`, `d_head` = feature dimension per attention head.
pub fn calculate_theta(
    token_position: usize,
    pair_position: usize,
    d_head: usize,
    rope_theta: f32,
) -> f32 {
    token_position as f32 * f32::powf(rope_theta, (-2.0 * pair_position as f32) / d_head as f32)
}

/// Precomputes cos/sin lookup tables for fast RoPE application.
///
/// For every token position and dimension pair:
///
/// cos = cos(m * 10000^(-2j / d_head))
/// sin = sin(m * 10000^(-2j / d_head))
///
/// Linear table index: `index = token_position * num_pairs + pair_position`
/// where `num_pairs = d_head / 2`.
///
/// Return (cos_table, sin_table)
///
/// Paper: https://arxiv.org/pdf/2104.09864
pub fn precompute_theta_tables(
    max_sequence_len: usize,
    d_head: usize,
    rope_theta: f32,
) -> Result<(TinyTensor, TinyTensor)> {
    let num_pairs = d_head / 2;
    let mut cos_table = Vec::with_capacity(max_sequence_len * num_pairs);
    let mut sin_table = Vec::with_capacity(max_sequence_len * num_pairs);

    // token_position is the m
    for token_position in 0..max_sequence_len {
        // pair_position is the j, computed by d_head / 2
        for pair_position in 0..num_pairs {
            let theta = calculate_theta(token_position, pair_position, d_head, rope_theta);
            let cos = f32::cos(theta);
            let sin = f32::sin(theta);

            cos_table.push(cos);
            sin_table.push(sin);
        }
    }

    let table_shape = [1, 1, max_sequence_len, d_head / 2];

    Ok((
        TinyTensor::new(&cos_table, &table_shape)?,
        TinyTensor::new(&sin_table, &table_shape)?,
    ))
}

/// Applies Rotary Position Embedding to a single attention head vector.
///
/// Rotates each adjacent 2D dimension pair with the precomputed angle:
///
/// x' = x * cos(θ) - y * sin(θ)
/// y' = x * sin(θ) + y * cos(θ)
pub fn compute_rotary_position_embeddings(
    input_tensor: &TinyTensor,
    cos_tensor: &TinyTensor,
    sin_tensor: &TinyTensor,
) -> Result<TinyTensor> {
    let last_dimension = input_tensor.rank() - 1; // The last dimension is where the head vectors sit
    let d_head = input_tensor.get_shape().dim(last_dimension)?;
    let pair_size = d_head / 2;

    let x = narrow(input_tensor, last_dimension, 0, pair_size)?;
    let y = narrow(input_tensor, last_dimension, pair_size, pair_size)?;

    let x_cos = broadcast_multiply(&x, cos_tensor)?;
    let y_sin = broadcast_multiply(&y, sin_tensor)?;

    let x_sin = broadcast_multiply(&x, sin_tensor)?;
    let y_cos = broadcast_multiply(&y, cos_tensor)?;

    let x = broadcast_substract(&x_cos, &y_sin)?;
    let y = broadcast_add(&x_sin, &y_cos)?;

    Ok(concatenate(&x, &y, last_dimension)?)
}

pub fn prepare_rope_for_this_step(
    current_position: usize,
    current_sequence_length: usize,
    cos_tensor: &TinyTensor,
    sin_tensor: &TinyTensor,
) -> Result<(TinyTensor, TinyTensor)> {
    Ok((
        narrow(cos_tensor, 2, current_position, current_sequence_length)?,
        narrow(sin_tensor, 2, current_position, current_sequence_length)?,
    ))
}

/// Root Mean Square Normalization (RMSNorm), an improved version of LayerNorm.
/// It reduces the computaion complexity by focusing on the re-scaling part of the original algorithm,
/// thus being more efficient.
///
/// In LLaMA's paper, they used RMSNorm instead as an optimization.
///
/// Paper: https://arxiv.org/pdf/1910.07467
pub fn compute_rms_norm(
    input_tensor: &TinyTensor,
    weights: &TinyTensor,
    epsilon: Option<f32>,
) -> Result<TinyTensor> {
    let epsilon = TinyTensor::new(&[epsilon.unwrap_or(1e-6)], &[1])?;

    let hidden_dimensions = input_tensor.rank() - 1;

    let squared = square(input_tensor)?;
    let mean = broadcast_add(&mean(&squared, hidden_dimensions)?, &epsilon)?;
    let square_root = square_root(&mean)?;

    let divided = &broadcast_divide(input_tensor, &square_root)?;

    Ok(broadcast_multiply(&divided, weights)?)
}

/// Attention-based LLMs predict the next token using only the tokens that have
/// already been generated or provided as input.
///
/// For example, given the sentence "Today is ...", when the model is predicting
/// the token after "is", it should only attend to the previous tokens,
/// "Today is". If the model can attend to future tokens during training, it can
/// leak information from the target sequence.
///
/// To prevent the model from attending to future tokens, we use a causal
/// attention mask. Future positions are set to negative infinity, so after the
/// mask is added to the attention scores and softmax is applied, those positions
/// receive probability 0.
///
/// For a sequence length of 4, the additive attention mask looks like:
/// -------------------------
/// 0    -inf -inf -inf
/// 0     0   -inf -inf
/// 0     0    0   -inf
/// 0     0    0    0
/// -------------------------
///
/// During training, the full sequence is known and processed in parallel, so the
/// mask prevents each position from seeing later positions. During generation,
/// future tokens have not been generated yet; when decoding one token at a time,
/// the current slice of the mask may contain only valid positions, but it follows
/// the same causal rule.
///
/// Attention mask shape:
/// &[1, 1, max_sequence_len, max_sequence_len]
pub fn create_attention_mask(max_sequence_len: usize) -> Result<TinyTensor> {
    let mut raw_mask: Vec<f32> = vec![0.0; max_sequence_len * max_sequence_len];
    let row_column_range = 0..max_sequence_len;

    for i in row_column_range.clone().into_iter() {
        for j in row_column_range.clone().into_iter() {
            if j > i {
                raw_mask[max_sequence_len * i + j] = f32::NEG_INFINITY;
            }
        }
    }

    Ok(TinyTensor::new(
        &raw_mask,
        &[1, 1, max_sequence_len, max_sequence_len],
    )?)
}

pub fn compute_current_attention_mask(
    attention_mask: &TinyTensor,
    current_token_position: usize,
) -> Result<TinyTensor> {
    // Slice on rows.
    // Only 1 row is needed.
    let tensor = narrow(attention_mask, 2, current_token_position, 1)?;

    // Slice on columns.
    // All columns until the current token are needed.
    narrow(&tensor, 3, 0, current_token_position + 1)
}

/// Scaled dot-product attention for one multi-head attention block.
///
/// In a Transformer, Q, K, V, and O are learned projection weights during training.
/// During inference, hidden states are the model's internal vector representations
/// of the current token sequence.
///
/// Hidden states initially come from token embeddings, then each Transformer layer
/// updates them using attention and feed-forward computations. They represent the
/// input tokens in context, not the predicted tokens directly.
///
/// For autoregressive generation, the model predicts one next token at a time.
/// After a token is selected, its token ID is appended to the input sequence and
/// passed through the model on the next step. With KV caching, the model can reuse
/// previously computed key/value tensors instead of recomputing the whole context.
///
/// Q/K/V/O are names from the retrieval analogy.
/// Mathematically they are learned weight matrices.
/// During training, the model learns whatever values reduce the loss,
/// but each matrix is constrained by its position in the computation graph.
///
/// Scaled dot-product attention computes:
///
/// attention(Q, K, V) = softmax((QK^T) / sqrt(head_dim) + mask) V
///
/// Tensor shape:
/// [batch size, num heads, sequence length, head dimension]
pub fn compute_scaled_dot_product_attention(
    q: &TinyTensor,
    k: &TinyTensor,
    v: &TinyTensor,
    current_attention_mask: Option<&TinyTensor>,
) -> Result<TinyTensor> {
    let square_root_k_dimension =
        f32::sqrt(k.get_shape().clone().dims().last().unwrap().to_owned() as f32);
    let tensor_sqrt_k_dimension = TinyTensor::new(&[square_root_k_dimension], &[1])?;

    let q_k = matrix_multiply(q, &transpose(k)?)?;
    let divided = broadcast_divide(&q_k, &tensor_sqrt_k_dimension)?;

    let applied_attention_mask = if let Some(attention_mask) = current_attention_mask {
        broadcast_add(&divided, attention_mask)?
    } else {
        divided
    };

    let softmaxed = softmax(&applied_attention_mask)?;

    Ok(matrix_multiply(&softmaxed, v)?)
}

/// Multi-head attention using already-projected Q, K, and V tensors.
///
/// This follows the Transformer attention pattern from "Attention Is All You Need":
///
///     MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
///
/// where each head is:
///
///     head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
///
/// In this implementation, the Q/K/V projections have already been applied before
/// this function is called. The input tensors are already shaped as:
///
///     [batch_size, num_heads, sequence_length, head_dim]
///
/// Therefore, `compute_scaled_dot_product_attention` computes all heads in parallel.
/// The result is then transposed to:
///
///     [batch_size, sequence_length, num_heads, head_dim]
///
/// and flattened back to:
///
///     [batch_size, sequence_length, hidden_dim]
///
/// Finally, the output projection `W_O` mixes information across heads and returns
/// the attention output in the model's hidden-state dimension.
///
/// Paper: https://arxiv.org/pdf/1706.03762
pub fn compute_multi_head_attention(
    q: &TinyTensor,
    k: &TinyTensor,
    v: &TinyTensor,
    weights: &TinyTensor, // Shape: [hidden_dim, hidden_dim]
    current_attention_mask: &TinyTensor,
) -> Result<TinyTensor> {
    let heads = compute_scaled_dot_product_attention(q, k, v, Some(current_attention_mask))?;

    let concatenated = transpose_with_dim(&heads, 1, 2)?;

    // Recover the output back to hidden state
    let flattened = flatten(&concatenated, 2, 3)?;

    // Ok(matrix_multiply(&flattened, weights)?)
    Ok(compute_linear_layer(weights, &flattened, None)?)
}

/// This is to align the shapes when using GQA
pub fn align_to_q(
    num_attnetion_heads: usize,
    num_kv_heads: usize,
    k: &TinyTensor,
    v: &TinyTensor,
) -> Result<(TinyTensor, TinyTensor)> {
    let num_groups = num_attnetion_heads / num_kv_heads;

    Ok((repeat_kv(k, num_groups)?, repeat_kv(v, num_groups)?))
}

/// SWISH: A SELF-GATED ACTIVATION FUNCTION: https://arxiv.org/pdf/1710.05941v1
/// GLU Variants Improve Transformer: https://arxiv.org/pdf/2002.05202
///
/// The `transformers` library implementation of swiglu swapped the `gate_projection` and `up`.
/// Therefore, when loading llama model weights, we will need to plugin the up to gate and so on.
pub fn compute_swiglu(
    hidden_state: &TinyTensor,    // x
    gate_projection: &TinyTensor, // V
    up_projection: &TinyTensor,   // W
    down_projection: &TinyTensor, // W2
) -> Result<TinyTensor> {
    // The matrix multiplication here uses linear, as it is mathematically identical without a bias,
    // and the linear implementation takes care of the dimensional differences.
    let gate = compute_linear_layer(gate_projection, hidden_state, None)?;

    let up = compute_linear_layer(up_projection, hidden_state, None)?;

    let activated_gate = silu(&gate)?;

    let apply_gate = broadcast_multiply(&activated_gate, &up)?;

    // Ok(matrix_multiply(&apply_gate, down_projection)?)
    Ok(compute_linear_layer(down_projection, &apply_gate, None)?)
}

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

/*
 * Start of UI Logics
 */

#[derive(Clone, Debug)]
struct TensorFlowStep {
    layer_index: Option<usize>,
    step_name: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    elapsed: Duration,
}

#[derive(Clone, Debug)]
struct CandidateLogit {
    token_id: u32,
    decoded_text: String,
    logit: f32,
}

#[derive(Clone, Debug)]
struct AttentionHeatmapSnapshot {
    layer_index: usize,
    head_index: usize,
    values: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, Default)]
struct InferenceDebugState {
    prompt: String,
    generated_text: String,
    current_token_id: Option<u32>,
    current_token_text: String,
    tensor_flow_steps: Vec<TensorFlowStep>,
    attention_heatmaps: Vec<AttentionHeatmapSnapshot>,
    candidate_logits: Vec<CandidateLogit>,
}

struct PredictionResult {
    next_token: u32,
    debug_state: InferenceDebugState,
}

type DebugTerminal = Terminal<CrosstermBackend<Stdout>>;

struct TerminalSession {
    terminal: DebugTerminal,
}

impl TerminalSession {
    fn start() -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let terminal = Terminal::new(CrosstermBackend::new(stdout))?;

        Ok(Self { terminal })
    }

    fn draw(&mut self, debug_state: &InferenceDebugState) -> Result<()> {
        self.terminal
            .draw(|frame| render_inference_debugger(frame, debug_state))?;
        Ok(())
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

fn shape_to_string(shape: &[usize]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn tensor_shape(tensor: &TinyTensor) -> Vec<usize> {
    tensor.get_shape().dims().to_vec()
}

fn elapsed_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
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

fn should_quit_tui() -> Result<bool> {
    if event::poll(Duration::from_millis(1))? {
        if let Event::Key(key_event) = event::read()? {
            return Ok(matches!(key_event.code, KeyCode::Char('q') | KeyCode::Esc));
        }
    }

    Ok(false)
}

fn render_inference_debugger(frame: &mut ratatui::Frame, debug_state: &InferenceDebugState) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),
            Constraint::Min(12),
            Constraint::Length(10),
        ])
        .split(frame.area());

    render_generated_text(frame, root[0], debug_state);

    let middle = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
        .split(root[1]);

    render_attention_heatmap(frame, middle[0], debug_state);
    render_candidate_logits(frame, middle[1], debug_state);
    render_tensor_flow(frame, root[2], debug_state);
}

fn render_generated_text(
    frame: &mut ratatui::Frame,
    area: Rect,
    debug_state: &InferenceDebugState,
) {
    let status_line = Line::from(vec![
        Span::styled(
            "q / Esc",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" quit  "),
        Span::styled("token", Style::default().fg(Color::Cyan)),
        Span::raw(format!(" {:?} ", debug_state.current_token_id)),
        Span::styled(
            debug_state.current_token_text.clone(),
            Style::default().fg(Color::Green),
        ),
    ]);

    let text = vec![
        status_line,
        Line::from(vec![
            Span::styled("Prompt: ", Style::default().fg(Color::Magenta)),
            Span::raw(debug_state.prompt.clone()),
        ]),
        Line::from(vec![
            Span::styled("Generated: ", Style::default().fg(Color::Cyan)),
            Span::raw(debug_state.generated_text.clone()),
        ]),
    ];

    let paragraph = Paragraph::new(text)
        .block(
            Block::default()
                .title(" MiniCPM inference debugger ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue)),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

fn render_attention_heatmap(
    frame: &mut ratatui::Frame,
    area: Rect,
    debug_state: &InferenceDebugState,
) {
    if debug_state.attention_heatmaps.is_empty() {
        let paragraph = Paragraph::new("attention values are not available yet").block(
            Block::default()
                .title(" Attention heatmaps ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Red)),
        );
        frame.render_widget(paragraph, area);
        return;
    };

    let heatmap_count = debug_state.attention_heatmaps.len();
    let columns = (heatmap_count as f64).sqrt().ceil() as usize;
    let rows = heatmap_count.div_ceil(columns);
    let row_constraints = vec![Constraint::Ratio(1, rows as u32); rows];
    let row_areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints(row_constraints)
        .split(area);

    for row_index in 0..rows {
        let start = row_index * columns;
        let end = usize::min(start + columns, heatmap_count);
        let column_count = end - start;
        let column_constraints = vec![Constraint::Ratio(1, column_count as u32); column_count];
        let column_areas = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(column_constraints)
            .split(row_areas[row_index]);

        for (column_index, heatmap) in debug_state.attention_heatmaps[start..end]
            .iter()
            .enumerate()
        {
            render_single_attention_heatmap(frame, column_areas[column_index], heatmap);
        }
    }
}

fn render_single_attention_heatmap(
    frame: &mut ratatui::Frame,
    area: Rect,
    heatmap: &AttentionHeatmapSnapshot,
) {
    let mut lines = Vec::new();

    for row in heatmap
        .values
        .iter()
        .rev()
        .take(area.height.saturating_sub(2) as usize)
        .rev()
    {
        let spans = row
            .iter()
            .take(area.width.saturating_sub(2) as usize)
            .map(|value| attention_value_span(*value));
        lines.push(Line::from(spans.collect::<Vec<_>>()));
    }

    let paragraph = Paragraph::new(lines).block(
        Block::default()
            .title(format!(
                " L{} H{} ",
                heatmap.layer_index, heatmap.head_index
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Magenta)),
    );

    frame.render_widget(paragraph, area);
}

fn attention_value_span(value: f32) -> Span<'static> {
    let (symbol, color) = if value < 0.05 {
        ("░", Color::DarkGray)
    } else if value < 0.15 {
        ("▒", Color::Blue)
    } else if value < 0.35 {
        ("▓", Color::Cyan)
    } else if value < 0.65 {
        ("█", Color::Yellow)
    } else {
        ("█", Color::Red)
    };

    Span::styled(symbol, Style::default().fg(color))
}

fn render_candidate_logits(
    frame: &mut ratatui::Frame,
    area: Rect,
    debug_state: &InferenceDebugState,
) {
    let rows = debug_state.candidate_logits.iter().map(|candidate| {
        Row::new(vec![
            Cell::from(candidate.token_id.to_string()).style(Style::default().fg(Color::Cyan)),
            Cell::from(candidate.decoded_text.replace('\n', "\\n"))
                .style(Style::default().fg(Color::Green)),
            Cell::from(format!("{:.3}", candidate.logit)).style(Style::default().fg(Color::Yellow)),
        ])
    });

    let table = Table::new(
        rows,
        [
            Constraint::Length(8),
            Constraint::Min(8),
            Constraint::Length(12),
        ],
    )
    .header(
        Row::new(vec!["token", "text", "logit"]).style(
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ),
    )
    .block(
        Block::default()
            .title(" Candidate logits ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green)),
    );

    frame.render_widget(table, area);
}

fn render_tensor_flow(frame: &mut ratatui::Frame, area: Rect, debug_state: &InferenceDebugState) {
    let visible_rows = area.height.saturating_sub(3) as usize;
    let steps = debug_state
        .tensor_flow_steps
        .iter()
        .rev()
        .take(visible_rows)
        .collect::<Vec<_>>();

    let rows = steps.into_iter().rev().map(|step| {
        let layer = step
            .layer_index
            .map_or("global".to_string(), |index| format!("layer {index}"));
        Row::new(vec![
            Cell::from(layer).style(Style::default().fg(Color::Blue)),
            Cell::from(step.step_name.clone()).style(Style::default().fg(Color::Cyan)),
            Cell::from(shape_to_string(&step.input_shape))
                .style(Style::default().fg(Color::DarkGray)),
            Cell::from("→").style(
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ),
            Cell::from(shape_to_string(&step.output_shape))
                .style(Style::default().fg(Color::Green)),
            Cell::from(format!("{:.2} ms", elapsed_ms(step.elapsed)))
                .style(Style::default().fg(Color::Yellow)),
        ])
    });

    let table = Table::new(
        rows,
        [
            Constraint::Length(9),
            Constraint::Length(20),
            Constraint::Percentage(25),
            Constraint::Length(2),
            Constraint::Percentage(25),
            Constraint::Length(12),
        ],
    )
    .header(
        Row::new(vec!["layer", "step", "input", "", "output", "time"]).style(
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ),
    )
    .block(
        Block::default()
            .title(" Tensor shape flow ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow)),
    );

    frame.render_widget(table, area);
}

fn tiny_tensor_to_f32_vec(tensor: &TinyTensor) -> Result<Vec<f32>> {
    Ok(tensor.inner.flatten_all()?.to_vec1::<f32>()?)
}

fn collect_top_candidate_logits(
    logits: &TinyTensor,
    tokenizer: &Tokenizer,
    top_k: usize,
) -> Result<Vec<CandidateLogit>> {
    let flattened_logits = tiny_tensor_to_f32_vec(logits)?;
    let mut indexed_logits = flattened_logits
        .iter()
        .enumerate()
        .map(|(token_id, logit)| (token_id as u32, *logit))
        .collect::<Vec<_>>();

    indexed_logits.sort_by(|(_, left), (_, right)| {
        right.partial_cmp(left).unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(indexed_logits
        .into_iter()
        .take(top_k)
        .map(|(token_id, logit)| CandidateLogit {
            token_id,
            decoded_text: tokenizer
                .decode(&[token_id], false)
                .unwrap_or_else(|_| "<?>".to_string()),
            logit,
        })
        .collect())
}

fn build_attention_heatmaps(
    layer_index: usize,
    q: &TinyTensor,
    k: &TinyTensor,
    attention_mask: &TinyTensor,
) -> Result<Vec<AttentionHeatmapSnapshot>> {
    let square_root_k_dimension =
        f32::sqrt(k.get_shape().clone().dims().last().unwrap().to_owned() as f32);
    let tensor_sqrt_k_dimension = TinyTensor::new(&[square_root_k_dimension], &[1])?;
    let q_k = matrix_multiply(q, &transpose(k)?)?;
    let divided = broadcast_divide(&q_k, &tensor_sqrt_k_dimension)?;
    let applied_attention_mask = broadcast_add(&divided, attention_mask)?;
    let softmaxed = softmax(&applied_attention_mask)?;

    let shape = softmaxed.get_shape();
    let num_heads = shape.dim(1)?;
    let sequence_length = shape.dim(2)?;
    let all_attention_values = tiny_tensor_to_f32_vec(&softmaxed)?;
    let mut heatmaps = Vec::with_capacity(num_heads);

    for head_index in 0..num_heads {
        let mut values = Vec::with_capacity(sequence_length);

        for query_position in 0..sequence_length {
            let mut row = Vec::with_capacity(sequence_length);
            for key_position in 0..sequence_length {
                let index = ((head_index * sequence_length + query_position) * sequence_length)
                    + key_position;
                row.push(all_attention_values.get(index).copied().unwrap_or(0.0));
            }
            values.push(row);
        }

        heatmaps.push(AttentionHeatmapSnapshot {
            layer_index,
            head_index,
            values,
        });
    }

    Ok(heatmaps)
}

/*
 * Start of the main loop
 */

pub fn parse_arguments() -> Result<(String, String)> {
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
    let mut debug_state = InferenceDebugState {
        prompt: prompt.to_string(),
        generated_text: generated_text.to_string(),
        ..InferenceDebugState::default()
    };
    // Wrap the token IDs in a tensor so that the embedding table can be indexed with them.
    // At this point, the tensor shape is [num_token_ids].
    // e.g., for 10 token IDs, the shape is [10], with each element a token ID.
    let token_ids_tensor = TinyTensor::new_without_shape(input_token_ids)?;

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
    let mut hidden_state = unsqueeze(&hidden_state, 0)?;
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
    let max_sequence_length = hidden_state.get_shape().dim(1)?;
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
                &q,
                (
                    1,
                    max_sequence_length,
                    model_configurations.num_attention_heads,
                    model_configurations.head_dim,
                ),
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
                &k,
                (
                    1,
                    max_sequence_length,
                    model_configurations.num_key_value_heads,
                    model_configurations.head_dim,
                ),
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
                &v,
                (
                    1,
                    max_sequence_length,
                    model_configurations.num_key_value_heads,
                    model_configurations.head_dim,
                ),
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
            let q = transpose_with_dim(&q, 1, 2)?;
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
            let k = transpose_with_dim(&k, 1, 2)?;
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
            let mut v = transpose_with_dim(&v, 1, 2)?;
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
                debug_state.attention_heatmaps =
                    build_attention_heatmaps(index, &q, &k, &attention_mask)?;
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
    let sliced = narrow(&normalized_hidden_state, 1, max_sequence_length - 1, 1)?;
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
    debug_state.candidate_logits = collect_top_candidate_logits(&logits, tokenizer, 10)?;

    let argmax = argmax(&logits, 2)?;

    let next_token: u32 = reshape(&argmax, ())?.to_scalar()?;

    Ok(PredictionResult {
        next_token,
        debug_state,
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

    println!("Model loaded!");

    let tokens = tokenizer.encode(prompt.clone(), true).unwrap();

    let mut input_token_ids = tokens.get_ids().to_vec();
    let mut generated_text = String::new();
    let mut terminal_session = TerminalSession::start()?;

    // An LLM predicts the next token from the input text, one token at a time.
    // Since we want a full response rather than a single token, we keep generating
    // tokens until the model emits an end-of-sequence token.
    loop {
        let mut prediction = predict_next_token(
            &model_configurations,
            &llama_model,
            &tokenizer,
            &prompt,
            &generated_text,
            &input_token_ids,
        )?;

        // Append the generated token to the "context"
        input_token_ids.push(prediction.next_token);

        let word = tokenizer.decode(&[prediction.next_token], false).unwrap();
        prediction.debug_state.current_token_id = Some(prediction.next_token);
        prediction.debug_state.current_token_text = word.clone();

        // Exit when the model says done
        if model_configurations
            .eos_token_id
            .contains(&prediction.next_token)
        {
            prediction.debug_state.generated_text = generated_text.clone();
            terminal_session.draw(&prediction.debug_state)?;
            break;
        }

        generated_text.push_str(&word);
        prediction.debug_state.generated_text = generated_text.clone();
        terminal_session.draw(&prediction.debug_state)?;

        if should_quit_tui()? {
            break;
        }
    }

    Ok(())
}
