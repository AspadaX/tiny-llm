pub mod algorithms;
pub mod tensors;

use std::{
    f32,
    io::{self, Stdout},
    path::{Path, PathBuf},
    str::FromStr,
    time::{Duration, Instant},
};

use anyhow::{Result, anyhow};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, Wrap},
};
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
    tensors::{
        TinyTensor, argmax, broadcast_add, broadcast_divide, make_contiguous_data, matrix_multiply,
        narrow, reshape, select_index, softmax, transpose, transpose_with_dim, unsqueeze,
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
    tensor.get_shape().to_vec()
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
    make_contiguous_data(tensor.clone())
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
    let square_root_k_dimension = k.get_shape()[k.rank() - 1] as f32;
    let tensor_sqrt_k_dimension =
        TinyTensor::new_from_vec(vec![square_root_k_dimension.sqrt()], &[1])?;
    let q_k = matrix_multiply(q, &transpose(k.clone())?)?;
    let divided = broadcast_divide(&q_k, &tensor_sqrt_k_dimension)?;
    let applied_attention_mask = broadcast_add(&divided, attention_mask)?;
    let softmaxed = softmax(&applied_attention_mask, applied_attention_mask.rank() - 1)?;

    let shape = softmaxed.get_shape();
    let num_heads = shape[1];
    let sequence_length = shape[2];
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
    debug_state.candidate_logits = collect_top_candidate_logits(&logits, tokenizer, 10)?;

    let argmax = argmax(&logits, 2)?;

    let next_token = argmax.to_scalar()? as u32;

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

    // Safetensors is no longer needed in the memory.
    // Removing it will reduce the memory footprint.
    drop(safetensors);

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
