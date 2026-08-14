use std::{
    io::{self, Stdout},
    time::Duration,
};

use anyhow::Result;
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
use tokenizers::Tokenizer;

use crate::{
    benchmark::{InferenceStatsSnapshot, TensorFlowStep, format_duration},
    tensors::{
        TinyTensor, broadcast_add, broadcast_divide, make_contiguous_data, matrix_multiply,
        softmax, transpose,
    },
};

#[derive(Clone, Debug)]
pub(crate) struct CandidateLogit {
    token_id: u32,
    decoded_text: String,
    logit: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct AttentionHeatmapSnapshot {
    layer_index: usize,
    head_index: usize,
    values: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct InferenceDebugState {
    pub(crate) prompt: String,
    pub(crate) generated_text: String,
    pub(crate) current_token_id: Option<u32>,
    pub(crate) current_token_text: String,
    pub(crate) tensor_flow_steps: Vec<TensorFlowStep>,
    pub(crate) attention_heatmaps: Vec<AttentionHeatmapSnapshot>,
    pub(crate) candidate_logits: Vec<CandidateLogit>,
    pub(crate) benchmark: InferenceStatsSnapshot,
}

type DebugTerminal = Terminal<CrosstermBackend<Stdout>>;

pub(crate) struct TerminalSession {
    terminal: DebugTerminal,
}

impl TerminalSession {
    pub(crate) fn start() -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let terminal = Terminal::new(CrosstermBackend::new(stdout))?;

        Ok(Self { terminal })
    }

    pub(crate) fn draw(&mut self, debug_state: &InferenceDebugState) -> Result<()> {
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

pub(crate) fn should_quit() -> Result<bool> {
    if !event::poll(Duration::from_millis(1))? {
        return Ok(false);
    }

    Ok(matches!(
        event::read()?,
        Event::Key(key_event) if matches!(key_event.code, KeyCode::Char('q') | KeyCode::Esc)
    ))
}

pub(crate) fn collect_top_candidate_logits(
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

pub(crate) fn build_attention_heatmaps(
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
    let query_length = shape[2];
    let key_length = shape[3];
    let all_attention_values = tiny_tensor_to_f32_vec(&softmaxed)?;
    let mut heatmaps = Vec::with_capacity(num_heads);

    for head_index in 0..num_heads {
        let mut values = Vec::with_capacity(query_length);

        for query_position in 0..query_length {
            let mut row = Vec::with_capacity(key_length);
            for key_position in 0..key_length {
                let index =
                    ((head_index * query_length + query_position) * key_length) + key_position;
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

fn tiny_tensor_to_f32_vec(tensor: &TinyTensor) -> Result<Vec<f32>> {
    make_contiguous_data(tensor.clone())
}

fn render_inference_debugger(frame: &mut ratatui::Frame, debug_state: &InferenceDebugState) {
    let benchmark_height = if frame.area().width >= 120 { 7 } else { 12 };
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),
            Constraint::Min(12),
            Constraint::Length(benchmark_height),
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
    render_benchmark(frame, root[2], debug_state);
    render_tensor_flow(frame, root[3], debug_state);
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

fn render_benchmark(frame: &mut ratatui::Frame, area: Rect, debug_state: &InferenceDebugState) {
    let benchmark = &debug_state.benchmark;
    let context_length = debug_state
        .attention_heatmaps
        .first()
        .and_then(|heatmap| heatmap.values.first())
        .map_or(benchmark.context_length, Vec::len);
    let card_areas = benchmark_card_areas(area);

    render_metric_card(
        frame,
        card_areas[0],
        " Session ",
        Color::Blue,
        vec![
            metric_line("Steps", benchmark.steps.to_string(), Color::Cyan),
            metric_line("Tokens", benchmark.emitted_tokens.to_string(), Color::Green),
            metric_line("Context", format!("{} tok", context_length), Color::Magenta),
            metric_line("Elapsed", format_duration(benchmark.total), Color::Yellow),
        ],
    );
    render_metric_card(
        frame,
        card_areas[1],
        " Latency ",
        Color::Yellow,
        vec![
            metric_line("Last", format_duration(benchmark.last), Color::Yellow),
            metric_line("Average", format_duration(benchmark.average), Color::Cyan),
            metric_line(
                "TTFT",
                format_duration(benchmark.time_to_first_token),
                Color::Green,
            ),
        ],
    );
    render_metric_card(
        frame,
        card_areas[2],
        " Throughput ",
        Color::Green,
        vec![
            metric_line(
                "Overall",
                format!("{:.2} tok/s", benchmark.overall_tokens_per_second),
                Color::Green,
            ),
            metric_line(
                "Latest 5",
                format!("{:.2} tok/s", benchmark.rolling_tokens_per_second),
                Color::LightGreen,
            ),
        ],
    );
    render_metric_card(
        frame,
        card_areas[3],
        " Distribution ",
        Color::Magenta,
        vec![
            metric_line("P50", format_duration(benchmark.p50), Color::Cyan),
            metric_line("P95", format_duration(benchmark.p95), Color::Magenta),
            metric_line("Minimum", format_duration(benchmark.minimum), Color::Green),
            metric_line("Maximum", format_duration(benchmark.maximum), Color::Red),
        ],
    );
}

fn benchmark_card_areas(area: Rect) -> Vec<Rect> {
    if area.width >= 120 {
        return Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ])
            .split(area)
            .to_vec();
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[0]);
    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[1]);

    vec![top[0], top[1], bottom[0], bottom[1]]
}

fn render_metric_card(
    frame: &mut ratatui::Frame,
    area: Rect,
    title: &'static str,
    color: Color,
    lines: Vec<Line<'static>>,
) {
    let card = Paragraph::new(lines).block(
        Block::default()
            .title(title)
            .title_style(Style::default().fg(color).add_modifier(Modifier::BOLD))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(color)),
    );
    frame.render_widget(card, area);
}

fn metric_line(label: &'static str, value: String, color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<9}"), Style::default().fg(Color::DarkGray)),
        Span::styled(
            value,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
    ])
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
            .rev()
            .take(area.width.saturating_sub(2) as usize)
            .rev()
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
            Cell::from(format!("{:.2} ms", step.elapsed.as_secs_f64() * 1_000.0))
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
