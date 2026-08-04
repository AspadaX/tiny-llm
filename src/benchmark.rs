use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
    time::Duration,
};

#[derive(Clone, Debug)]
pub(crate) struct TensorFlowStep {
    pub(crate) layer_index: Option<usize>,
    pub(crate) step_name: String,
    pub(crate) input_shape: Vec<usize>,
    pub(crate) output_shape: Vec<usize>,
    pub(crate) elapsed: Duration,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum CompletionReason {
    EndOfSequence,
    UserInterrupted,
}

impl Display for CompletionReason {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EndOfSequence => formatter.write_str("end-of-sequence"),
            Self::UserInterrupted => formatter.write_str("user-interrupted"),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct InferenceStatsSnapshot {
    pub(crate) steps: usize,
    pub(crate) emitted_tokens: usize,
    pub(crate) context_length: usize,
    pub(crate) total: Duration,
    pub(crate) last: Duration,
    pub(crate) time_to_first_token: Duration,
    pub(crate) average: Duration,
    pub(crate) p50: Duration,
    pub(crate) p95: Duration,
    pub(crate) minimum: Duration,
    pub(crate) maximum: Duration,
    pub(crate) overall_tokens_per_second: f64,
    pub(crate) rolling_tokens_per_second: f64,
}

#[derive(Debug)]
pub(crate) struct InferenceStats {
    prompt_tokens: usize,
    emitted_tokens: usize,
    context_length: usize,
    latencies: Vec<Duration>,
    operation_totals: HashMap<String, Duration>,
    completion_reason: Option<CompletionReason>,
}

impl InferenceStats {
    pub(crate) fn new(prompt_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            emitted_tokens: 0,
            context_length: prompt_tokens,
            latencies: Vec::new(),
            operation_totals: HashMap::new(),
            completion_reason: None,
        }
    }

    pub(crate) fn record_prediction(
        &mut self,
        context_length: usize,
        duration: Duration,
        emitted_token: bool,
        steps: &[TensorFlowStep],
    ) {
        self.context_length = context_length;
        self.latencies.push(duration);
        self.emitted_tokens += usize::from(emitted_token);

        for step in steps {
            *self
                .operation_totals
                .entry(step.step_name.clone())
                .or_default() += step.elapsed;
        }
    }

    pub(crate) fn finish(&mut self, reason: CompletionReason) {
        self.completion_reason = Some(reason);
    }

    pub(crate) fn snapshot(&self) -> InferenceStatsSnapshot {
        if self.latencies.is_empty() {
            return InferenceStatsSnapshot {
                context_length: self.context_length,
                ..InferenceStatsSnapshot::default()
            };
        }

        let total: Duration = self.latencies.iter().copied().sum();
        let steps = self.latencies.len();
        let rolling_count = steps.min(5);
        let rolling_total: Duration = self.latencies[steps - rolling_count..]
            .iter()
            .copied()
            .sum();

        InferenceStatsSnapshot {
            steps,
            emitted_tokens: self.emitted_tokens,
            context_length: self.context_length,
            total,
            last: *self.latencies.last().unwrap_or(&Duration::ZERO),
            time_to_first_token: self.latencies[0],
            average: total.div_f64(steps as f64),
            p50: percentile(&self.latencies, 0.50),
            p95: percentile(&self.latencies, 0.95),
            minimum: *self.latencies.iter().min().unwrap_or(&Duration::ZERO),
            maximum: *self.latencies.iter().max().unwrap_or(&Duration::ZERO),
            overall_tokens_per_second: rate(steps, total),
            rolling_tokens_per_second: rate(rolling_count, rolling_total),
        }
    }
}

impl Display for InferenceStats {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        let snapshot = self.snapshot();
        let completion = self
            .completion_reason
            .map_or("unknown".to_string(), |reason| reason.to_string());

        writeln!(formatter, "Inference statistics")?;
        writeln!(formatter, "--------------------")?;
        writeln!(formatter, "Completion:             {completion}")?;
        writeln!(formatter, "Prompt tokens:          {}", self.prompt_tokens)?;
        writeln!(formatter, "Prediction steps:       {}", snapshot.steps)?;
        writeln!(
            formatter,
            "Emitted tokens:         {}",
            snapshot.emitted_tokens
        )?;
        writeln!(
            formatter,
            "Last context length:    {}",
            snapshot.context_length
        )?;

        if snapshot.steps == 0 {
            return writeln!(formatter, "No prediction samples were recorded.");
        }

        writeln!(
            formatter,
            "Time to first token:    {}",
            format_duration(snapshot.time_to_first_token)
        )?;
        writeln!(
            formatter,
            "Total inference time:   {}",
            format_duration(snapshot.total)
        )?;
        writeln!(
            formatter,
            "Average latency:        {}/token",
            format_duration(snapshot.average)
        )?;
        writeln!(
            formatter,
            "Median latency (p50):   {}",
            format_duration(snapshot.p50)
        )?;
        writeln!(
            formatter,
            "P95 latency:            {}",
            format_duration(snapshot.p95)
        )?;
        writeln!(
            formatter,
            "Minimum latency:        {}",
            format_duration(snapshot.minimum)
        )?;
        writeln!(
            formatter,
            "Maximum latency:        {}",
            format_duration(snapshot.maximum)
        )?;
        writeln!(
            formatter,
            "Average throughput:     {:.2} tokens/s",
            snapshot.overall_tokens_per_second
        )?;

        if self.operation_totals.is_empty() {
            return Ok(());
        }

        let instrumented_total: Duration = self.operation_totals.values().copied().sum();
        let mut operations = self.operation_totals.iter().collect::<Vec<_>>();
        operations.sort_by(|left, right| right.1.cmp(left.1));

        writeln!(formatter)?;
        writeln!(formatter, "Operation breakdown")?;
        writeln!(formatter, "-------------------")?;
        writeln!(
            formatter,
            "Instrumented operation totals may overlap and exclude uninstrumented work."
        )?;

        for (name, duration) in operations {
            let percentage = if instrumented_total.is_zero() {
                0.0
            } else {
                duration.as_secs_f64() / instrumented_total.as_secs_f64() * 100.0
            };
            writeln!(
                formatter,
                "{name:<24} {:>10}  {:>5.1}%",
                format_duration(*duration),
                percentage
            )?;
        }

        Ok(())
    }
}

pub(crate) fn format_duration(duration: Duration) -> String {
    if duration.as_secs_f64() >= 1.0 {
        format!("{:.2} s", duration.as_secs_f64())
    } else {
        format!("{:.2} ms", duration.as_secs_f64() * 1_000.0)
    }
}

fn rate(count: usize, duration: Duration) -> f64 {
    if duration.is_zero() {
        0.0
    } else {
        count as f64 / duration.as_secs_f64()
    }
}

fn percentile(samples: &[Duration], quantile: f64) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }

    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = (quantile * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_calculates_latency_statistics() {
        let mut stats = InferenceStats::new(3);
        for milliseconds in [10, 20, 30, 40, 50] {
            stats.record_prediction(3, Duration::from_millis(milliseconds), true, &[]);
        }

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.steps, 5);
        assert_eq!(snapshot.emitted_tokens, 5);
        assert_eq!(snapshot.total, Duration::from_millis(150));
        assert_eq!(snapshot.average, Duration::from_millis(30));
        assert_eq!(snapshot.p50, Duration::from_millis(30));
        assert_eq!(snapshot.p95, Duration::from_millis(50));
        assert_eq!(snapshot.minimum, Duration::from_millis(10));
        assert_eq!(snapshot.maximum, Duration::from_millis(50));
    }

    #[test]
    fn eos_prediction_counts_as_step_but_not_emitted_token() {
        let mut stats = InferenceStats::new(2);
        stats.record_prediction(2, Duration::from_millis(10), false, &[]);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.steps, 1);
        assert_eq!(snapshot.emitted_tokens, 0);
    }
}
