use std::time::Instant;

use anyhow::Result;
use tokenizers::Tokenizer;

use crate::{
    LlamaModel, ModelConfigurations, PredictionResult, TransformerBlock,
    algorithms::{
        align_to_q, compute_current_attention_mask, compute_linear_layer,
        compute_multi_head_attention, compute_rms_norm, compute_rotary_position_embeddings,
        compute_swiglu, create_attention_mask, precompute_theta_tables,
    },
    kv_cache::KVCache,
    record_tensor_flow_step, tensor_shape,
    tensors::{
        TinyTensor, argmax, broadcast_add, concatenate, narrow, reshape, select_index,
        transpose_with_dim, unsqueeze,
    },
    tui::{InferenceDebugState, build_attention_heatmaps, collect_top_candidate_logits},
};

pub fn compute_input_tensor(
    model_configurations: &ModelConfigurations,
    llama_model: &LlamaModel,
    input_token_ids: &[u32],
    debug_state: &mut InferenceDebugState,
) -> Result<TinyTensor, anyhow::Error> {
    let token_ids = input_token_ids
        .iter()
        .map(|&token_id| token_id as f32)
        .collect::<Vec<_>>();
    let token_ids_length = token_ids.len();
    let token_ids_tensor = TinyTensor::new_from_vec(token_ids, &[token_ids_length])?;
    let embedding_started_at = Instant::now();
    let hidden_state = match &llama_model.embedding_tokens {
        Some(result) => select_index(&token_ids_tensor, result, 0)?,
        None => select_index(&token_ids_tensor, &llama_model.lm_head, 0)?,
    };
    record_tensor_flow_step(
        debug_state,
        None,
        "token embedding",
        tensor_shape(&token_ids_tensor),
        tensor_shape(&hidden_state),
        embedding_started_at,
    );
    let unsqueeze_started_at = Instant::now();
    let hidden_state = unsqueeze(hidden_state, 0)?;
    record_tensor_flow_step(
        debug_state,
        None,
        "add batch dim",
        vec![input_token_ids.len(), model_configurations.hidden_size],
        tensor_shape(&hidden_state),
        unsqueeze_started_at,
    );

    Ok(hidden_state)
}

pub fn predict_next_token_with_kv_cache(
    model_configurations: &ModelConfigurations,
    llama_model: &LlamaModel,
    tokenizer: &Tokenizer,
    mut debug_state: InferenceDebugState,
    mut hidden_state: TinyTensor,
    kv_cache: &mut KVCache,
) -> Result<PredictionResult, anyhow::Error> {
    let mut visualization_overhead = std::time::Duration::ZERO;

    let hidden_state_sequence_length = hidden_state.get_shape()[1];
    let cache_sequence_length = match kv_cache.get(0) {
        Some((k, _v)) => k.get_shape()[2],
        None => 0,
    };
    let total_sequence_length = hidden_state_sequence_length + cache_sequence_length;
    // We will use this later to mark the token position info.
    let (cos_table, sin_table) = precompute_theta_tables(
        total_sequence_length,
        model_configurations.head_dim,
        model_configurations.rope_theta,
    )?;

    let cos_table = narrow(
        cos_table,
        2,
        cache_sequence_length,
        hidden_state_sequence_length,
    )?;

    let sin_table = narrow(
        sin_table,
        2,
        cache_sequence_length,
        hidden_state_sequence_length,
    )?;

    // Create an attention mask with the max sequence length we got from above.
    let attention_mask = create_attention_mask(total_sequence_length)?;

    let has_cache = cache_sequence_length > 0;
    let attention_mask = match has_cache {
        true => compute_current_attention_mask(
            &attention_mask,
            total_sequence_length - hidden_state_sequence_length,
        )?,
        false => attention_mask,
    };

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

            let q = compute_q(
                index,
                layer,
                &normalized_hidden_state,
                hidden_state_sequence_length,
                model_configurations,
                &mut debug_state,
                &cos_table,
                &sin_table,
            )?;

            let (k, v) = compute_kv(
                index,
                layer,
                &normalized_hidden_state,
                hidden_state_sequence_length,
                model_configurations,
                &mut debug_state,
                &cos_table,
                &sin_table,
            )?;

            let (k, v) = match kv_cache.get(index) {
                Some((old_k, old_v)) => {
                    let k = concatenate(&old_k, &k, 2)?;
                    let v = concatenate(&old_v, &v, 2)?;

                    (k, v)
                }
                None => (k, v),
            };

            // I cache the KV here for optimal speed,
            // because this will avoid recomputing algin_to_q / GQA over and over.
            kv_cache.update(index, k.clone(), v.clone());

            let does_attention_head_match_kv_heads = model_configurations.num_attention_heads
                == model_configurations.num_key_value_heads;

            let (k, v) = match does_attention_head_match_kv_heads {
                true => (k, v),
                false => compute_gqa(&k, &v, index, model_configurations, &mut debug_state)?,
            };

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
        hidden_state_sequence_length - 1,
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

fn compute_q(
    index: usize,
    layer: &TransformerBlock,
    normalized_hidden_state: &TinyTensor,
    max_sequence_length: usize,
    model_configurations: &ModelConfigurations,
    debug_state: &mut InferenceDebugState,
    cos_table: &TinyTensor,
    sin_table: &TinyTensor,
) -> Result<TinyTensor> {
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
        debug_state,
        Some(index),
        "q projection",
        tensor_shape(&normalized_hidden_state),
        tensor_shape(&q),
        q_projection_started_at,
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
        debug_state,
        Some(index),
        "q reshape",
        q_shape_before_reshape,
        tensor_shape(&q),
        q_reshape_started_at,
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
        debug_state,
        Some(index),
        "q transpose",
        q_shape_before_transpose,
        tensor_shape(&q),
        q_transpose_started_at,
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
        debug_state,
        Some(index),
        "q rope",
        q_shape_before_rope,
        tensor_shape(&q),
        q_rope_started_at,
    );

    Ok(q)
}

fn compute_kv(
    index: usize,
    layer: &TransformerBlock,
    normalized_hidden_state: &TinyTensor,
    max_sequence_length: usize,
    model_configurations: &ModelConfigurations,
    debug_state: &mut InferenceDebugState,
    cos_table: &TinyTensor,
    sin_table: &TinyTensor,
) -> Result<(TinyTensor, TinyTensor)> {
    let k_projection_started_at = Instant::now();
    let k = compute_linear_layer(&layer.k_projection, &normalized_hidden_state, None)?;
    record_tensor_flow_step(
        debug_state,
        Some(index),
        "k projection",
        tensor_shape(&normalized_hidden_state),
        tensor_shape(&k),
        k_projection_started_at,
    );

    let v_projection_started_at = Instant::now();
    let v = compute_linear_layer(&layer.v_projection, &normalized_hidden_state, None)?;
    record_tensor_flow_step(
        debug_state,
        Some(index),
        "v projection",
        tensor_shape(&normalized_hidden_state),
        tensor_shape(&v),
        v_projection_started_at,
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
        debug_state,
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
        debug_state,
        Some(index),
        "v reshape",
        v_shape_before_reshape,
        tensor_shape(&v),
        v_reshape_started_at,
    );

    let k_shape_before_transpose = tensor_shape(&k);
    let k_transpose_started_at = Instant::now();
    let k = transpose_with_dim(k, 1, 2)?;
    record_tensor_flow_step(
        debug_state,
        Some(index),
        "k transpose",
        k_shape_before_transpose,
        tensor_shape(&k),
        k_transpose_started_at,
    );

    let v_shape_before_transpose = tensor_shape(&v);
    let v_transpose_started_at = Instant::now();
    let v = transpose_with_dim(v, 1, 2)?;
    record_tensor_flow_step(
        debug_state,
        Some(index),
        "v transpose",
        v_shape_before_transpose,
        tensor_shape(&v),
        v_transpose_started_at,
    );

    let k_shape_before_rope = tensor_shape(&k);
    let k_rope_started_at = Instant::now();
    let k = compute_rotary_position_embeddings(&k, &cos_table, &sin_table)?;
    record_tensor_flow_step(
        debug_state,
        Some(index),
        "k rope",
        k_shape_before_rope,
        tensor_shape(&k),
        k_rope_started_at,
    );

    Ok((k, v))
}

pub fn compute_gqa(
    k: &TinyTensor,
    v: &TinyTensor,
    index: usize,
    model_configurations: &ModelConfigurations,
    debug_state: &mut InferenceDebugState,
) -> Result<(TinyTensor, TinyTensor)> {
    let align_started_at = Instant::now();
    let k_shape_before_align = tensor_shape(&k);
    let v_shape_before_align = tensor_shape(&v);
    let (k, v) = align_to_q(
        model_configurations.num_attention_heads,
        model_configurations.num_key_value_heads,
        &k,
        &v,
    )?;
    record_tensor_flow_step(
        debug_state,
        Some(index),
        "gqa align k",
        k_shape_before_align,
        tensor_shape(&k),
        align_started_at,
    );
    record_tensor_flow_step(
        debug_state,
        Some(index),
        "gqa align v",
        v_shape_before_align,
        tensor_shape(&v),
        align_started_at,
    );

    Ok((k, v))
}