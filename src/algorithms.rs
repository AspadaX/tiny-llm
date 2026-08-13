use anyhow::{Result, anyhow};

use crate::tensors::{
    TinyTensor, broadcast_add, broadcast_divide, broadcast_multiply, broadcast_subtract,
    concatenate, flatten, matrix_multiply, mean, narrow, repeat, reshape, silu, softmax, square,
    square_root, transpose, transpose_with_dim, unsqueeze,
};

/*
 * Start of math computations
 */

fn flatten_to_2d(x: &TinyTensor) -> Result<TinyTensor, anyhow::Error> {
    let x_shape = x.get_shape();
    let in_dim = *x_shape
        .last()
        .ok_or_else(|| anyhow!("Cannot flatten a scalar tensor"))?;
    let new_batch_size: usize = x_shape[..x_shape.len() - 1].iter().product();

    reshape(x.clone(), &[new_batch_size, in_dim])
}

fn bloat_back_to_original_dimension(
    weights: &TinyTensor,
    original_x: &TinyTensor,
    matrix_multiplied_x: TinyTensor,
) -> Result<TinyTensor, anyhow::Error> {
    let out_dim = weights.get_shape()[0];
    let original_shape = original_x.get_shape();
    let mut new_shape = original_shape[..original_shape.len() - 1].to_vec();
    new_shape.push(out_dim);

    reshape(matrix_multiplied_x, &new_shape)
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
    let transposed = transpose(weights.clone())?;

    // Matrix multiplication
    let matrix_multiplied = matrix_multiply(&flattened_x, &transposed)?;

    // Reshape x back to the original dimension
    let x = bloat_back_to_original_dimension(weights, x, matrix_multiplied)?;

    // Add bias if any
    Ok(match bias {
        Some(bias) => broadcast_add(&x, bias)?,
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
        TinyTensor::new_from_vec(cos_table, &table_shape)?,
        TinyTensor::new_from_vec(sin_table, &table_shape)?,
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
    let d_head = input_tensor.get_shape()[last_dimension];
    let pair_size = d_head / 2;

    let x = narrow(input_tensor.clone(), last_dimension, 0, pair_size)?;
    let y = narrow(input_tensor.clone(), last_dimension, pair_size, pair_size)?;

    let x_cos = broadcast_multiply(&x, cos_tensor)?;
    let y_sin = broadcast_multiply(&y, sin_tensor)?;

    let x_sin = broadcast_multiply(&x, sin_tensor)?;
    let y_cos = broadcast_multiply(&y, cos_tensor)?;

    let x = broadcast_subtract(&x_cos, &y_sin)?;
    let y = broadcast_add(&x_sin, &y_cos)?;

    Ok(concatenate(&x, &y, last_dimension)?)
}

#[allow(dead_code)]
pub fn prepare_rope_for_this_step(
    current_position: usize,
    current_sequence_length: usize,
    cos_tensor: &TinyTensor,
    sin_tensor: &TinyTensor,
) -> Result<(TinyTensor, TinyTensor)> {
    Ok((
        narrow(
            cos_tensor.clone(),
            2,
            current_position,
            current_sequence_length,
        )?,
        narrow(
            sin_tensor.clone(),
            2,
            current_position,
            current_sequence_length,
        )?,
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
    let epsilon = TinyTensor::new_from_vec(vec![epsilon.unwrap_or(1e-6)], &[1])?;

    let hidden_dimension = input_tensor.rank() - 1;

    let squared = square(input_tensor.clone())?;
    let mean = broadcast_add(&mean(&squared, &[hidden_dimension])?, &epsilon)?;
    let square_root = square_root(mean)?;

    let divided = broadcast_divide(input_tensor, &square_root)?;

    broadcast_multiply(&divided, weights)
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

    Ok(TinyTensor::new_from_vec(
        raw_mask,
        &[1, 1, max_sequence_len, max_sequence_len],
    )?)
}

#[allow(dead_code)]
pub fn compute_current_attention_mask(
    attention_mask: &TinyTensor,
    current_token_position: usize,
) -> Result<TinyTensor> {
    // Slice on rows.
    // Only 1 row is needed.
    let tensor = narrow(attention_mask.clone(), 2, current_token_position, 1)?;

    // Slice on columns.
    // All columns until the current token are needed.
    narrow(tensor, 3, 0, current_token_position + 1)
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
    let square_root_k_dimension = k.get_shape()[k.rank() - 1] as f32;
    let tensor_sqrt_k_dimension =
        TinyTensor::new_from_vec(vec![square_root_k_dimension.sqrt()], &[1])?;

    let q_k = matrix_multiply(q, &transpose(k.clone())?)?;
    let divided = broadcast_divide(&q_k, &tensor_sqrt_k_dimension)?;

    let applied_attention_mask = if let Some(attention_mask) = current_attention_mask {
        broadcast_add(&divided, attention_mask)?
    } else {
        divided
    };

    let softmaxed = softmax(&applied_attention_mask, applied_attention_mask.rank() - 1)?;

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

    let concatenated = transpose_with_dim(heads, 1, 2)?;

    // Recover the output back to hidden state
    let flattened = flatten(concatenated, 2, 3)?;

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

pub fn repeat_kv(a: &TinyTensor, n_repetition: usize) -> Result<TinyTensor> {
    if a.rank() != 4 {
        return Err(anyhow!("Input tensor for kv repetition must be rank 4"));
    }

    let shape = a.get_shape();
    let (batch_size, num_kv_heads, sequence_length, head_dim) =
        (shape[0], shape[1], shape[2], shape[3]);

    // Add a new dimension after the KV-head dimension to hold repetitions.
    let new_a = unsqueeze(a.clone(), 2)?;
    let repeated = repeat(&new_a, &[1, 1, n_repetition, 1, 1])?;

    reshape(
        repeated,
        &[
            batch_size,
            n_repetition * num_kv_heads,
            sequence_length,
            head_dim,
        ],
    )
}

/// SWISH: A SELF-GATED ACTIVATION FUNCTION: https://arxiv.org/pdf/1710.05941v1
/// GLU Variants Improve Transformer: https://arxiv.org/pdf/2002.05202
///
/// The `transformers` library implementation of swiglu swapped the `gate_projection` and `up`.
/// Therefore, when loading llama model weights, we will need to plugin the up to gate and so on.
pub fn compute_swiglu(
    hidden_state: &TinyTensor, // x [batch_size, sequence_length, hidden_size]
    gate_projection: &TinyTensor, // V [intermediate_size, hidden_size]
    up_projection: &TinyTensor, // W [intermediate_size, hidden_size]
    down_projection: &TinyTensor, // W2 [batch_size, sequence_length, hidden_size]
) -> Result<TinyTensor> {
    // The matrix multiplication here uses linear, as it is mathematically identical without a bias,
    // and the linear implementation takes care of the dimensional differences.
    let gate = compute_linear_layer(gate_projection, hidden_state, None)?;

    let up = compute_linear_layer(up_projection, hidden_state, None)?;

    let activated_gate = silu(gate)?;

    let apply_gate = broadcast_multiply(&activated_gate, &up)?;

    // Ok(matrix_multiply(&apply_gate, down_projection)?)
    Ok(compute_linear_layer(down_projection, &apply_gate, None)?)
}
