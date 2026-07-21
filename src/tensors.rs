use std::sync::Arc;

use anyhow::{Result, anyhow};
use candle_core::{Shape, Tensor, shape::Dim};
use gemm::gemm;
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};

/// I will swap the inner with my own minimal tensor,
/// but for now, I am just using the candle tensor for implementing the
/// inference engine first.
pub struct TinyTensor {
    strides: Vec<usize>,
    shape: Vec<usize>,
    data: Arc<Vec<f32>>,
}

impl TinyTensor {
    pub fn new(data: &[f32], shape: &[usize]) -> Self {
        Self {
            strides: Self::compute_strides(shape),
            shape: shape.to_vec(),
            data: Arc::new(data.to_vec()),
        }
    }

    pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
        // Stride length matches shape
        let mut strides = vec![1];

        for dimension in shape.iter().rev().take(shape.len().saturating_sub(1)) {
            strides.insert(0, dimension * strides[0]);
        }

        strides
    }

    pub fn new_without_reallocate(data: Arc<Vec<f32>>, shape: Vec<usize>) -> Self {
        Self {
            strides: Self::compute_strides(&shape),
            shape: shape,
            data: data,
        }
    }

    pub fn load_weight(safetensors: &SafeTensors, tensor_name: &str) -> Result<Self> {
        let tensor_view = safetensors.tensor(tensor_name)?;
        let shape = tensor_view.shape();
        let data_type = tensor_view.dtype();
        let raw_bytes = tensor_view.data();

        let data: Vec<f32> = match data_type {
            Dtype::F32 => raw_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            Dtype::BF16 => raw_bytes
                .chunks_exact(2)
                .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            Dtype::F16 => raw_bytes
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            _ => return Err(anyhow!("Data type {} unsupported", data_type)),
        };

        Ok(Self::new(&data, shape))
    }

    pub fn get_shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of dimensions in this matrix
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Convert a rank 0 tensor into a scalar value.
    /// Return error if the tensor is higher than 0.
    pub fn to_scalar(self) -> Result<f32> {
        if self.rank() != 0 {
            return Err(anyhow!(
                "Only tensors with 1 dimension can be converted to a scalar"
            ));
        }

        Ok(self.data[0])
    }
}

pub fn reshape(a: TinyTensor, shape: &[usize]) -> Result<TinyTensor> {
    if shape.iter().product::<usize>() != a.data.len() {
        return Err(anyhow!("Shape mismatches the data"));
    }

    Ok(TinyTensor::new_without_reallocate(a.data, shape.to_vec()))
}

/// Please make sure the indexes are integers in f32,
/// because this implementation does not check against it.
///
/// Formula for each index:
///
/// new_shape = old_shape
/// new_shape[dim] = indexes.len()
///
/// outer_group_count = collect items from shape until dim, then product
///
/// For outer_index in 0..outer_group_count
///     outer_base = outer_index * shape[dim] * strides[dim]
///
///     For index_position in indexes:
///         start_i = outer_base + strides[dim] * indexes[i]
///         end_i = start_i + strides[dim]
pub fn select_index(a: &TinyTensor, indexes: &TinyTensor, dim: usize) -> Result<TinyTensor> {
    if dim >= a.rank() {
        return Err(anyhow!(
            "You should be using a valid dim index that is smaller than the tensor rank"
        ));
    }

    if indexes
        .data
        .iter()
        .any(|item| (*item as usize) >= a.shape[dim])
    {
        return Err(anyhow!(
            "Indexes should never exceed the specified dim size"
        ));
    }

    let mut new_shape = a.shape.to_owned();
    new_shape[dim] = indexes.data.len();

    let outer_group_count: usize = a.shape[..dim].iter().product();
    let stride = a.strides[dim];

    let mut new_data = Vec::with_capacity(outer_group_count * indexes.data.len() * stride);

    for outer_index in 0..outer_group_count {
        let outer_base = outer_index * a.shape[dim] * stride;

        for index_position in 0..indexes.data.len() {
            let start = outer_base + (stride * indexes.data[index_position] as usize);
            let end = start + stride;

            new_data.extend_from_slice(&a.data[start..end]);
        }
    }

    Ok(TinyTensor::new(&new_data, &new_shape))
}

pub fn unsqueeze(mut a: TinyTensor, dim: usize) -> Result<TinyTensor> {
    if dim > a.rank() {
        return Err(anyhow!("Dim should not exceed the tensor rank"));
    }

    let stride = if dim < a.shape.len() {
        a.strides[dim]
    } else {
        1
    };

    a.strides.insert(dim, stride);
    a.shape.insert(dim, 1);

    Ok(a)
}

fn compute_bmnk(a: &TinyTensor, b: &TinyTensor) -> Result<(usize, usize, usize, usize)> {
    if a.rank() < 2 || b.rank() < 2 {
        return Err(anyhow!(
            "Matrix multiplication requires tensors with at least 2 dimensions"
        ));
    }

    let outer_shape_a = &a.shape[..a.rank() - 2];
    let outer_shape_b = &b.shape[..b.rank() - 2];

    if outer_shape_a != outer_shape_b {
        return Err(anyhow!("Outer shape mismatches"));
    }

    let batch: usize = outer_shape_a.iter().product();

    if a.shape[a.rank() - 1] != b.shape[b.rank() - 2] {
        return Err(anyhow!("Inner dimensions mismatch"));
    }

    let m = a.shape[a.rank() - 2];
    let n = b.shape[b.rank() - 1];
    let k = a.shape[a.rank() - 1];

    Ok((batch, m, n, k))
}

/// Compute how many elements to move forward on the data array.
///
/// Supports tensors with rank 2 through 4, inclusive.
///
/// The left parameter indicates whether this is a left or right tensor.
/// For example, in matrix multiplication, being left or right can alter the result.
fn compute_skip(a: &TinyTensor, left: bool, m: usize, n: usize, k: usize) -> Result<usize> {
    let batch_strides = &a.strides[..a.rank() - 2];

    // This happens when having no batch strides
    if batch_strides.is_empty() {
        match left {
            true => return Ok(m * k),
            false => return Ok(n * k),
        }
    }

    if batch_strides.len() == 1 {
        // Directly return if it is a 3-D tensor.
        return Ok(batch_strides[0]);
    }

    Ok(match batch_strides {
        // If the dim shape at index 1 times the corresponding stride equal to the outer most dim's stride,
        // we just need the stride at the index 1.
        [stride_dim_zero, stride_dim_one] if *stride_dim_zero == stride_dim_one * a.shape[1] => {
            *stride_dim_one
        }
        // Ignore the dims that are 1.
        [stride_dim_zero, _] if a.shape[1] == 1 => *stride_dim_zero,
        [_, stride_dim_one] if a.shape[0] == 1 => *stride_dim_one,
        _ => {
            return Err(anyhow!(
                "Input tensor must not exceed 4-D nor lower than 2-D"
            ));
        }
    })
}

// b, m, n, k
// b: number of batch matrices
// m: rows of a, lhs, and destination/result, dst, rows
// n: columns of b, rhs, and destination/result, dst, columns
// k: shared reduced dimensions
//
// In Candle's matrix multiplication, it implies 4-D tensor inputs.
// If the input tensors are not 4-D, we will need to reshape them.
// And if the dims positions are different, we will need to permute.
pub fn matrix_multiply(a: &TinyTensor, b: &TinyTensor) -> Result<TinyTensor> {
    let (batch, m, n, k) = compute_bmnk(a, b)?;
    let (a_data, b_data) = (a.data.read().unwrap(), b.data.read().unwrap());

    let left_hand_side_data = a_data.as_slice();
    let right_hand_side_data = b_data.as_slice();

    // The destination is a matrix, a 2-D tensor.
    // The two dims of the destination derives from m and n.
    let destination_strides = TinyTensor::compute_strides(&[m, n]);
    let destination_column_stride = destination_strides[1];
    let destination_row_stride = destination_strides[0];

    // We use this variable to store the destination computed by GEMM
    let mut destination: Vec<f32> = vec![0.0; batch * m * n];
    let destination_skip = m * n;

    let (left_skip, right_skip) = (
        compute_skip(a, true, m, n, k)?,
        compute_skip(b, false, m, n, k)?,
    );

    let left_hand_side_column_stride = a.strides[a.rank() - 1];
    let left_hand_side_row_stride = a.strides[a.rank() - 2];
    let right_hand_side_column_stride = b.strides[b.rank() - 1];
    let right_hand_side_row_stride = b.strides[b.rank() - 2];

    let parallelism = Parallelism::Rayon(
        std::thread::available_parallelism()
            .map(|item| item.get())
            .unwrap_or(1),
    );

    for step in 0..batch {
        let left_hand_side_data_this_step = &left_hand_side_data[step * left_skip..];
        let right_hand_side_data_this_step = &right_hand_side_data[step * right_skip..];
        let destination_pointer = &mut destination[step * destination_skip..];

        unsafe {
            gemm(
                m,
                n,
                k,
                destination_pointer.as_mut_ptr(),
                destination_column_stride as isize,
                destination_row_stride as isize,
                false,
                left_hand_side_data_this_step.as_ptr(),
                left_hand_side_column_stride as isize,
                left_hand_side_row_stride as isize,
                right_hand_side_data_this_step.as_ptr(),
                right_hand_side_column_stride as isize,
                right_hand_side_row_stride as isize,
                0.0,
                0.0,
                false,
                false,
                false,
                parallelism,
            );
        }
    }

    let mut destination_shape = a.shape[..a.rank() - 2].to_vec();
    destination_shape.extend([m, n]);

    Ok(TinyTensor::new(&destination, &destination_shape))
}

pub fn matrix_add(a: &TinyTensor, b: &TinyTensor) -> Result<TinyTensor> {
    let new = a.inner.broadcast_add(&b.inner)?;

    Ok(TinyTensor { inner: new })
}

pub fn transpose(tensor: &TinyTensor) -> Result<TinyTensor> {
    Ok(TinyTensor {
        inner: tensor.inner.t()?,
    })
}

pub fn transpose_with_dim(a: &TinyTensor, dim1: usize, dim2: usize) -> Result<TinyTensor> {
    Ok(TinyTensor {
        inner: a.inner.transpose(dim1, dim2)?,
    })
}

pub fn flatten(a: &TinyTensor, start_dim: usize, end_dim: usize) -> Result<TinyTensor> {
    Ok(TinyTensor {
        inner: a.inner.flatten(start_dim, end_dim)?,
    })
}

pub fn narrow<D: Dim>(
    tensor: &TinyTensor,
    dim: D,
    start: usize,
    length: usize,
) -> Result<TinyTensor> {
    Ok(TinyTensor {
        inner: tensor.inner.narrow(dim, start, length)?,
    })
}

pub fn broadcast_multiply(a: &TinyTensor, b: &TinyTensor) -> Result<TinyTensor> {
    Ok(TinyTensor {
        inner: a.inner.broadcast_mul(&b.inner)?,
    })
}

pub fn broadcast_divide(a: &TinyTensor, b: &TinyTensor) -> Result<TinyTensor> {
    Ok(TinyTensor {
        inner: a.inner.broadcast_div(&b.inner)?,
    })
}

pub fn broadcast_substract(a: &TinyTensor, b: &TinyTensor) -> Result<TinyTensor> {
    Ok(TinyTensor {
        inner: a.inner.broadcast_sub(&b.inner)?,
    })
}

pub fn broadcast_add(a: &TinyTensor, b: &TinyTensor) -> Result<TinyTensor> {
    Ok(TinyTensor {
        inner: a.inner.broadcast_add(&b.inner)?,
    })
}

pub fn concatenate(a: &TinyTensor, b: &TinyTensor, dim: usize) -> Result<TinyTensor> {
    Ok(TinyTensor {
        inner: Tensor::cat(&[&a.inner, &b.inner], dim)?,
    })
}

pub fn concatenate_all(tensors: &[TinyTensor], dim: usize) -> Result<TinyTensor> {
    Ok(TinyTensor {
        inner: Tensor::cat(
            &tensors
                .iter()
                .map(|item| &item.inner)
                .collect::<Vec<&Tensor>>(),
            dim,
        )?,
    })
}

pub fn softmax(a: &TinyTensor) -> Result<TinyTensor> {
    Ok(TinyTensor {
        inner: candle_nn::ops::softmax_last_dim(&a.inner)?,
    })
}

pub fn square(a: &TinyTensor) -> Result<TinyTensor> {
    Ok(TinyTensor {
        inner: a.inner.sqr()?,
    })
}

pub fn mean<D>(a: &TinyTensor, dim: D) -> Result<TinyTensor>
where
    D: Dim,
{
    Ok(TinyTensor {
        inner: a.inner.mean_keepdim(dim)?,
    })
}

pub fn square_root(a: &TinyTensor) -> Result<TinyTensor> {
    Ok(TinyTensor {
        inner: a.inner.sqrt()?,
    })
}

pub fn silu(a: &TinyTensor) -> Result<TinyTensor> {
    Ok(TinyTensor {
        inner: a.inner.silu()?,
    })
}

pub fn argmax(a: &TinyTensor, dim: usize) -> Result<TinyTensor> {
    Ok(TinyTensor {
        inner: a.inner.argmax(dim)?,
    })
}

pub fn repeat(a: &TinyTensor, shape: impl Into<Shape>) -> Result<TinyTensor> {
    Ok(TinyTensor {
        inner: a.inner.repeat(shape)?,
    })
}

pub fn repeat_kv(a: &TinyTensor, n_repetition: usize) -> Result<TinyTensor> {
    if a.get_shape().dims().len() != 4 {
        return Err(anyhow!("Input tensor for kv repetition must be rank 4"));
    }

    let shape = a.get_shape();
    let (batch_size, num_kv_heads, sequence_length, head_dim) =
        (shape.dim(0)?, shape.dim(1)?, shape.dim(2)?, shape.dim(3)?);

    // Add a new dim after dim 0, 1 for storing repetition number
    let new_a = unsqueeze(a, 2)?;

    let repeated = repeat(&new_a, (1, 1, n_repetition, 1, 1))?;

    Ok(reshape(
        &repeated,
        (
            batch_size,
            n_repetition * num_kv_heads,
            sequence_length,
            head_dim,
        ),
    )?)
}

pub fn broadcast_matrix_multiply(a: &TinyTensor, b: &TinyTensor) -> Result<TinyTensor> {
    Ok(TinyTensor {
        inner: a.inner.broadcast_matmul(&b.inner)?,
    })
}
