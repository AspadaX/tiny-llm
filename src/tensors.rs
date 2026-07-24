use std::{
    cmp::max,
    sync::{Arc, RwLock},
};

use anyhow::{Result, anyhow};
use candle_core::{Shape, Tensor, shape::Dim};
use gemm::{Parallelism, gemm};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};

/// I will swap the inner with my own minimal tensor,
/// but for now, I am just using the candle tensor for implementing the
/// inference engine first.
pub struct TinyTensor {
    strides: Vec<usize>,
    shape: Vec<usize>,
    data: Arc<RwLock<Vec<f32>>>,
}

impl TinyTensor {
    pub fn new(data: &[f32], shape: &[usize]) -> Self {
        Self {
            strides: Self::compute_strides(shape),
            shape: shape.to_vec(),
            data: Arc::new(RwLock::new(data.to_vec())),
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

    pub fn new_without_reallocate(data: Arc<RwLock<Vec<f32>>>, shape: Vec<usize>) -> Self {
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

    /// `count_from_end` will be ignored for 0 dim index.
    /// empty `dim_indexes` will return the whole shape.
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

        Ok(self.data.read().unwrap()[0])
    }
}

pub fn reshape(a: TinyTensor, shape: &[usize]) -> Result<TinyTensor> {
    if shape.iter().product::<usize>() != a.data.read().unwrap().len() {
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

    let indexes_data = indexes.data.read().unwrap();
    let a_data = a.data.read().unwrap();

    if indexes_data
        .iter()
        .any(|item| (*item as usize) >= a.shape[dim])
    {
        return Err(anyhow!(
            "Indexes should never exceed the specified dim size"
        ));
    }

    let mut new_shape = a.shape.to_owned();
    new_shape[dim] = indexes_data.len();

    let outer_group_count: usize = a.shape[..dim].iter().product();
    let stride = a.strides[dim];

    let mut new_data = Vec::with_capacity(outer_group_count * indexes_data.len() * stride);

    for outer_index in 0..outer_group_count {
        let outer_base = outer_index * a.shape[dim] * stride;

        for index_position in 0..indexes_data.len() {
            let start = outer_base + (stride * indexes_data[index_position] as usize);
            let end = start + stride;

            new_data.extend_from_slice(&a_data[start..end]);
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

/// Compute the new tensor's shape after a broadcasting computation.
fn broadcast_shape(a: &TinyTensor, b: &TinyTensor) -> Result<Vec<usize>> {
    let new_dim_length = max(a.shape.len(), b.shape.len());
    let mut new_shape: Vec<usize> = vec![0; new_dim_length];

    for (index, dimension) in new_shape.iter_mut().enumerate() {
        let reversed_index = new_dim_length - index;
        let mut a_dimension = 1;
        let mut b_dimension = 1;

        // Align shapes from the trailing dimensions.
        // Missing leading dimensions are treated as 1 for broadcasting.
        if reversed_index <= a.rank() {
            // Compute the index offset from right
            a_dimension = a.shape[a.rank() - reversed_index];
        }

        if reversed_index <= b.rank() {
            b_dimension = b.shape[b.rank() - reversed_index];
        }

        // Dimensions are compatible if they are equal, or if either one is 1.
        if a_dimension != b_dimension && a_dimension != 1 && b_dimension != 1 {
            return Err(anyhow!("Dimensions mismatch"));
        }

        *dimension = max(a_dimension, b_dimension);
    }

    Ok(new_shape)
}

/// Compute the broadcasted shape's strides
fn broadcast_as(
    original_shape: &[usize],
    original_stride: &[usize],
    new_shape: &[usize],
) -> Result<Vec<usize>> {
    // original shape | original strides | target shape | expected strides
    // [3]            | [1]              | [2, 3]       | [0, 1]
    // [2, 1]         | [1, 1]           | [2, 4]       | [1, 0]
    // [2, 3]         | [3, 1]           | [2, 4]       | error

    if original_shape.len() != original_stride.len() {
        return Err(anyhow!(
            "Original tensor shapes and strides length mismatched"
        ));
    }

    if new_shape.len() < original_shape.len() {
        return Err(anyhow!("New shape should not be smaller than the old one"));
    }

    let added_shape = new_shape.len() - original_shape.len();
    let mut new_strides = vec![0; added_shape];

    for dimension in 0..original_shape.len() {
        let original_shape_dimension = original_shape[dimension];
        let new_shape_dimension = new_shape[added_shape + dimension];
        let original_stride_dimension = original_stride[dimension];

        let stride = if original_shape_dimension == new_shape_dimension {
            original_stride_dimension
        } else if original_shape_dimension != 1 {
            return Err(anyhow!("Incompatible broadcast shape"));
        } else {
            0
        };

        new_strides.push(stride);
    }

    Ok(new_strides)
}

/// Create a broadcasted view of the tensor without copying its data.
///
/// Broadcasting is represented by changing shape/strides only.
/// A stride of `0` means that dimension reuses the same underlying value.
fn broadcast_view(a: &TinyTensor, shape: &[usize]) -> Result<TinyTensor> {
    Ok(TinyTensor {
        strides: broadcast_as(&a.shape, &a.strides, shape)?,
        shape: shape.to_vec(),
        data: a.data.clone(),
    })
}

fn get_offset_from_linear_index(mut index: usize, shape: &[usize], strides: &[usize]) -> usize {
    let mut offset = 0;

    // For shape [2, 3, 4], the rightmost dimension changes fastest:
    //
    // linear index:  0  1  2  3  4  5 ...
    // coordinates:  [0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,1,0], [0,1,1] ...
    //
    // We loop over dimension indexes in reverse order: 2, 1, 0.
    // That means we process dimension sizes 4, then 3, then 2.
    for dim in (0..shape.len()).rev() {
        // Size of the current dimension.
        //
        // Example:
        // shape = [2, 3, 4]
        //
        // dim = 2 -> dimension = 4
        // dim = 1 -> dimension = 3
        // dim = 0 -> dimension = 2
        let dimension = shape[dim];

        // Find the coordinate for this dimension.
        //
        // `% dimension` gives "where we are" inside the current dimension.
        //
        // Example:
        // shape = [2, 3, 4]
        // index = 17
        //
        // For dim 2:
        // coordinate = 17 % 4 = 1
        //
        // So in the last dimension, we are at position 1.
        let coordinate = index % dimension;

        // Remove the coordinate we just extracted.
        //
        // After this division, the next loop iteration can extract the coordinate
        // for the dimension to the left.
        //
        // Example:
        // index = 17
        //
        // After dim 2:
        // index = 17 / 4 = 4
        //
        // Then dim 1 can use this reduced index to extract the coordinate
        // for the next higher dimension.
        index /= dimension;

        // Convert this dimension's coordinate into movement inside the flat data buffer.
        //
        // `strides[dim]` tells us how far we move in memory when this coordinate
        // increases by 1.
        //
        // Example with broadcasting:
        // shape   = [2, 3]
        // strides = [0, 1]
        // coords  = [1, 1]
        //
        // offset contribution:
        // dim 0 -> 1 * 0 = 0
        // dim 1 -> 1 * 1 = 1
        //
        // total offset = 1
        //
        // The `0` stride is what makes broadcasting reuse the same row.
        offset += coordinate * strides[dim];
    }

    offset
}

fn perform_broadcast_binary_operation<F>(
    a: &TinyTensor,
    b: &TinyTensor,
    operation: F,
) -> Result<TinyTensor>
where
    F: Fn(f32, f32) -> f32,
{
    let broadcasted_shape = broadcast_shape(a, b)?;

    let a_view = broadcast_view(a, &broadcasted_shape)?;
    let b_view = broadcast_view(b, &broadcasted_shape)?;

    let a_data = a.data.read().unwrap();
    let b_data = b.data.read().unwrap();

    let output_data_length: usize = broadcasted_shape.iter().product();
    let mut output_data = Vec::with_capacity(output_data_length);

    for index in 0..output_data_length {
        let offset_a = get_offset_from_linear_index(index, &broadcasted_shape, &a_view.strides);
        let offset_b = get_offset_from_linear_index(index, &broadcasted_shape, &b_view.strides);

        output_data.push(operation(a_data[offset_a], b_data[offset_b]));
    }

    Ok(TinyTensor::new(&output_data, &broadcasted_shape))
}

pub fn broadcast_add(a: &TinyTensor, b: &TinyTensor) -> Result<TinyTensor> {
    perform_broadcast_binary_operation(a, b, |a, b| a + b)
}

pub fn broadcast_multiply(a: &TinyTensor, b: &TinyTensor) -> Result<TinyTensor> {
    perform_broadcast_binary_operation(a, b, |a, b| a * b)
}

pub fn broadcast_divide(a: &TinyTensor, b: &TinyTensor) -> Result<TinyTensor> {
    perform_broadcast_binary_operation(a, b, |a, b| a / b)
}

pub fn broadcast_subtract(a: &TinyTensor, b: &TinyTensor) -> Result<TinyTensor> {
    perform_broadcast_binary_operation(a, b, |a, b| a - b)
}

// pub fn transpose(tensor: &TinyTensor) -> Result<TinyTensor> {
//     Ok(TinyTensor {
//         inner: tensor.inner.t()?,
//     })
// }

// pub fn transpose_with_dim(a: &TinyTensor, dim1: usize, dim2: usize) -> Result<TinyTensor> {
//     Ok(TinyTensor {
//         inner: a.inner.transpose(dim1, dim2)?,
//     })
// }

// pub fn flatten(a: &TinyTensor, start_dim: usize, end_dim: usize) -> Result<TinyTensor> {
//     Ok(TinyTensor {
//         inner: a.inner.flatten(start_dim, end_dim)?,
//     })
// }

// pub fn narrow<D: Dim>(
//     tensor: &TinyTensor,
//     dim: D,
//     start: usize,
//     length: usize,
// ) -> Result<TinyTensor> {
//     Ok(TinyTensor {
//         inner: tensor.inner.narrow(dim, start, length)?,
//     })
// }

// pub fn concatenate(a: &TinyTensor, b: &TinyTensor, dim: usize) -> Result<TinyTensor> {
//     Ok(TinyTensor {
//         inner: Tensor::cat(&[&a.inner, &b.inner], dim)?,
//     })
// }

// pub fn concatenate_all(tensors: &[TinyTensor], dim: usize) -> Result<TinyTensor> {
//     Ok(TinyTensor {
//         inner: Tensor::cat(
//             &tensors
//                 .iter()
//                 .map(|item| &item.inner)
//                 .collect::<Vec<&Tensor>>(),
//             dim,
//         )?,
//     })
// }

// pub fn softmax(a: &TinyTensor) -> Result<TinyTensor> {
//     Ok(TinyTensor {
//         inner: candle_nn::ops::softmax_last_dim(&a.inner)?,
//     })
// }

// pub fn square(a: &TinyTensor) -> Result<TinyTensor> {
//     Ok(TinyTensor {
//         inner: a.inner.sqr()?,
//     })
// }

// pub fn mean<D>(a: &TinyTensor, dim: D) -> Result<TinyTensor>
// where
//     D: Dim,
// {
//     Ok(TinyTensor {
//         inner: a.inner.mean_keepdim(dim)?,
//     })
// }

// pub fn square_root(a: &TinyTensor) -> Result<TinyTensor> {
//     Ok(TinyTensor {
//         inner: a.inner.sqrt()?,
//     })
// }

// pub fn silu(a: &TinyTensor) -> Result<TinyTensor> {
//     Ok(TinyTensor {
//         inner: a.inner.silu()?,
//     })
// }

// pub fn argmax(a: &TinyTensor, dim: usize) -> Result<TinyTensor> {
//     Ok(TinyTensor {
//         inner: a.inner.argmax(dim)?,
//     })
// }

// pub fn repeat(a: &TinyTensor, shape: impl Into<Shape>) -> Result<TinyTensor> {
//     Ok(TinyTensor {
//         inner: a.inner.repeat(shape)?,
//     })
// }

// pub fn repeat_kv(a: &TinyTensor, n_repetition: usize) -> Result<TinyTensor> {
//     if a.get_shape().dims().len() != 4 {
//         return Err(anyhow!("Input tensor for kv repetition must be rank 4"));
//     }

//     let shape = a.get_shape();
//     let (batch_size, num_kv_heads, sequence_length, head_dim) =
//         (shape.dim(0)?, shape.dim(1)?, shape.dim(2)?, shape.dim(3)?);

//     // Add a new dim after dim 0, 1 for storing repetition number
//     let new_a = unsqueeze(a, 2)?;

//     let repeated = repeat(&new_a, (1, 1, n_repetition, 1, 1))?;

//     Ok(reshape(
//         &repeated,
//         (
//             batch_size,
//             n_repetition * num_kv_heads,
//             sequence_length,
//             head_dim,
//         ),
//     )?)
// }

// pub fn broadcast_matrix_multiply(a: &TinyTensor, b: &TinyTensor) -> Result<TinyTensor> {
//     Ok(TinyTensor {
//         inner: a.inner.broadcast_matmul(&b.inner)?,
//     })
// }
