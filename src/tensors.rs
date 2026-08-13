use std::{
    cmp::max,
    sync::{Arc, RwLock},
};

use anyhow::{anyhow, Result};

use gemm::{gemm, Parallelism};
use half::{bf16, f16};
use rten_simd::SimdOp;
use safetensors::{Dtype, SafeTensors};

use crate::simd::{BinaryOperation, BroadcastBinaryOperation};

#[derive(Debug, Clone)]
pub struct TinyTensor {
    strides: Vec<usize>,
    shape: Vec<usize>,
    data: Arc<RwLock<Vec<f32>>>,
    offset: usize,
}

impl TinyTensor {
    pub fn new_from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        if data.len() != shape.iter().product::<usize>() {
            return Err(anyhow!("Data length does not match shape"));
        }

        Ok(Self {
            strides: Self::compute_strides(shape),
            shape: shape.to_vec(),
            data: Arc::new(RwLock::new(data)),
            offset: 0,
        })
    }

    pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
        // Stride length matches shape
        let mut strides = vec![1];

        for dimension in shape.iter().rev().take(shape.len().saturating_sub(1)) {
            strides.insert(0, dimension * strides[0]);
        }

        strides
    }

    pub fn new_without_reallocate(
        data: Arc<RwLock<Vec<f32>>>,
        shape: Vec<usize>,
        offset: usize,
    ) -> Self {
        Self {
            strides: Self::compute_strides(&shape),
            shape,
            data,
            offset,
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

        Ok(Self::new_from_vec(data, shape)?)
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
        if self.shape.iter().product::<usize>() != 1 {
            return Err(anyhow!(
                "Only tensors with exactly 1 element can be converted to a scalar"
            ));
        }

        Ok(self.data.read().unwrap()[self.offset])
    }

    /// Check whether the tensor is C-style, row-major contiguous.
    ///
    /// In row-major layout, adjacent elements along the last dimension are stored
    /// next to each other. The last dimension has stride 1, and each earlier stride
    /// is the product of all dimensions to its right.
    ///
    /// Example:
    /// shape   = [2, 3, 4]
    /// strides = [12, 4, 1]
    ///
    /// Size-1 dimensions are ignored because their stride does not affect the
    /// logical traversal order.
    pub fn is_contiguous(&self) -> bool {
        if self.strides.len() != self.shape.len() {
            return false;
        }

        let mut expected_stride = 1;

        for (stride, dim) in self.strides.iter().zip(self.shape.iter()).rev() {
            if *dim != 1 && expected_stride != *stride {
                return false;
            }

            expected_stride *= *dim;
        }

        true
    }
}

pub fn make_contiguous_data(a: TinyTensor) -> Result<Vec<f32>> {
    let total_elements: usize = a.shape.iter().product();
    let mut new_data = Vec::with_capacity(total_elements);
    let old_data = a.data.read().unwrap();

    for element in 0..total_elements {
        let offset = compute_offset_from_linear_index(element, &a.shape, &a.strides, a.offset);
        new_data.push(old_data[offset]);
    }

    Ok(new_data)
}

pub fn reshape(a: TinyTensor, shape: &[usize]) -> Result<TinyTensor> {
    if shape.iter().product::<usize>() != a.shape.iter().product::<usize>() {
        return Err(anyhow!("Shape mismatches the data"));
    }

    if !a.is_contiguous() {
        let new_data = make_contiguous_data(a)?;
        return Ok(TinyTensor::new_from_vec(new_data, shape)?);
    }

    Ok(TinyTensor::new_without_reallocate(
        a.data,
        shape.to_vec(),
        a.offset,
    ))
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
pub fn select_index(indexes: &TinyTensor, a: &TinyTensor, dim: usize) -> Result<TinyTensor> {
    if dim >= a.rank() {
        return Err(anyhow!(
            "You should be using a valid dim index that is smaller than the tensor rank"
        ));
    }

    let indexes_data = indexes.data.read().unwrap();
    let a_data = a.data.read().unwrap();
    let index_count: usize = indexes.shape.iter().product();
    let index_values: Vec<usize> = (0..index_count)
        .map(|linear_index| {
            let offset = compute_offset_from_linear_index(
                linear_index,
                &indexes.shape,
                &indexes.strides,
                indexes.offset,
            );
            indexes_data[offset] as usize
        })
        .collect();

    if index_values.iter().any(|index| *index >= a.shape[dim]) {
        return Err(anyhow!(
            "Indexes should never exceed the specified dim size"
        ));
    }

    let mut new_shape = a.shape.to_owned();
    new_shape[dim] = index_values.len();

    let outer_group_count: usize = a.shape[..dim].iter().product();
    let stride = a.strides[dim];

    let mut new_data = Vec::with_capacity(outer_group_count * index_values.len() * stride);

    for outer_index in 0..outer_group_count {
        let outer_base = compute_offset_from_linear_index(
            outer_index,
            &a.shape[..dim],
            &a.strides[..dim],
            a.offset,
        );

        for index in &index_values {
            let start = outer_base + (stride * *index);
            let end = start + stride;

            new_data.extend_from_slice(&a_data[start..end]);
        }
    }

    Ok(TinyTensor::new_from_vec(new_data, &new_shape)?)
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

// b, m, n, k
// b: number of batch matrices
// m: rows of a, lhs, and destination/result, dst, rows
// n: columns of b, rhs, and destination/result, dst, columns
// k: shared reduced dimensions
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

    let left_hand_side_column_stride = a.strides[a.rank() - 1];
    let left_hand_side_row_stride = a.strides[a.rank() - 2];
    let right_hand_side_column_stride = b.strides[b.rank() - 1];
    let right_hand_side_row_stride = b.strides[b.rank() - 2];

    let batch_shape = &a.shape[..a.rank() - 2];
    let a_batch_strides = &a.strides[..a.rank() - 2];
    let b_batch_strides = &b.strides[..b.rank() - 2];

    let parallelism = match m {
        1 => Parallelism::None,
        _ => Parallelism::Rayon(
            std::thread::available_parallelism()
                .map(|item| item.get())
                .unwrap_or(1),
        ),
    };

    for step in 0..batch {
        let a_offset =
            compute_offset_from_linear_index(step, &batch_shape, &a_batch_strides, a.offset);
        let b_offset =
            compute_offset_from_linear_index(step, &batch_shape, &b_batch_strides, b.offset);

        let left_hand_side_data_this_step = &left_hand_side_data[a_offset..];
        let right_hand_side_data_this_step = &right_hand_side_data[b_offset..];
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
                1.0,
                false,
                false,
                false,
                parallelism,
            );
        }
    }

    let mut destination_shape = a.shape[..a.rank() - 2].to_vec();
    destination_shape.extend([m, n]);

    Ok(TinyTensor::new_from_vec(destination, &destination_shape)?)
}

/// Compute the new tensor's shape after a broadcasting computation.
fn broadcast_shape(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>> {
    let new_dim_length = max(a_shape.len(), b_shape.len());
    let mut new_shape: Vec<usize> = vec![0; new_dim_length];

    for (index, dimension) in new_shape.iter_mut().enumerate() {
        let reversed_index = new_dim_length - index;
        let mut a_dimension = 1;
        let mut b_dimension = 1;

        // Align shapes from the trailing dimensions.
        // Missing leading dimensions are treated as 1 for broadcasting.
        if reversed_index <= a_shape.len() {
            // Compute the index offset from right
            a_dimension = a_shape[a_shape.len() - reversed_index];
        }

        if reversed_index <= b_shape.len() {
            b_dimension = b_shape[b_shape.len() - reversed_index];
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
        offset: a.offset,
    })
}

pub fn compute_offset_from_linear_index(
    mut index: usize,
    shape: &[usize],
    strides: &[usize],
    base_offset: usize,
) -> usize {
    let mut offset = base_offset;

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

fn perform_broadcast_binary_operation(
    a: &TinyTensor,
    b: &TinyTensor,
    operation: BinaryOperation,
) -> Result<TinyTensor> {
    let broadcasted_shape = broadcast_shape(&a.shape, &b.shape)?;

    let a_view = broadcast_view(a, &broadcasted_shape)?;
    let b_view = broadcast_view(b, &broadcasted_shape)?;

    let a_data = a.data.read().unwrap();
    let b_data = b.data.read().unwrap();

    let output_data_length: usize = broadcasted_shape.iter().product();
    let mut output_data = vec![0.0; output_data_length];

    BroadcastBinaryOperation {
        a_data: &a_data,
        b_data: &b_data,
        output_data: &mut output_data,

        shape: &broadcasted_shape,
        a_strides: &a_view.strides,
        b_strides: &b_view.strides,

        a_offset: a_view.offset,
        b_offset: b_view.offset,
        operation,
    }
    .dispatch();

    Ok(TinyTensor::new_from_vec(output_data, &broadcasted_shape)?)
}

pub fn broadcast_add(a: &TinyTensor, b: &TinyTensor) -> Result<TinyTensor> {
    perform_broadcast_binary_operation(a, b, BinaryOperation::Add)
}

pub fn broadcast_multiply(a: &TinyTensor, b: &TinyTensor) -> Result<TinyTensor> {
    perform_broadcast_binary_operation(a, b, BinaryOperation::Multiply)
}

pub fn broadcast_divide(a: &TinyTensor, b: &TinyTensor) -> Result<TinyTensor> {
    perform_broadcast_binary_operation(a, b, BinaryOperation::Divide)
}

pub fn broadcast_subtract(a: &TinyTensor, b: &TinyTensor) -> Result<TinyTensor> {
    perform_broadcast_binary_operation(a, b, BinaryOperation::Subtract)
}

#[allow(dead_code)]
pub fn broadcast_matrix_multiply(a: &TinyTensor, b: &TinyTensor) -> Result<TinyTensor> {
    if a.rank() < 2 || b.rank() < 2 {
        return Err(anyhow!(
            "Broadcast matrix multiplication requires both tensors have at least 2 dimensions"
        ));
    }

    if a.shape[a.rank() - 1] != b.shape[b.rank() - 2] {
        return Err(anyhow!("Both tensors' k value must match"));
    }

    // broadcast matrix multiplication broadcasts the batch dimensions
    let a_batch_dimension_shape = &a.shape[..(a.rank() - 2)];
    let b_batch_dimension_shape = &b.shape[..(b.rank() - 2)];

    // therefore, we only broadcast the batch dimensions for now
    let broadcasted_shape = broadcast_shape(a_batch_dimension_shape, b_batch_dimension_shape)?;

    // Shape A matrix: [m, k]
    // Shape B matrix: [k, n]
    // Shape A tensor: [...batch dims, m, k]
    // Shape B tensor: [...batch dims, k, n]
    let m = a.shape[a.rank() - 2];
    let n = b.shape[b.rank() - 1];
    let k = a.shape[a.rank() - 1];

    let mut a_broadcasted_shape = broadcasted_shape.clone();
    a_broadcasted_shape.extend([m, k]);

    // No need for a second clone
    let mut b_broadcasted_shape = broadcasted_shape;
    b_broadcasted_shape.extend([k, n]);

    let a_view = broadcast_view(a, &a_broadcasted_shape)?;
    let b_view = broadcast_view(b, &b_broadcasted_shape)?;

    Ok(matrix_multiply(&a_view, &b_view)?)
}

/// Return a view with the last two dimensions swapped.
///
/// This does not copy or reorder the underlying data. It only swaps the
/// corresponding shape and stride entries.
pub fn transpose(tensor: TinyTensor) -> Result<TinyTensor> {
    if tensor.rank() < 2 {
        return Err(anyhow!(
            "Tensors with a shape smaller than 2 will not be able to transpose"
        ));
    }

    let dim1 = tensor.rank() - 2;
    let dim2 = tensor.rank() - 1;
    transpose_with_dim(tensor, dim1, dim2)
}

/// Return a view with two specified dimensions swapped.
///
/// This does not copy or reorder the underlying data. It only swaps the
/// corresponding shape and stride entries.
pub fn transpose_with_dim(a: TinyTensor, dim1: usize, dim2: usize) -> Result<TinyTensor> {
    if a.rank() < 2 {
        return Err(anyhow!(
            "Tensors with a shape smaller than 2 will not be able to transpose"
        ));
    }

    if dim1 >= a.rank() || dim2 >= a.rank() {
        return Err(anyhow!(
            "Specified dimension exceeded the total dimension numbers"
        ));
    }

    if dim1 == dim2 {
        return Ok(a);
    }

    let stride_left = a.strides[dim1];
    let shape_left = a.shape[dim1];

    let stride_right = a.strides[dim2];
    let shape_right = a.shape[dim2];

    let mut new_shape = a.shape;
    new_shape[dim1] = shape_right;
    new_shape[dim2] = shape_left;

    let mut new_strides = a.strides;
    new_strides[dim1] = stride_right;
    new_strides[dim2] = stride_left;

    Ok(TinyTensor {
        strides: new_strides,
        shape: new_shape,
        data: a.data,
        offset: a.offset,
    })
}

/// Collapse dimensions from `start_dim` through `end_dim`, inclusive, into one dimension.
///
/// This uses `reshape`, so non-contiguous tensors may be materialized into
/// contiguous storage depending on `reshape`'s behavior.
pub fn flatten(a: TinyTensor, start_dim: usize, end_dim: usize) -> Result<TinyTensor> {
    if start_dim > end_dim || end_dim >= a.rank() {
        return Err(anyhow!(
            "Start dimension can't be larger than the end dimension. And end dimension must be smaller than the rank"
        ));
    }

    if start_dim == end_dim {
        return Ok(a);
    }

    let flattened_dim: usize = a.shape[start_dim..=end_dim].iter().product();

    let mut new_shape = a.shape.clone();
    new_shape.drain(start_dim..=end_dim);
    new_shape.insert(start_dim, flattened_dim);

    reshape(a, &new_shape)
}

pub fn narrow(tensor: TinyTensor, dim: usize, start: usize, length: usize) -> Result<TinyTensor> {
    if dim >= tensor.rank() {
        return Err(anyhow!("Dimension is out of bounds"));
    }

    if start + length > tensor.shape[dim] {
        return Err(anyhow!("start + length can't exceed the dim's length"));
    }

    let mut shape = tensor.shape;
    shape[dim] = length;

    Ok(TinyTensor {
        offset: tensor.offset + tensor.strides[dim] * start,
        strides: tensor.strides,
        shape,
        data: tensor.data,
    })
}

fn concatenate_contiguous(tensors: &[TinyTensor], dim: usize) -> Result<TinyTensor> {
    let new_shape = compute_new_shape_for_concatenation(tensors, dim);
    let new_stride = TinyTensor::compute_strides(&new_shape);
    let mut new_data = Vec::with_capacity(new_shape.iter().product::<usize>());

    let contiguous_data = tensors
        .iter()
        .map(|tensor| make_contiguous_data(tensor.clone()))
        .collect::<Result<Vec<_>>>()?;

    // In Rust, a product on an empty array will end up with 1
    let inner: usize = new_shape[dim + 1..].iter().product();
    let outer: usize = new_shape[..dim].iter().product();

    for outer_index in 0..outer {
        for (tensor, tensor_data) in tensors.iter().zip(contiguous_data.iter()) {
            // How many contiguous elements this window copies from the flat data array
            let block_length = inner * tensor.shape[dim];

            // Visualize the moving window on each tensor's flat data array:
            //
            // a & b, all shape in (2, 2)
            //
            // a flat data: [a00, a01, a10, a11]
            // b flat data: [b00, b01, b10, b11]
            //
            // Dim: 1
            // New Shape: (2, 4)
            // New Stride: (4, 1)
            // New Flat Data: [0, 0, 0, 0, 0, 0, 0, 0]
            //
            // outer = 2
            // inner = 1
            //
            // At outer_index = 0
            //
            // For tensor a:
            // block_length = 1 * 2 = 2
            // start = 0 * 2 = 0
            // end = 0 + 2 = 2
            //
            // With a block length of 2, it will move like:
            // [ [ a00, a01 ] ->, a10, a11]
            //
            // Then it will move the selected range of flat data to the new_data
            // new_data after extend: [a00, a01]
            //
            // For tensor b:
            // block_length = 1 * 2 = 2
            // start = 0 * 2 = 0
            // end = 0 + 2 = 2
            //
            // [ [ b00, b01 ] ->, b10, b11]
            // new_data after extend: [a00, a01, b00, b01]
            //
            // At outer_index = 1
            //
            // For tensor a:
            // start = 1 * 2 = 2
            // end = 2 + 2 = 4
            //
            // [a00, a01, [ a10, a11 ] ->]
            // new_data after extend: [a00, a01, b00, b01, a10, a11]
            //
            // For tensor b:
            // start = 1 * 2 = 2
            // end = 2 + 2 = 4
            //
            // [b00, b01, [ b10, b11 ] ->]
            // new_data after extend:
            // [a00, a01, b00, b01, a10, a11, b10, b11]
            let start = outer_index * block_length;
            let end = start + block_length;

            new_data.extend(tensor_data[start..end].iter().copied());
        }
    }

    Ok(TinyTensor {
        strides: new_stride,
        shape: new_shape,
        data: Arc::new(RwLock::new(new_data)),
        offset: 0,
    })
}

fn concatenate_on_zero_dimension(tensors: &[TinyTensor]) -> Result<TinyTensor> {
    let new_shape = compute_new_shape_for_concatenation(tensors, 0);
    let new_strides = TinyTensor::compute_strides(&new_shape);
    let mut new_data = Vec::with_capacity(new_shape.iter().product::<usize>());

    for tensor in tensors.iter() {
        let tensor_data = make_contiguous_data(tensor.clone())?;

        // Because dim = 0, each tensor is copied as one whole block.
        let block_length = tensor_data.len();
        let start = 0;
        let end = start + block_length;

        // Visualize concatenating on dimension 0:
        //
        // a & b, all shape in (2, 2)
        //
        // a flat data: [a00, a01, a10, a11]
        // b flat data: [b00, b01, b10, b11]
        //
        // Dim: 0
        // New Shape: (4, 2)
        // New Stride: (2, 1)
        // New Flat Data: [0, 0, 0, 0, 0, 0, 0, 0]
        //
        // Because dim = 0, each tensor is one whole block.
        // It does not need to move row by row like dim = 1.
        //
        // For tensor a:
        // block_length = a.shape[0] * a.shape[1] = 2 * 2 = 4
        // start = 0
        // end = 4
        //
        // With a block length of 4, it will move like:
        // [ [ a00, a01, a10, a11 ] -> ]
        //
        // Then it will move the selected range of flat data to the new_data
        // new_data after extend: [a00, a01, a10, a11]
        //
        // For tensor b:
        // block_length = b.shape[0] * b.shape[1] = 2 * 2 = 4
        // start = 0
        // end = 4
        //
        // [ [ b00, b01, b10, b11 ] -> ]
        // new_data after extend:
        // [a00, a01, a10, a11, b00, b01, b10, b11]
        new_data.extend(tensor_data[start..end].iter().copied());
    }

    Ok(TinyTensor {
        strides: new_strides,
        shape: new_shape,
        data: Arc::new(RwLock::new(new_data)),
        offset: 0,
    })
}

fn compute_new_shape_for_concatenation(tensors: &[TinyTensor], dim: usize) -> Vec<usize> {
    let mut new_shape = tensors[0].shape.clone();
    let mut insertion_dim_shape = 0;
    for tensor in tensors.iter() {
        insertion_dim_shape += tensor.shape[dim];
    }
    new_shape[dim] = insertion_dim_shape;
    new_shape
}

pub fn concatenate(a: &TinyTensor, b: &TinyTensor, dim: usize) -> Result<TinyTensor> {
    let tensors = [a.clone(), b.clone()];
    concatenate_all(&tensors, dim)
}

pub fn concatenate_all(tensors: &[TinyTensor], dim: usize) -> Result<TinyTensor> {
    if tensors.is_empty() {
        return Err(anyhow!("No tensors to concatenate with"));
    }

    if dim >= tensors[0].rank() {
        return Err(anyhow!("Dimension is out of bounds"));
    }

    for tensor in tensors.iter() {
        if tensor.rank() != tensors[0].rank() {
            return Err(anyhow!("Tensor rank mismatch during concatenation"));
        }

        for dimension_index in 0..tensors[0].rank() {
            if dimension_index != dim
                && tensor.shape[dimension_index] != tensors[0].shape[dimension_index]
            {
                return Err(anyhow!("Shape mismatch during concatenation"));
            }
        }
    }

    if tensors.len() == 1 {
        return Ok(tensors[0].clone());
    }

    // Concatenating on dim 0 is the simple cat0 path: copy one whole tensor,
    // then copy the next whole tensor, then copy the next whole tensor.
    if dim == 0 {
        return concatenate_on_zero_dimension(tensors);
    }

    // Concatenating on later dimensions needs the generic path: copy a small
    // window from each tensor, then move to the next outer block.
    concatenate_contiguous(tensors, dim)
}

pub enum Reduction {
    /// Calculate the max value of a given dimension of a tensor,
    /// then reducing the given dimension to 1 with the only value being the max value.
    Max,
    Sum,
    ArgMax,
}

fn reduce_dim(tensor: &TinyTensor, operation: Reduction, dim: usize) -> Result<TinyTensor> {
    if dim >= tensor.shape.len() {
        return Err(anyhow!("Dimension exceeds the maximum"));
    }

    if tensor.shape[dim] == 0 {
        return Err(anyhow!("Can't reduce empty dimensions"));
    }

    let mut new_shape: Vec<usize> = tensor.shape.clone();
    new_shape[dim] = 1;

    // For example,
    // shape = [2, 2, 2]
    // dim = 1
    // data =
    // [
    //      [
    //          [1, 2], [3, 4]
    //      ],
    //      [
    //          [5, 6], [7, 8]
    //      ],
    // ]
    //
    // new shape = [2, 1, 2]
    //
    // iterater over all data (old shape):
    // [0][0][0] = 1
    // [0][0][1] = 2
    // [0][1][0] = 3
    // [0][1][1] = 4
    // [1][0][0] = 5
    // [1][0][1] = 6
    // [1][1][0] = 7
    // [1][1][1] = 8
    //
    // iterate over all data (new shape)
    // [0][0][0] = 3
    // [0][0][1] = 4
    // [1][0][0] = 7
    // [1][0][1] = 8

    let new_data_length = new_shape.iter().product();
    let mut new_data = Vec::with_capacity(new_data_length);
    let old_data = tensor.data.read().unwrap();

    for index in 0..new_data_length {
        let base_offset =
            compute_offset_from_linear_index(index, &new_shape, &tensor.strides, tensor.offset);

        match operation {
            Reduction::Max => {
                inner_reduce_dim_to_max(tensor, dim, &mut new_data, &old_data, base_offset)
            }
            Reduction::Sum => {
                inner_reduce_dim_to_sum(tensor, dim, &mut new_data, &old_data, base_offset)
            }
            Reduction::ArgMax => {
                inner_reduce_dim_to_argmax(tensor, dim, &mut new_data, &old_data, base_offset)
            }
        }
    }

    Ok(TinyTensor::new_from_vec(new_data, &new_shape)?)
}

fn inner_reduce_dim_to_max(
    tensor: &TinyTensor,
    dim: usize,
    new_data: &mut Vec<f32>,
    old_data: &[f32],
    base_offset: usize,
) {
    let mut max_value = old_data[base_offset];

    for dim_index in 1..tensor.shape[dim] {
        let old_data_offset = base_offset + tensor.strides[dim] * dim_index;
        max_value = max_value.max(old_data[old_data_offset]);
    }

    new_data.push(max_value);
}

fn inner_reduce_dim_to_sum(
    tensor: &TinyTensor,
    dim: usize,
    new_data: &mut Vec<f32>,
    old_data: &[f32],
    base_offset: usize,
) {
    let mut sum = old_data[base_offset];

    for dim_index in 1..tensor.shape[dim] {
        let old_data_offset = base_offset + tensor.strides[dim] * dim_index;
        sum += old_data[old_data_offset];
    }

    new_data.push(sum);
}

/// Finds the index of the maximum value along `dim`.
fn inner_reduce_dim_to_argmax(
    tensor: &TinyTensor,
    dim: usize,
    new_data: &mut Vec<f32>,
    old_data: &[f32],
    base_offset: usize,
) {
    let mut max_value = (0, old_data[base_offset]);

    for dim_index in 1..tensor.shape[dim] {
        let old_data_offset = base_offset + tensor.strides[dim] * dim_index;
        let old_data_value = old_data[old_data_offset];
        if old_data_value > max_value.1 {
            max_value = (dim_index, old_data_value)
        }
    }

    new_data.push(max_value.0 as f32);
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOperation {
    EulerExponential,
    Square,
    SquareRoot,
    MultiplyScalar(f32),
    Silu,
}

pub fn compute_unary_operations(
    tensor: TinyTensor,
    unary_operation: UnaryOperation,
) -> Result<TinyTensor> {
    let old_data = tensor.data.read().unwrap();
    let data_length = tensor.shape.iter().product();
    let mut new_data = Vec::with_capacity(data_length);

    for index in 0..data_length {
        let old_data_offset =
            compute_offset_from_linear_index(index, &tensor.shape, &tensor.strides, tensor.offset);

        let result = match unary_operation {
            UnaryOperation::EulerExponential => old_data[old_data_offset].exp(),
            UnaryOperation::Square => old_data[old_data_offset] * old_data[old_data_offset],
            UnaryOperation::SquareRoot => old_data[old_data_offset].sqrt(),
            UnaryOperation::MultiplyScalar(scale) => old_data[old_data_offset] * scale,
            UnaryOperation::Silu => {
                old_data[old_data_offset] / (1.0 + (-old_data[old_data_offset]).exp())
            }
        };

        new_data.push(result)
    }

    Ok(TinyTensor::new_from_vec(new_data, &tensor.shape)?)
}

pub fn compute_euler_exponential(tensor: TinyTensor) -> Result<TinyTensor> {
    compute_unary_operations(tensor, UnaryOperation::EulerExponential)
}

/// Normalizes elements along the specified dimension to values between 0 and 1.
///
/// The values along that dimension sum to 1, so they can be interpreted as
/// probabilities.
pub fn softmax(a: &TinyTensor, dim: usize) -> Result<TinyTensor> {
    let max = reduce_dim(a, Reduction::Max, dim)?;
    let shifted = broadcast_subtract(a, &max)?;
    let exp = compute_euler_exponential(shifted)?;
    let sum = reduce_dim(&exp, Reduction::Sum, dim)?;

    broadcast_divide(&exp, &sum)
}

pub fn square(a: TinyTensor) -> Result<TinyTensor> {
    compute_unary_operations(a, UnaryOperation::Square)
}

pub fn square_root(a: TinyTensor) -> Result<TinyTensor> {
    compute_unary_operations(a, UnaryOperation::SquareRoot)
}

pub fn argmax(a: &TinyTensor, dim: usize) -> Result<TinyTensor> {
    reduce_dim(a, Reduction::ArgMax, dim)
}

/// Computes the mean across the specified dimensions,
/// retaining reduced dimensions with size `1`.
pub fn mean(a: &TinyTensor, dims: &[usize]) -> Result<TinyTensor> {
    let mut reduced_dimension_count: usize = 1;
    let mut seen_dimensions: Vec<bool> = vec![false; a.rank()];
    let mut result = a.clone();

    for dim in dims {
        if *dim >= a.rank() {
            return Err(anyhow!("Dimension cannot exceed the rank"));
        }

        if seen_dimensions[*dim] {
            return Err(anyhow!("Duplicated dimension found"));
        }

        seen_dimensions[*dim] = true;

        reduced_dimension_count *= a.shape[*dim];

        result = reduce_dim(&result, Reduction::Sum, *dim)?;
    }

    let scale: f32 = 1.0 / reduced_dimension_count as f32;

    compute_unary_operations(result, UnaryOperation::MultiplyScalar(scale))
}

/// Paper: https://arxiv.org/pdf/1702.03118
pub fn silu(a: TinyTensor) -> Result<TinyTensor> {
    compute_unary_operations(a, UnaryOperation::Silu)
}

/// In PyTorch, it does not allow repeats to be lower rank than the tensor.
/// However, in Candle, this is allowed.
///
/// This implementation decided to follow the PyTorch way.
pub fn repeat(a: &TinyTensor, repeats: &[usize]) -> Result<TinyTensor> {
    if repeats.len() < a.rank() {
        return Err(anyhow!(
            "the number of repeat dimensions must be at least the tensor rank"
        ));
    }

    let mut tensor = if repeats.len() > a.rank() {
        reshape(
            a.clone(),
            &[vec![1; repeats.len() - a.rank()], a.shape.to_owned()].concat(),
        )?
    } else {
        a.clone()
    };

    for (index, repeat) in repeats.iter().enumerate() {
        if *repeat > 1 {
            tensor = concatenate_all(&vec![tensor; *repeat], index)?;
        }
    }

    Ok(tensor)
}
