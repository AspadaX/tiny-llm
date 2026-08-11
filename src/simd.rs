use rten_simd::ops::{BitOps, FloatOps};
use rten_simd::{Isa, SimdOp};

use crate::tensors::compute_offset_from_linear_index;

#[derive(Debug, Copy, Clone)]
pub enum BinaryOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl BinaryOperation {
    #[inline(always)]
    fn scalar(self, a: f32, b: f32) -> f32 {
        match self {
            Self::Add => a + b,
            Self::Multiply => a * b,
            Self::Divide => a / b,
            Self::Subtract => a - b,
        }
    }

    #[inline(always)]
    fn simd<O: FloatOps<f32>>(self, ops: O, a: O::Simd, b: O::Simd) -> O::Simd {
        match self {
            Self::Add => ops.add(a, b),
            Self::Multiply => ops.mul(a, b),
            Self::Divide => ops.div(a, b),
            Self::Subtract => ops.sub(a, b),
        }
    }
}

pub struct BroadcastBinaryOperation<'a> {
    pub a_data: &'a [f32],
    pub b_data: &'a [f32],
    pub output_data: &'a mut [f32],

    pub shape: &'a [usize],
    pub a_strides: &'a [usize],
    pub b_strides: &'a [usize],

    pub a_offset: usize,
    pub b_offset: usize,
    pub operation: BinaryOperation,
}

impl<'a> SimdOp for BroadcastBinaryOperation<'a> {
    type Output = ();

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let ops = isa.f32();

        let inner_length = *self.shape.last().unwrap();
        let outer_length = self.output_data.len() / inner_length;
        let a_inner_stride = *self.a_strides.last().unwrap();
        let b_inner_stride = *self.b_strides.last().unwrap();

        for outer_index in 0..outer_length {
            let offset_a = compute_offset_from_linear_index(
                outer_index,
                &self.shape[..self.shape.len() - 1],
                &self.a_strides[..self.a_strides.len() - 1],
                self.a_offset,
            );
            let offset_b = compute_offset_from_linear_index(
                outer_index,
                &self.shape[..self.shape.len() - 1],
                &self.b_strides[..self.shape.len() - 1],
                self.b_offset,
            );

            let output_start = outer_index * inner_length;
            let output_row = &mut self.output_data[output_start..output_start + inner_length];
            let is_contiguous = a_inner_stride <= 1 && b_inner_stride <= 1;

            // Fallback for arbitrary strided views.
            match is_contiguous {
                true => {
                    let is_a_stride_1 = a_inner_stride == 1;
                    let is_b_stride_1 = b_inner_stride == 1;

                    let (a_row, b_row) = match (is_a_stride_1, is_b_stride_1) {
                        (true, true) => (
                            Some(&self.a_data[offset_a..offset_a + inner_length]),
                            Some(&self.b_data[offset_b..offset_b + inner_length]),
                        ),
                        (false, false) => (None, None),
                        (true, false) => {
                            (Some(&self.a_data[offset_a..offset_a + inner_length]), None)
                        }
                        (false, true) => {
                            (None, Some(&self.b_data[offset_b..offset_b + inner_length]))
                        }
                    };

                    let vector_length = ops.len();
                    // Largest multiple of the SIMD width that fits in the row;
                    // any remaining elements are processed by the scalar tail loop.
                    let simd_length = inner_length - (inner_length % vector_length);
                    let mut inner_index = 0;

                    while inner_index < simd_length {
                        let vector_a = match a_row {
                            Some(a_row) => ops.load(&a_row[inner_index..]),
                            None => ops.splat(self.a_data[offset_a]),
                        };

                        let vector_b = match b_row {
                            Some(b_row) => ops.load(&b_row[inner_index..]),
                            None => ops.splat(self.b_data[offset_b]),
                        };

                        let result = self.operation.simd(ops, vector_a, vector_b);

                        ops.store(result, &mut output_row[inner_index..]);
                        inner_index += vector_length;
                    }

                    // Handle the final incomplete SIMD vector.
                    for inner_index in simd_length..inner_length {
                        let a = self.a_data[offset_a + a_inner_stride * inner_index];
                        let b = self.b_data[offset_b + b_inner_stride * inner_index];

                        output_row[inner_index] = self.operation.scalar(a, b);
                    }
                }
                false => {
                    for inner_index in 0..inner_length {
                        let a = self.a_data[offset_a + a_inner_stride * inner_index];
                        let b = self.b_data[offset_b + b_inner_stride * inner_index];

                        output_row[inner_index] = self.operation.scalar(a, b);
                    }
                }
            }
        }

        ()
    }
}
