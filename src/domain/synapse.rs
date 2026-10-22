use crate::domain::numeric::{Fix32, NumericScalar};

#[derive(Clone, Copy, Debug)]
pub struct Synapse<T: NumericScalar = Fix32> {
    pub source_idx: usize,
    pub target_idx: usize,
    pub weight: T,
}

impl<T: NumericScalar> Synapse<T> {
    pub fn new(source_idx: usize, target_idx: usize, weight: T) -> Self {
        Synapse {
            source_idx,
            target_idx,
            weight,
        }
    }
}
