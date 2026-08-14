use std::collections::HashMap;

use crate::tensors::TinyTensor;

#[derive(Debug, Clone)]
pub struct KVCache {
    cache: HashMap<usize, (TinyTensor, TinyTensor)>,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    // Existin cache will be overwritten.
    pub fn update(&mut self, layer: usize, k: TinyTensor, v: TinyTensor) {
        self.cache.insert(layer, (k, v));
    }

    // Return a tuple, where the first is K and the second is V.
    //
    // It will MOVE the cached tensor out.
    pub fn get(&mut self, layer: usize) -> Option<&(TinyTensor, TinyTensor)> {
        self.cache.get(&layer)
    }
}
