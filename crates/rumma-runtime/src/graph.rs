#[derive(Default, Debug)]
pub struct GraphRegistry {
    decode_captured: bool,
}

impl GraphRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mark_decode_captured(&mut self) {
        self.decode_captured = true;
    }

    pub fn decode_captured(&self) -> bool {
        self.decode_captured
    }
}
