use super::*;
impl AiState {
    /// Advance one simulated epoch every ~0.5s.
    pub fn tick_training(&mut self, dt: f64) {
        if !self.training {
            return;
        }
        self.epoch_timer += dt;
        if self.epoch_timer < 0.5 {
            return;
        }
        self.epoch_timer = 0.0;
        let progress = self.epoch as f64 / self.total_epochs as f64;
        if self.loss_history.len() < self.epoch + 1 {
            let base = 2.5 * (-3.0 * progress).exp();
            let noise = (self.epoch as f64 * 7.3).sin() * 0.1;
            let loss = (base + noise).max(0.01);
            self.loss_history.push(loss);
            self.logs.push(format!(
                "Epoch {}/{} — loss: {:.4}",
                self.epoch + 1,
                self.total_epochs,
                loss
            ));
            self.epoch += 1;
            if self.epoch >= self.total_epochs {
                self.training = false;
                self.logs.push("Training complete.".to_string());
                if let (Some(m), Some(d)) = (self.model, self.dataset) {
                    self.runs.push(RunEntry {
                        model: m.label(),
                        dataset: d.label(),
                        epochs: self.total_epochs,
                        final_loss: *self.loss_history.last().unwrap_or(&0.0),
                    });
                }
            }
        }
    }

    /// Start training with current planner configuration.
    pub fn start_training(&mut self) {
        self.training = true;
        self.epoch = 0;
        self.epoch_timer = 0.0;
        self.loss_history.clear();
        self.logs.clear();
        self.logs.push(format!(
            "Starting {} on {} with {}",
            self.model.map_or("?", |m| m.label()),
            self.dataset.map_or("?", |d| d.label()),
            self.task.map_or("?", |t| t.label()),
        ));
    }
}

