pub trait TimeProvider {
    /// Returns the current simulation time step.
    fn current_step(&self) -> usize;

    /// Increments the simulation time by one step.
    fn advance(&mut self);

    /// Returns true if the simulation should continue, false if it has reached the end.
    fn should_continue(&self, total_steps: usize) -> bool;
}
