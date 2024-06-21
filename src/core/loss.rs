use super::network;

pub struct LossFunction {
    pub derivate: fn(f32, f32) -> f32,
    pub forward: fn(f32, f32) -> f32,
}

pub const LOSS_BCE: LossFunction =  LossFunction {
    derivate: |a: f32, y_hat: f32| -> f32 {
        match a {
            a if a == 0f32 => -100f32,
            _ => -y_hat/a + (1f32-y_hat)/(1f32-a + network::EPSILON)
        }
    },
    forward: |a: f32, y_hat: f32| -> f32 {
        - y_hat * a.log10() - (1f32-y_hat)*(1f32-a + network::EPSILON).log10()
    },
};