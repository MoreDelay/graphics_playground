use std::num::NonZeroUsize;

pub struct GaussFilter {
    sigma: f32,
    radius: NonZeroUsize,
}

impl GaussFilter {
    pub fn new(sigma: f32) -> Option<Self> {
        if sigma <= 0. {
            return None;
        }

        #[expect(clippy::cast_possible_truncation)]
        let radius = (3. * sigma).ceil() as usize;
        let radius = NonZeroUsize::new(radius)?;
        Some(Self { sigma, radius })
    }

    pub fn blur_kernel(&self) -> Vec<f32> {
        let &Self { sigma, radius } = self;
        let radius = radius.get();

        let mut kernel = vec![0.; radius + 1 + radius];
        for (i, k) in kernel.iter_mut().enumerate() {
            #[expect(clippy::cast_precision_loss)]
            let t = i as f32 - radius as f32;
            let factor = 1. / ((2. * std::f32::consts::PI).sqrt() * sigma);
            let exponent = (-t * t) / (2. * sigma * sigma);
            let gauss_value = factor * exponent.exp();

            *k = gauss_value;
        }
        let total: f32 = kernel.iter().sum();
        for v in &mut kernel {
            *v /= total;
        }
        kernel
    }
}
