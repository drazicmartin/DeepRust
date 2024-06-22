use super::network;

pub struct LossFunction<T,V,L> {
    pub derivate: fn(T, V) -> L,
    pub forward: fn(T, V) -> L,
}

pub const LOSS_BCE: LossFunction<f32, f32, f32> =  LossFunction {
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


fn softmax(a: Vec<f32>, idx: usize) -> f32 {
    (a[idx]).exp() / (a.iter().map(|x| x.exp()).sum::<f32>() + network::EPSILON)
}

// let Loss = - H(q,p) = - Sum(pi * log(qi))
// with pi the target probalibilty (one hot encoded meaning [0,0,0,1,0,0] if the idx 3 is the correct class)
//          also yi is the idx of the correct class, here yi = 3
//      qi the output of the softmax(qi)
// so this simplify the Loss(q, yi) = - p_(yi) * log(q_yi)
//                                  = - log(q_yi)             (as p_k = 0 for k != yi)
// 
// let's rememer the partial derivative of softmax(z_i) with respect to z_k
// if k==i : d softmax(z_i) / dz_k = softmax(z_i)*(1-softmax(z_i))
// else    : d softmax(z_i) / dz_i = - softmax(z_k)*softmax(z_i)

// after caculous this gave use d Loss(z_i, yi) / d z_i :
// if yi == i : d Loss(z_i, yi) / d z_i = softmax(z_i) - 1
// if yi != i : d Loss(z_i, yi) / d z_i = softmax(z_i)

pub const LOSS_CROSS_ENTROPY: LossFunction<Vec<f32>, usize, Vec<f32>> =  LossFunction {
    derivate: |a: Vec<f32>, y_hat: usize| -> Vec<f32> {
        let mut res: Vec<f32> = vec![0f32; a.len()];
        for idx in 0..a.len(){
            if y_hat == idx {
                res[idx] = softmax(a.clone(), idx) - 1f32;
            }else {
                res[idx] = softmax(a.clone(), idx);
            }
        }
        res
    },
    forward: |a: Vec<f32>, y_hat: usize| -> Vec<f32> {
        vec![softmax(a, y_hat).log10()]
    },
};