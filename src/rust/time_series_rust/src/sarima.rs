use std::f64;

pub struct SARIMA{
    p: usize,
    q: usize,
    d: usize,
    P: usize,
    Q: usize,
    D: usize,
    m: usize,
    ar_coeffs: Vec<f64>,
    ma_coeffs: Vec<f64>,
    seasonal_ar_coeffs: Vec<f64>,
    seasonal_ma_coeffs: Vec<f64>,
    differenced_series: Vec<f64>,
}

impl SARIMA{
    pub fn new(p: usize, d: usize, q: usize, P: usize, Q: usize, D: usize, m: usize, ar_coeffs: Vec<f64>, ma_coeffs: Vec<f64>, seasonal_ar_coeffs: Vec<f64>, seasonal_ma_coeffs: Vec<f64>) -> Self{
        SARIMA{
            p, d, q, P, Q, D, m, ar_coeffs, ma_coeffs, seasonal_ar_coeffs, seasonal_ma_coeffs, differenced_series: Vec::new()
        }
    }

    pub fn difference(&mut self, series: &[f64]) -> Vec<f64>{
        let mut differenced = series.to_vec();

        for _ in 0..self.d{
            differenced = differenced.windows(2).map(|w| w[1] - w[0]).collect()
        }

        for _ in 0..self.D{
            if differenced.len() > self.m{
                differenced = differenced.iter().skip(self.m).zip(differenced.iter()).map(|(a, b)| a - b).collect();
            }
        }

        self.differenced_series = differenced.clone();
        return differenced;
    }

    pub fn inverse_difference(&self, forecast: f64, original_series: &[f64]) -> f64{
        let last_original = original_series[original_series.len() - self.d];
        return last_original + forecast;
    }

    pub fn predict(&self, series: &[f64]) -> f64{
        let diff_series = self.differenced_series.clone();
        let mut ar_term = 0.0;
        let mut ma_term = 0.0;
        let mut seasonal_ar_term = 0.0;
        let mut seasonal_ma_term = 0.0;
        let noise = 0.0;  

        if self.p > 0{
            let p_values: Vec<f64> = diff_series.iter().rev().take(self.p).cloned().collect();
            for(i, &phi) in self.ar_coeffs.iter().enumerate(){
                if i < p_values.len(){
                    ar_term += phi * p_values[i]
                }
            }
        }

        if self.q > 0{
            let q_values: Vec<f64> = diff_series.iter().rev().take(self.q).cloned().collect();
            for(i, &theta) in self.ma_coeffs.iter().enumerate(){
                if i < q_values.len(){
                    ma_term += theta * noise;
                }
            }
        }

        if self.P > 0{
            let seasonal_P_values: Vec<f64> = diff_series.iter().rev().take(self.P * self.m).step_by(self.m).cloned().collect();
            for(i, &phi_s) in self.seasonal_ar_coeffs.iter().enumerate(){
                if i < seasonal_P_values.len(){
                    seasonal_ar_term += phi_s * seasonal_P_values[i];
                }
            }
        }

        if self.Q > 0{
            let seasonal_Q_values: Vec<f64> = diff_series.iter().rev().take(self.Q * self.m).step_by(self.m).cloned().collect();
            for(i, &theta_s) in self.seasonal_ma_coeffs.iter().enumerate(){
                if i < seasonal_Q_values.len(){
                    seasonal_ma_term += theta_s * noise;
                }
            }
        }

        let forecast = ar_term + ma_term + seasonal_ar_term + seasonal_ma_term;
        return self.inverse_difference(forecast, series);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sarima_creation() {
        let sarima = SARIMA::new(
            1, 1, 1, 1, 1, 1, 12,
            vec![0.5], vec![0.3], vec![0.2], vec![0.1]
        );
        assert_eq!(sarima.p, 1);
        assert_eq!(sarima.m, 12);
    }

    #[test]
    fn test_sarima_differencing() {
        let mut sarima = SARIMA::new(
            1, 1, 1, 0, 0, 0, 4,
            vec![0.5], vec![0.3], vec![], vec![]
        );
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let diff = sarima.difference(&series);
        assert!(diff.len() > 0);
    }

    #[test]
    fn test_sarima_predict() {
        let mut sarima = SARIMA::new(
            1, 1, 1, 1, 1, 1, 4,
            vec![0.5], vec![0.3], vec![0.2], vec![0.1]
        );
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        sarima.difference(&series);
        let prediction = sarima.predict(&series);
        assert!(!prediction.is_nan());
    }
}
