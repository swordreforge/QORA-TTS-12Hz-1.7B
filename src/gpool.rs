//! Minimal persistent thread pool for GEMV work dispatch.
//!
//! Replaces per-call `std::thread::spawn` (TLS + clone overhead dominated
//! small GEMVs, e.g. the code predictor's 15-step loop). Workers are
//! persistent OS threads blocked on a shared job queue; `dispatch` submits
//! one closure and returns a one-shot receiver for its result.

use std::sync::{Arc, Mutex, OnceLock, mpsc};

type Job = Box<dyn FnOnce() -> Vec<f32> + Send + 'static>;

struct Pool {
    tx: mpsc::Sender<(Job, mpsc::Sender<Vec<f32>>)>,
    workers: usize,
}

fn global_pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let avail = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(6)
            .max(1);
        // QORA_THREADS overrides pool size (e.g. pin to P-cores).
        // Default caps at 16: measured saturation (two2/54f, ×2 runs each) —
        // 16: 6.9/7.0s, 20: 7.6s, 22: 7.6/7.6s. Past 16 the shared-queue
        // mutex + LP-E tails cost more than extra workers give (memory-bound
        // GEMV + small predictor GEMVs stop scaling ~12-16).
        let workers = std::env::var("QORA_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(avail.min(16));
        let actual = workers.min(avail * 2).max(1);
        let (tx, rx) = mpsc::channel::<(Job, mpsc::Sender<Vec<f32>>)>();
        let rx = Arc::new(Mutex::new(rx));
        for _ in 0..actual {
            let rx = Arc::clone(&rx);
            std::thread::spawn(move || loop {
                let (job, done) = match rx.lock().unwrap().recv() {
                    Ok(pair) => pair,
                    Err(_) => break, // pool shut down (never for the global)
                };
                let out = job();
                let _ = done.send(out);
            });
        }
        Pool { tx, workers: actual }
    })
}

/// Number of worker threads in the pool.
pub fn num_workers() -> usize {
    global_pool().workers
}

/// Submit one GEMV chunk job; returns a receiver for the partial result.
pub fn dispatch<F>(f: F) -> mpsc::Receiver<Vec<f32>>
where
    F: FnOnce() -> Vec<f32> + Send + 'static,
{
    let pool = global_pool();
    let (done_tx, done_rx) = mpsc::channel();
    pool.tx.send((Box::new(f), done_tx)).expect("pool workers alive");
    done_rx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_echo() {
        let rx = dispatch(|| vec![1.0, 2.0, 3.0]);
        assert_eq!(rx.recv().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_pool_parallel_sum() {
        // N jobs run across workers; results collected in order.
        let rxs: Vec<_> = (0..32).map(|i| dispatch(move || vec![i as f32])).collect();
        let mut got: Vec<f32> = rxs.into_iter().map(|r| r.recv().unwrap()[0]).collect();
        got.sort_by(|a, b| a.total_cmp(b));
        let expect: Vec<f32> = (0..32).map(|i| i as f32).collect();
        assert_eq!(got, expect);
    }

    #[test]
    fn test_pool_workers_sane() {
        assert!(num_workers() >= 1);
    }
}
