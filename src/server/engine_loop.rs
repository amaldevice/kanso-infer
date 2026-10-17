//! The engine's one and only step loop (#46).
//!
//! `Engine::step` is heavy — a full forward per step — and owns the shared KV pool, so
//! exactly one thread ever calls it. Requests arrive as commands on a std channel; each
//! request's tokens leave on its own `tokio::sync::mpsc` channel. A step driven by one
//! client therefore cannot consume another client's output, and no async worker thread
//! ever waits on the engine.

use crate::engine::Engine;
use crate::error::EngineError;
use crate::scheduler::{SamplingParams, SequenceStatus};
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender, channel};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio::sync::oneshot;

/// What a request's private stream carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// One generated token id.
    Token(u32),
    /// The sequence retired (stop token or `max_tokens`); no further events follow.
    Finished,
    /// A fatal engine error ended the run.
    Error(String),
}

/// Engine occupancy as the loop sees it.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoadStats {
    pub running: usize,
    pub waiting: usize,
    pub free_blocks: usize,
}

/// Why a submit did not become a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitError {
    /// Admission refused: the engine holds `waiting` queued / `active` live requests.
    QueueFull { waiting: usize, active: usize },
    /// The loop thread is gone; no more work can be scheduled.
    Shutdown,
}

struct Request {
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    stop_token_ids: Vec<u32>,
    sampling: SamplingParams,
    events: UnboundedSender<Event>,
    ack: oneshot::Sender<Result<u64, SubmitError>>,
}

enum Command {
    Submit(Box<Request>),
    Stats(oneshot::Sender<LoadStats>),
    Shutdown,
}

/// Handle to the dedicated engine thread. Dropping it stops the loop.
pub struct EngineLoop {
    tx: Sender<Command>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl EngineLoop {
    /// Takes ownership of `engine` and runs it on its own thread.
    pub fn spawn(engine: Engine, max_queue: Option<usize>) -> Self {
        let (tx, rx) = channel();
        let thread = std::thread::Builder::new()
            .name("engine-loop".to_string())
            .spawn(move || run(engine, max_queue, rx))
            .expect("failed to spawn the engine loop thread");
        Self {
            tx,
            thread: Some(thread),
        }
    }

    /// Queues a request and waits for the loop to admit it. The returned receiver is that
    /// request's own token stream — nobody else can read from it.
    pub async fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        stop_token_ids: Vec<u32>,
        sampling: SamplingParams,
    ) -> Result<(u64, UnboundedReceiver<Event>), SubmitError> {
        let (events_tx, events_rx) = unbounded_channel();
        let (ack_tx, ack_rx) = oneshot::channel();
        let command = Command::Submit(Box::new(Request {
            prompt_tokens,
            max_tokens,
            stop_token_ids,
            sampling,
            events: events_tx,
            ack: ack_tx,
        }));
        if self.tx.send(command).is_err() {
            return Err(SubmitError::Shutdown);
        }
        match ack_rx.await {
            Ok(Ok(id)) => Ok((id, events_rx)),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(SubmitError::Shutdown),
        }
    }

    /// Current engine occupancy, or `None` once the loop has stopped.
    pub async fn stats(&self) -> Option<LoadStats> {
        let (tx, rx) = oneshot::channel();
        self.tx.send(Command::Stats(tx)).ok()?;
        rx.await.ok()
    }
}

impl Drop for EngineLoop {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Shutdown);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn run(mut engine: Engine, max_queue: Option<usize>, rx: Receiver<Command>) {
    let mut subscribers: HashMap<u64, UnboundedSender<Event>> = HashMap::new();

    loop {
        // Block for work, then take everything already queued: a burst of concurrent
        // submits is admitted against one consistent engine state.
        let Ok(first) = rx.recv() else {
            return;
        };
        if !apply(first, &mut engine, max_queue, &mut subscribers) {
            return;
        }
        for command in rx.try_iter() {
            if !apply(command, &mut engine, max_queue, &mut subscribers) {
                return;
            }
        }

        while engine.has_unfinished_requests() {
            match engine.step() {
                Ok(outputs) => {
                    for out in outputs {
                        // ponytail: a client that disconnects mid-stream leaves its
                        // sequence running to `max_tokens` (`send` just fails); add
                        // per-request cancellation when disconnects must free KV early.
                        if let Some(tx) = subscribers.get(&out.request_id) {
                            let _ = tx.send(Event::Token(out.token_id));
                            if out.is_finished {
                                let _ = tx.send(Event::Finished);
                                subscribers.remove(&out.request_id);
                            }
                        }
                    }
                }
                Err(e) => {
                    fail_all(&mut engine, &mut subscribers, &e);
                    break;
                }
            }
            // Retired sequences leave the engine here, so a long-lived server does not
            // rescan them every step and their KV blocks are already back in the pool.
            engine.drain_finished();

            for command in rx.try_iter() {
                if !apply(command, &mut engine, max_queue, &mut subscribers) {
                    return;
                }
            }
        }
    }
}

/// Handles one command; `false` means the loop must stop.
fn apply(
    command: Command,
    engine: &mut Engine,
    max_queue: Option<usize>,
    subscribers: &mut HashMap<u64, UnboundedSender<Event>>,
) -> bool {
    match command {
        Command::Shutdown => return false,
        Command::Stats(tx) => {
            let _ = tx.send(LoadStats {
                running: engine.running_count(),
                waiting: engine.waiting_count(),
                free_blocks: engine.free_blocks_count(),
            });
        }
        Command::Submit(request) => {
            let active = engine.sequences.iter().filter(|s| !s.is_finished()).count();
            let waiting = engine.waiting_count();
            if max_queue
                .is_some_and(|limit| waiting >= limit || active >= engine.max_running + limit)
            {
                let _ = request
                    .ack
                    .send(Err(SubmitError::QueueFull { waiting, active }));
                return true;
            }
            let id = engine.add_request_with_sampling(
                request.prompt_tokens,
                request.max_tokens,
                request.stop_token_ids,
                request.sampling,
            );
            subscribers.insert(id, request.events);
            let _ = request.ack.send(Ok(id));
        }
    }
    true
}

/// Fatal step error: tell every subscriber and retire the sequences, so the loop cannot
/// spin on the same failure. A fatal engine error means the server is done anyway.
fn fail_all(
    engine: &mut Engine,
    subscribers: &mut HashMap<u64, UnboundedSender<Event>>,
    error: &EngineError,
) {
    eprintln!("engine loop: fatal step error: {error}");
    for (_, tx) in subscribers.drain() {
        let _ = tx.send(Event::Error(error.to_string()));
    }
    for seq in engine.sequences.iter_mut() {
        seq.status = SequenceStatus::Finished;
    }
    engine.drain_finished();
}
