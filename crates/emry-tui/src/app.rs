//! The dashboard run loop and input handling.
//!
//! [`run`] is generic over the ratatui [`Backend`], drains the event bus and an
//! input channel each frame, and redraws at ~15 Hz. It is driven by channels so
//! tests exercise the whole loop against a `TestBackend`. [`run_terminal`] is the
//! thin real-terminal shell (raw mode + a key-reader thread) that is not unit
//! tested.

use crate::ui::{render, UiState};
use crossbeam_channel::Receiver;
use emry_core::Event;
use ratatui::backend::Backend;
use ratatui::crossterm::event::{self, KeyCode, KeyEvent, KeyEventKind};
use ratatui::Terminal;
use std::io;
use std::time::Duration;

/// Maximum redraw rate: one frame per ~66 ms (≈15 Hz).
pub const FRAME_INTERVAL: Duration = Duration::from_millis(66);

/// A decoded user action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Quit the dashboard.
    Quit,
    /// Select the metric chart at this zero-based index.
    Select(usize),
    /// Toggle the paused state.
    TogglePause,
    /// No-op (key not bound).
    Ignore,
}

/// Maps a key code to an [`Action`]: `q`/`Esc` quit, `1`–`4` select a metric,
/// `p` pause.
#[must_use]
pub fn map_key(code: KeyCode) -> Action {
    match code {
        KeyCode::Char('q' | 'Q') | KeyCode::Esc => Action::Quit,
        KeyCode::Char('p' | 'P') => Action::TogglePause,
        KeyCode::Char(c @ '1'..='4') => Action::Select(c as usize - '1' as usize),
        _ => Action::Ignore,
    }
}

/// How long the loop should keep running.
#[derive(Debug, Clone, Copy)]
pub enum RunLimit {
    /// Run until quit or the run finishes (real-terminal use).
    UntilQuit,
    /// Render exactly this many frames, then stop (tests).
    Frames(usize),
}

/// Runs the dashboard loop against `terminal`, reducing `events` into `state`,
/// applying `input` actions, and redrawing each frame.
///
/// Returns when an [`Action::Quit`] is received, the frame limit is hit, or the
/// run finished and the event channel has drained.
///
/// # Errors
///
/// Returns any terminal draw error.
pub fn run<B: Backend>(
    terminal: &mut Terminal<B>,
    events: &Receiver<Event>,
    input: &Receiver<Action>,
    state: &mut UiState,
    limit: RunLimit,
    frame_interval: Duration,
) -> io::Result<()> {
    let mut frames = 0usize;
    loop {
        while let Ok(event) = events.try_recv() {
            state.apply(&event);
        }
        while let Ok(action) = input.try_recv() {
            match action {
                Action::Quit => return Ok(()),
                Action::Select(n) => state.select(n),
                Action::TogglePause => state.toggle_pause(),
                Action::Ignore => {}
            }
        }

        // Pausing freezes the display (state keeps updating underneath); the
        // dashboard also stays up after a run finishes so final values can be
        // read — it exits only on Quit (or the test frame limit).
        if !state.paused {
            terminal.draw(|frame| render(frame, state))?;
        }
        frames += 1;

        if let RunLimit::Frames(max) = limit {
            if frames >= max {
                return Ok(());
            }
        }
        if !frame_interval.is_zero() {
            std::thread::sleep(frame_interval);
        }
    }
}

/// Reads key events from the real terminal until quit, forwarding decoded
/// [`Action`]s on `tx`. Blocks; intended for a dedicated thread.
///
/// Uses `try_send` on the bounded channel: under key-autorepeat flooding while
/// the loop sleeps through a frame, excess actions are dropped rather than
/// growing the queue.
///
/// # Errors
///
/// Returns a crossterm read error.
pub fn read_keys(tx: &crossbeam_channel::Sender<Action>) -> io::Result<()> {
    use crossbeam_channel::TrySendError;
    loop {
        if let event::Event::Key(KeyEvent {
            code,
            kind: KeyEventKind::Press,
            ..
        }) = event::read()?
        {
            let action = map_key(code);
            match tx.try_send(action) {
                Ok(()) if action == Action::Quit => return Ok(()),
                Ok(()) | Err(TrySendError::Full(_)) => {}
                Err(TrySendError::Disconnected(_)) => return Ok(()),
            }
        }
    }
}

/// Real-terminal entry point: sets up raw mode + the alternate screen, spawns a
/// key-reader thread, runs the dashboard until quit, then restores the terminal.
///
/// This is the thin, untested shell over [`run`]; the loop and rendering it
/// drives are covered by `TestBackend` tests.
///
/// # Errors
///
/// Returns a terminal setup or draw error.
pub fn run_terminal(events: &Receiver<Event>, mut state: UiState) -> io::Result<()> {
    use ratatui::backend::CrosstermBackend;
    use ratatui::crossterm::execute;
    use ratatui::crossterm::terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
    };

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    let (input_tx, input_rx) = crossbeam_channel::bounded(64);
    let reader = std::thread::spawn(move || {
        let _ = read_keys(&input_tx);
    });

    let result = run(
        &mut terminal,
        events,
        &input_rx,
        &mut state,
        RunLimit::UntilQuit,
        FRAME_INTERVAL,
    );

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    let _ = reader.join();
    result
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_precision_loss)]
    use super::*;
    use crossbeam_channel::unbounded;
    use emry_core::{MetricId, Phase};
    use ratatui::backend::TestBackend;

    fn batch(step: u64, value: f64) -> Event {
        Event::MetricsBatch {
            step,
            epoch: 0,
            phase: Phase::Train,
            values: vec![(MetricId(0), value)],
        }
    }

    #[test]
    fn map_key_bindings() {
        assert_eq!(map_key(KeyCode::Char('q')), Action::Quit);
        assert_eq!(map_key(KeyCode::Esc), Action::Quit);
        assert_eq!(map_key(KeyCode::Char('p')), Action::TogglePause);
        assert_eq!(map_key(KeyCode::Char('1')), Action::Select(0));
        assert_eq!(map_key(KeyCode::Char('4')), Action::Select(3));
        assert_eq!(map_key(KeyCode::Char('x')), Action::Ignore);
    }

    #[test]
    fn loop_applies_events_then_renders_fixed_frames() {
        let (etx, erx) = unbounded();
        let (_itx, irx) = unbounded();
        for step in 0..5 {
            etx.send(batch(step, 1.0 / (step as f64 + 1.0))).unwrap();
        }
        let mut state = UiState::with_labels(&[(MetricId(0), "loss")]);
        let mut terminal = Terminal::new(TestBackend::new(60, 16)).unwrap();

        run(
            &mut terminal,
            &erx,
            &irx,
            &mut state,
            RunLimit::Frames(2),
            Duration::ZERO,
        )
        .unwrap();

        // Events drained into state before rendering.
        assert_eq!(state.step, 4);
        assert_eq!(state.metrics[0].history.len(), 5);
    }

    #[test]
    fn quit_action_returns_immediately() {
        let (_etx, erx) = unbounded::<Event>();
        let (itx, irx) = unbounded();
        itx.send(Action::Quit).unwrap();
        let mut state = UiState::default();
        let mut terminal = Terminal::new(TestBackend::new(40, 10)).unwrap();
        // Frames(1000) would loop, but Quit must short-circuit on the first pass.
        run(
            &mut terminal,
            &erx,
            &irx,
            &mut state,
            RunLimit::Frames(1000),
            Duration::ZERO,
        )
        .unwrap();
    }

    #[test]
    fn input_actions_drive_selection_and_pause() {
        let (_etx, erx) = unbounded::<Event>();
        let (itx, irx) = unbounded();
        let mut state = UiState::default();
        state.apply(&batch(0, 1.0));
        state.apply(&Event::MetricsBatch {
            step: 1,
            epoch: 0,
            phase: Phase::Train,
            values: vec![(MetricId(1), 2.0)],
        });
        itx.send(Action::Select(1)).unwrap();
        itx.send(Action::TogglePause).unwrap();

        let mut terminal = Terminal::new(TestBackend::new(40, 10)).unwrap();
        run(
            &mut terminal,
            &erx,
            &irx,
            &mut state,
            RunLimit::Frames(1),
            Duration::ZERO,
        )
        .unwrap();
        assert_eq!(state.selected, 1);
        assert!(state.paused);
    }

    #[test]
    fn paused_state_skips_redraw_but_keeps_consuming_events() {
        let (etx, erx) = unbounded();
        let (itx, irx) = unbounded();
        itx.send(Action::TogglePause).unwrap();
        etx.send(batch(0, 1.0)).unwrap();
        let mut state = UiState::default();
        let mut terminal = Terminal::new(TestBackend::new(40, 10)).unwrap();
        run(
            &mut terminal,
            &erx,
            &irx,
            &mut state,
            RunLimit::Frames(1),
            Duration::ZERO,
        )
        .unwrap();
        // Paused: event still applied to state, but the screen was not drawn.
        assert!(state.paused);
        assert_eq!(state.metrics.len(), 1);
    }
}
