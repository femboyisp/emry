//! Binary entrypoint for the `emry` CLI.

use emry_cli::execute_from;
use std::process;

fn main() -> process::ExitCode {
    execute_from(std::env::args())
}
