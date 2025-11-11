use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_gui_flag_without_feature() {
    let mut cmd = Command::cargo_bin("rumma-cli").unwrap();
    cmd.arg("--gui")
        .assert()
        .failure()
        .stderr(predicate::str::contains("GUI feature is not enabled."));
}

#[test]
#[cfg(feature = "gui")]
fn test_gui_flag_with_feature() {
    let mut cmd = Command::cargo_bin("rumma-cli").unwrap();
    cmd.arg("--gui").features("gui").assert().success();
}
