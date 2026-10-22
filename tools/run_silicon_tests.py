#!/usr/bin/env python3
"""
tools/run_silicon_tests.py

Automated runner for executing and validating snn_core verification tests
directly on a Seeed Studio Wio-E5 MCU (STM32WLE5JC / Cortex-M4 @ 48 MHz).

This script:
1. Compiles the bare-metal `silicon_test_runner` binary in release mode.
2. Flashes the chip via `probe-rs run --chip STM32WLE5JC`.
3. Captures and analyzes real-time RTT telemetry.
4. Verifies that 100% of physical hardware assertions succeed (exit code 0).
"""

import sys
import os
import subprocess
import time

CHIP_NAME = "STM32WLE5JC"
CRATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wio_e5_benchmark"))
BIN_PATH = os.path.join(CRATE_DIR, "target/thumbv7em-none-eabi/release/silicon_test_runner")

def main():
    print("================================================================================")
    print("🚀 WIO-E5 SILICON TEST AUTOMATION RUNNER (STM32WLE5JC @ 48 MHz)")
    print("================================================================================\n")

    # 1. Bare-metal compilation
    print("🔨 Compiling bare-metal silicon_test_runner binary...")
    build_cmd = ["cargo", "build", "--release", "--bin", "silicon_test_runner"]
    build_proc = subprocess.run(build_cmd, cwd=CRATE_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if build_proc.returncode != 0:
        print("❌ Compilation error:")
        print(build_proc.stderr)
        sys.exit(1)
    print("✅ Compilation succeeded.\n")

    # 2. On-chip execution via probe-rs
    print(f"⚡ Flashing & executing on {CHIP_NAME} via probe-rs...")
    run_cmd = ["probe-rs", "run", "--chip", CHIP_NAME, BIN_PATH]
    
    proc = subprocess.Popen(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    captured_lines = []
    success = False
    failure = False
    start_time = time.time()
    
    try:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                captured_lines.append(line.rstrip())
                print(line, end="")
                if "TOTAL SUCCESS" in line:
                    success = True
                    break
                if "FAILURE" in line or "FAIL" in line:
                    failure = True
                    break
            if time.time() - start_time > 20: # 20s timeout
                print("\n⏱️  Execution timeout reached (20s).")
                break
    finally:
        proc.kill()
        proc.wait()

    print("\n--------------------------------------------------------------------------------")
    if success and not failure:
        print("🎉 SILICON VALIDATION REPORT: 100% OF PHYSICAL HARDWARE TESTS PASSED!")
        print("--------------------------------------------------------------------------------\n")
        sys.exit(0)
    else:
        print("❌ SILICON TESTS FAILED OR TIMED OUT.")
        print("--------------------------------------------------------------------------------\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
