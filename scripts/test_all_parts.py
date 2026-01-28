#!/usr/bin/env python3
import os
import subprocess
import sys


def run_pytest() -> bool:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..", "src")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        env=env,
        cwd=os.path.dirname(__file__) + "/..",
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1 and "No module named pytest" in result.stderr:
        return False
    raise SystemExit(result.returncode)


def smoke_imports() -> None:
    root = os.path.join(os.path.dirname(__file__), "..", "src")
    sys.path.insert(0, os.path.abspath(root))
    import gpt5_roleplay_system.server  # noqa: F401
    from gpt5_roleplay_system.controller import SessionController

    controller = SessionController(persona="Smoke", user_id="smoke")
    assert controller.memory_summary() == ""


if __name__ == "__main__":
    smoke_imports()
    if not run_pytest():
        print("pytest not installed; running fallback checks")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        src_root = os.path.join(project_root, "src")
        if src_root not in sys.path:
            sys.path.insert(0, src_root)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        import tests.test_protocol as test_protocol
        import tests.test_memory as test_memory
        import tests.test_pipeline as test_pipeline

        test_protocol.test_decode_message_with_json_string_data()
        test_protocol.test_encode_message_includes_envelope()
        test_protocol.test_build_chat_response()
        test_memory.test_rolling_buffer_trims()
        test_memory.test_conversation_memory_compresses()
        test_pipeline.test_pipeline_returns_action()
    print("All parts passed")
