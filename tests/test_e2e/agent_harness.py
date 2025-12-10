"""Agent Test Harness for Vox.

This harness enables AI agents (like Claude) to orchestrate end-to-end tests
by providing a structured interface for:
1. Launching/controlling the Vox app
2. Simulating user inputs (hotkeys, audio)
3. Observing application state
4. Verifying text insertion in target apps
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TestContext:
    """State passed to agent for test orchestration."""

    app_running: bool = False
    app_pid: int | None = None
    current_icon_state: str = "mic_off"  # "mic_on" | "mic_off"
    is_recording: bool = False
    last_transcription: str | None = None
    target_app: str | None = None  # App where text is being inserted
    target_app_content: str | None = None  # Content of target text field
    error: str | None = None


@dataclass
class HarnessAction:
    """Action the agent can request."""

    action: str  # See AVAILABLE_ACTIONS
    params: dict[str, Any] = field(default_factory=dict)


AVAILABLE_ACTIONS = {
    "launch_app": "Launch the Vox application",
    "quit_app": "Quit the Vox application",
    "press_hotkey": "Simulate hotkey press (params: key, duration_ms)",
    "release_hotkey": "Simulate hotkey release (params: key)",
    "play_audio": "Play audio file to microphone (params: file_path)",
    "speak_text": "Use TTS to speak text (params: text)",
    "open_target_app": "Open app for text insertion (params: app_name)",
    "focus_text_field": "Focus a text field in target app",
    "get_target_content": "Get current content of focused text field",
    "clear_target_content": "Clear the target text field",
    "wait": "Wait for specified duration (params: seconds)",
    "get_state": "Get current TestContext",
    "assert_contains": "Assert target contains text (params: expected)",
    "assert_recording": "Assert recording state (params: expected_state)",
}


class AgentTestHarness:
    """Harness for agent-orchestrated E2E tests.

    The agent communicates via structured JSON actions and receives
    structured JSON state updates.
    """

    def __init__(self, vox_app_path: str = "vox") -> None:
        """Initialize the test harness.

        Args:
            vox_app_path: Path to the vox command or app.
        """
        self.vox_app_path = vox_app_path
        self.app_process: subprocess.Popen | None = None
        self.context = TestContext()

    def execute_action(self, action: HarnessAction) -> TestContext:
        """Execute an action and return updated context.

        Args:
            action: The action to execute.

        Returns:
            Updated test context after action.
        """
        handler = getattr(self, f"_action_{action.action}", None)
        if handler is None:
            self.context.error = f"Unknown action: {action.action}"
            return self.context

        try:
            handler(**action.params)
            self.context.error = None
        except Exception as e:
            self.context.error = str(e)

        return self.context

    def _action_launch_app(self) -> None:
        """Launch Vox in test mode."""
        env = os.environ.copy()
        env["VOX_TEST_MODE"] = "1"  # Enable test hooks

        self.app_process = subprocess.Popen(
            ["uv", "run", "vox"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(2)  # Wait for app to start

        self.context.app_running = True
        self.context.app_pid = self.app_process.pid

    def _action_quit_app(self) -> None:
        """Quit Vox."""
        if self.app_process:
            self.app_process.terminate()
            try:
                self.app_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.app_process.kill()

        self.context.app_running = False
        self.context.app_pid = None

    def _action_press_hotkey(self, key: str, duration_ms: int = 0) -> None:
        """Simulate hotkey press via AppleScript.

        Args:
            key: The key to press (option, escape, etc.)
            duration_ms: How long to hold the key in milliseconds.
        """
        key_map = {
            "option": "option down",
            "escape": "key code 53",  # escape key code
            "cmd+shift+d": 'keystroke "d" using {command down, shift down}',
        }

        key_action = key_map.get(key, key)

        if "down" in key_action:
            script = f"""
            tell application "System Events"
                key down {key_action.replace(" down", "")}
            end tell
            """
        else:
            script = f"""
            tell application "System Events"
                {key_action}
            end tell
            """

        subprocess.run(["osascript", "-e", script], capture_output=True)

        if duration_ms > 0:
            time.sleep(duration_ms / 1000)

    def _action_release_hotkey(self, key: str) -> None:
        """Simulate hotkey release.

        Args:
            key: The key to release.
        """
        key_map = {"option": "option"}

        key_name = key_map.get(key, key)

        script = f"""
        tell application "System Events"
            key up {key_name}
        end tell
        """
        subprocess.run(["osascript", "-e", script], capture_output=True)

    def _action_play_audio(self, file_path: str) -> None:
        """Route audio file to virtual microphone.

        Note: Requires BlackHole or similar virtual audio device.

        Args:
            file_path: Path to audio file to play.
        """
        # This requires a virtual audio device setup
        subprocess.run(
            [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-f",
                "lavfi",
                "-i",
                f"amovie={file_path}",
                "-af",
                "aformat=sample_rates=16000",
            ],
            capture_output=True,
        )

    def _action_speak_text(self, text: str) -> None:
        """Use macOS TTS to speak text.

        Args:
            text: Text to speak.
        """
        subprocess.run(["say", "-v", "Alex", text])

    def _action_open_target_app(self, app_name: str) -> None:
        """Open target app for text insertion testing.

        Args:
            app_name: Name of app to open (e.g., "TextEdit").
        """
        script = f"""
        tell application "{app_name}"
            activate
            make new document
        end tell
        """
        subprocess.run(["osascript", "-e", script], capture_output=True)
        time.sleep(1)
        self.context.target_app = app_name

    def _action_focus_text_field(self) -> None:
        """Focus a text field in the target app."""
        # Most apps focus text field automatically on new document
        pass

    def _action_get_target_content(self) -> None:
        """Get content of focused text field."""
        script = """
        tell application "System Events"
            set frontApp to name of first application process whose frontmost is true
            tell process frontApp
                try
                    set focusedElement to value of attribute "AXFocusedUIElement"
                    return value of focusedElement
                on error
                    return ""
                end try
            end tell
        end tell
        """
        result = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True
        )
        self.context.target_app_content = result.stdout.strip()

    def _action_clear_target_content(self) -> None:
        """Clear target text field."""
        script = """
        tell application "System Events"
            keystroke "a" using command down
            key code 51  -- delete
        end tell
        """
        subprocess.run(["osascript", "-e", script], capture_output=True)
        self.context.target_app_content = ""

    def _action_wait(self, seconds: float) -> None:
        """Wait for specified duration.

        Args:
            seconds: Duration to wait.
        """
        time.sleep(seconds)

    def _action_get_state(self) -> None:
        """Refresh state from running app."""
        # In a full implementation, this would query the app's state
        pass

    def _action_assert_contains(self, expected: str) -> None:
        """Assert target content contains expected text.

        Args:
            expected: Text expected to be in target.

        Raises:
            AssertionError: If expected text not found.
        """
        self._action_get_target_content()
        content = self.context.target_app_content or ""
        if expected.lower() not in content.lower():
            raise AssertionError(f"Expected '{expected}' in '{content}'")

    def _action_assert_recording(self, expected_state: bool) -> None:
        """Assert recording state.

        Args:
            expected_state: Expected recording state.

        Raises:
            AssertionError: If state doesn't match.
        """
        if self.context.is_recording != expected_state:
            raise AssertionError(
                f"Expected recording={expected_state}, got {self.context.is_recording}"
            )

    def to_agent_prompt(self) -> str:
        """Generate prompt for agent with current state and available actions.

        Returns:
            Formatted prompt string for the agent.
        """
        return f"""
## Vox E2E Test Harness

### Current State
```json
{json.dumps(asdict(self.context), indent=2)}
```

### Available Actions
{json.dumps(AVAILABLE_ACTIONS, indent=2)}

### Action Format
To execute an action, respond with JSON:
```json
{{"action": "action_name", "params": {{"key": "value"}}}}
```

### Test Objectives
1. Verify latch mode works (hold Option → speak → release → text appears)
2. Verify toggle mode works (double-tap Option → speak → double-tap → text appears)
3. Verify escape cancels recording
4. Verify text appears in target application
5. Verify app is robust over extended use (10+ dictation cycles)
"""

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.app_process and self.context.app_running:
            self._action_quit_app()
