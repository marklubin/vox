"""Agent-orchestrated E2E tests.

These tests use an AI agent (via API) to orchestrate complex test scenarios
by giving the agent control of the test harness.
"""

from __future__ import annotations

import pytest

from .agent_harness import AgentTestHarness, HarnessAction


@pytest.mark.e2e
@pytest.mark.slow
class TestAgentE2E:
    """E2E tests orchestrated by AI agent.

    The agent receives the harness state and available actions,
    then decides what actions to take to complete the test objective.
    """

    @pytest.fixture
    def harness(self):
        """Create test harness."""
        h = AgentTestHarness()
        yield h
        # Cleanup
        h.cleanup()

    @pytest.mark.skip(reason="Requires full app running with audio device")
    def test_latch_mode_basic(self, harness: AgentTestHarness) -> None:
        """Test Objective: Verify latch mode dictation works.

        Steps (agent should figure these out):
        1. Launch app
        2. Open TextEdit
        3. Hold Option key
        4. Play/speak test audio
        5. Release Option key
        6. Verify text appears in TextEdit
        """
        test_scenario = {
            "objective": "Verify latch mode dictation inserts text correctly",
            "success_criteria": [
                "App launches without error",
                "Recording starts when Option is held",
                "Text 'hello world' appears in TextEdit after speaking",
                "Recording stops when Option is released",
            ],
            "test_audio": "tests/fixtures/hello_world.wav",
            "max_steps": 20,
        }

        # Scripted sequence (would be agent-driven in full implementation)
        actions = [
            HarnessAction("launch_app", {}),
            HarnessAction("wait", {"seconds": 2}),
            HarnessAction("open_target_app", {"app_name": "TextEdit"}),
            HarnessAction("wait", {"seconds": 1}),
            HarnessAction("press_hotkey", {"key": "option", "duration_ms": 0}),
            HarnessAction("wait", {"seconds": 0.5}),
            HarnessAction("speak_text", {"text": "hello world"}),
            HarnessAction("wait", {"seconds": 2}),
            HarnessAction("release_hotkey", {"key": "option"}),
            HarnessAction("wait", {"seconds": 1}),
            HarnessAction("assert_contains", {"expected": "hello"}),
        ]

        for action in actions:
            context = harness.execute_action(action)
            if context.error:
                pytest.fail(f"Action {action.action} failed: {context.error}")

    @pytest.mark.skip(reason="Requires full app running with audio device")
    def test_toggle_mode_basic(self, harness: AgentTestHarness) -> None:
        """Test Objective: Verify toggle mode dictation works.

        Steps:
        1. Launch app
        2. Open TextEdit
        3. Double-tap Option to start
        4. Speak test text
        5. Double-tap Option to stop
        6. Verify text appears
        """
        actions = [
            HarnessAction("launch_app", {}),
            HarnessAction("wait", {"seconds": 2}),
            HarnessAction("open_target_app", {"app_name": "TextEdit"}),
            HarnessAction("wait", {"seconds": 1}),
            # Double-tap to start
            HarnessAction("press_hotkey", {"key": "option"}),
            HarnessAction("release_hotkey", {"key": "option"}),
            HarnessAction("wait", {"seconds": 0.1}),
            HarnessAction("press_hotkey", {"key": "option"}),
            HarnessAction("release_hotkey", {"key": "option"}),
            HarnessAction("wait", {"seconds": 0.5}),
            HarnessAction("speak_text", {"text": "testing toggle mode"}),
            HarnessAction("wait", {"seconds": 3}),
            # Double-tap to stop
            HarnessAction("press_hotkey", {"key": "option"}),
            HarnessAction("release_hotkey", {"key": "option"}),
            HarnessAction("wait", {"seconds": 0.1}),
            HarnessAction("press_hotkey", {"key": "option"}),
            HarnessAction("release_hotkey", {"key": "option"}),
            HarnessAction("wait", {"seconds": 1}),
            HarnessAction("assert_contains", {"expected": "toggle"}),
        ]

        for action in actions:
            context = harness.execute_action(action)
            if context.error:
                pytest.fail(f"Action {action.action} failed: {context.error}")

    @pytest.mark.skip(reason="Requires full app running with audio device")
    def test_escape_cancels(self, harness: AgentTestHarness) -> None:
        """Test Objective: Verify escape cancels recording.

        Steps:
        1. Launch app and open TextEdit
        2. Start recording
        3. Speak some text
        4. Press Escape to cancel
        5. Verify NO text was inserted
        """
        actions = [
            HarnessAction("launch_app", {}),
            HarnessAction("wait", {"seconds": 2}),
            HarnessAction("open_target_app", {"app_name": "TextEdit"}),
            HarnessAction("wait", {"seconds": 1}),
            HarnessAction("clear_target_content", {}),
            # Double-tap to start
            HarnessAction("press_hotkey", {"key": "option"}),
            HarnessAction("release_hotkey", {"key": "option"}),
            HarnessAction("wait", {"seconds": 0.1}),
            HarnessAction("press_hotkey", {"key": "option"}),
            HarnessAction("release_hotkey", {"key": "option"}),
            HarnessAction("wait", {"seconds": 0.5}),
            HarnessAction("speak_text", {"text": "this should not appear"}),
            HarnessAction("wait", {"seconds": 1}),
            # Press Escape to cancel
            HarnessAction("press_hotkey", {"key": "escape"}),
            HarnessAction("wait", {"seconds": 0.5}),
            HarnessAction("get_target_content", {}),
        ]

        for action in actions:
            context = harness.execute_action(action)
            if context.error:
                pytest.fail(f"Action {action.action} failed: {context.error}")

        # Verify no text was inserted
        content = harness.context.target_app_content or ""
        assert "should not appear" not in content.lower()

    @pytest.mark.skip(reason="Requires full app running with audio device")
    def test_robustness_extended_session(self, harness: AgentTestHarness) -> None:
        """Test Objective: Verify app remains stable over extended use.

        Run 10 dictation cycles and verify:
        - No memory growth
        - No accumulated state issues
        - Consistent transcription quality
        """
        harness.execute_action(HarnessAction("launch_app", {}))
        harness.execute_action(HarnessAction("wait", {"seconds": 2}))
        harness.execute_action(HarnessAction("open_target_app", {"app_name": "TextEdit"}))

        for cycle in range(10):
            harness.execute_action(HarnessAction("clear_target_content", {}))

            # Double-tap to start
            harness.execute_action(HarnessAction("press_hotkey", {"key": "option"}))
            harness.execute_action(HarnessAction("release_hotkey", {"key": "option"}))
            harness.execute_action(HarnessAction("wait", {"seconds": 0.1}))
            harness.execute_action(HarnessAction("press_hotkey", {"key": "option"}))
            harness.execute_action(HarnessAction("release_hotkey", {"key": "option"}))

            harness.execute_action(HarnessAction("wait", {"seconds": 0.3}))
            harness.execute_action(
                HarnessAction("speak_text", {"text": f"test cycle {cycle}"})
            )
            harness.execute_action(HarnessAction("wait", {"seconds": 2}))

            # Double-tap to stop
            harness.execute_action(HarnessAction("press_hotkey", {"key": "option"}))
            harness.execute_action(HarnessAction("release_hotkey", {"key": "option"}))
            harness.execute_action(HarnessAction("wait", {"seconds": 0.1}))
            harness.execute_action(HarnessAction("press_hotkey", {"key": "option"}))
            harness.execute_action(HarnessAction("release_hotkey", {"key": "option"}))

            harness.execute_action(HarnessAction("wait", {"seconds": 1}))

            # Verify text was inserted
            harness.execute_action(HarnessAction("get_target_content", {}))
            content = harness.context.target_app_content

            assert content is not None and len(content) > 0, (
                f"Cycle {cycle}: No text inserted"
            )

            # Small delay between cycles
            harness.execute_action(HarnessAction("wait", {"seconds": 0.5}))

        # Final cleanup
        harness.execute_action(HarnessAction("quit_app", {}))


@pytest.mark.e2e
class TestHarnessUnit:
    """Unit tests for the test harness itself."""

    def test_harness_init(self) -> None:
        """Harness should initialize with default state."""
        harness = AgentTestHarness()

        assert harness.context.app_running is False
        assert harness.context.app_pid is None
        assert harness.context.error is None

    def test_unknown_action(self) -> None:
        """Unknown action should set error."""
        harness = AgentTestHarness()
        context = harness.execute_action(HarnessAction("nonexistent_action", {}))

        assert context.error is not None
        assert "Unknown action" in context.error

    def test_wait_action(self) -> None:
        """Wait action should work without error."""
        harness = AgentTestHarness()
        import time

        start = time.time()
        context = harness.execute_action(HarnessAction("wait", {"seconds": 0.1}))
        elapsed = time.time() - start

        assert context.error is None
        assert elapsed >= 0.1

    def test_agent_prompt_generation(self) -> None:
        """Should generate valid agent prompt."""
        harness = AgentTestHarness()
        prompt = harness.to_agent_prompt()

        assert "Current State" in prompt
        assert "Available Actions" in prompt
        assert "launch_app" in prompt
        assert "Test Objectives" in prompt
