"""Unit tests for text insertion module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-specific tests")
class TestTextInserterDarwin:
    """Tests for TextInserter class on macOS."""

    @pytest.fixture
    def mock_accessibility_enabled(self):
        """Mock accessibility APIs when accessibility is enabled."""
        with patch("vox.insert_darwin.AXIsProcessTrusted") as mock_trusted:
            with patch("vox.insert_darwin.HAS_TRUSTED_OPTIONS", False):
                mock_trusted.return_value = True
                with patch("vox.insert_darwin.AXUIElementCreateSystemWide") as mock_system:
                    with patch("vox.insert_darwin.AXUIElementCopyAttributeValue") as mock_copy:
                        with patch("vox.insert_darwin.AXUIElementSetAttributeValue") as mock_set:
                            mock_system.return_value = MagicMock()
                            mock_copy.return_value = (0, MagicMock())  # (err, element)
                            mock_set.return_value = 0  # success
                            yield {
                                "trusted": mock_trusted,
                                "system": mock_system,
                                "copy": mock_copy,
                                "set": mock_set,
                            }

    @pytest.fixture
    def mock_accessibility_disabled(self):
        """Mock accessibility APIs when accessibility is disabled."""
        with patch("vox.insert_darwin.AXIsProcessTrusted") as mock_trusted:
            with patch("vox.insert_darwin.HAS_TRUSTED_OPTIONS", False):
                mock_trusted.return_value = False
                yield mock_trusted

    @pytest.fixture
    def mock_clipboard(self):
        """Mock clipboard operations."""
        with patch("vox.insert_darwin.pyperclip") as mock_clip:
            mock_clip.paste.return_value = "original clipboard"
            with patch("vox.insert_darwin.time"):
                yield {
                    "pyperclip": mock_clip,
                }

    def test_init_with_accessibility(self, mock_accessibility_enabled) -> None:
        """Should use accessibility when available."""
        from vox.insert_darwin import TextInserter

        inserter = TextInserter()
        assert inserter.use_accessibility is True

    def test_init_without_accessibility(self, mock_accessibility_disabled) -> None:
        """Should fall back to clipboard when accessibility unavailable."""
        from vox.insert_darwin import TextInserter

        inserter = TextInserter()
        assert inserter.use_accessibility is False

    def test_insert_empty_string(self, mock_accessibility_enabled) -> None:
        """Should handle empty string insertion."""
        from vox.insert_darwin import TextInserter

        inserter = TextInserter()
        result = inserter.insert("")

        assert result is True
        # Should not call any accessibility APIs
        mock_accessibility_enabled["system"].assert_not_called()

    def test_insert_accessibility_success(self, mock_accessibility_enabled) -> None:
        """Should insert text via accessibility API."""
        from vox.insert_darwin import TextInserter

        inserter = TextInserter()
        result = inserter.insert("hello world")

        assert result is True
        mock_accessibility_enabled["set"].assert_called_once()
        # Verify the text was passed
        call_args = mock_accessibility_enabled["set"].call_args
        assert call_args[0][2] == "hello world"

    def test_insert_accessibility_failure_falls_back(
        self, mock_accessibility_enabled, mock_clipboard
    ) -> None:
        """Should fall back to clipboard when accessibility fails."""
        from vox.insert_darwin import TextInserter

        # Make accessibility fail
        mock_accessibility_enabled["copy"].return_value = (1, None)  # Error

        inserter = TextInserter()
        result = inserter.insert("test text")

        assert result is True
        # Should have used clipboard
        mock_clipboard["pyperclip"].copy.assert_called()

    def test_insert_clipboard_preserves_original(
        self, mock_accessibility_disabled, mock_clipboard
    ) -> None:
        """Should preserve and restore original clipboard contents."""
        from vox.insert_darwin import TextInserter

        inserter = TextInserter()
        result = inserter.insert("new text")

        assert result is True
        # Should have read original clipboard
        mock_clipboard["pyperclip"].paste.assert_called()
        # Should have restored original clipboard
        calls = mock_clipboard["pyperclip"].copy.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == "new text"  # First: set new text
        assert calls[1][0][0] == "original clipboard"  # Second: restore

    def test_insert_clipboard_sends_paste(
        self, mock_accessibility_disabled, mock_clipboard
    ) -> None:
        """Should send Cmd+V via Quartz CGEvent."""
        mock_quartz = MagicMock()
        mock_quartz.CGEventCreateKeyboardEvent.return_value = MagicMock()
        mock_quartz.kCGEventFlagMaskCommand = 0x100000
        mock_quartz.kCGHIDEventTap = 0

        with patch.dict(sys.modules, {"Quartz": mock_quartz}):
            from vox.insert_darwin import TextInserter

            inserter = TextInserter()
            inserter.insert("test")

            # Should have created key down and key up events
            assert mock_quartz.CGEventCreateKeyboardEvent.call_count == 2
            # Should have posted both events
            assert mock_quartz.CGEventPost.call_count == 2

    def test_check_accessibility_static(self, mock_accessibility_enabled) -> None:
        """Static method should check accessibility status."""
        from vox.insert_darwin import TextInserter

        result = TextInserter.check_accessibility()
        assert result is True

    def test_refresh_accessibility_status(self) -> None:
        """Should update accessibility status on refresh."""
        with patch("vox.insert_darwin.AXIsProcessTrusted") as mock_trusted:
            with patch("vox.insert_darwin.HAS_TRUSTED_OPTIONS", False):
                mock_trusted.return_value = False
                from vox.insert_darwin import TextInserter

                inserter = TextInserter(prompt_for_access=False)
                assert inserter.use_accessibility is False

                # Enable accessibility
                mock_trusted.return_value = True
                inserter.refresh_accessibility_status()
                assert inserter.use_accessibility is True

    def test_insert_accessibility_exception_handled(
        self, mock_accessibility_enabled, mock_clipboard
    ) -> None:
        """Should handle exceptions from accessibility API."""
        from vox.insert_darwin import TextInserter

        # Make accessibility raise exception
        mock_accessibility_enabled["copy"].side_effect = Exception("AX error")

        inserter = TextInserter()
        result = inserter.insert("test")

        # Should fall back to clipboard and succeed
        assert result is True
        mock_clipboard["pyperclip"].copy.assert_called()

    def test_insert_clipboard_exception_handled(
        self, mock_accessibility_disabled, mock_clipboard
    ) -> None:
        """Should handle exceptions from clipboard operations."""
        mock_quartz = MagicMock()
        mock_quartz.CGEventCreateKeyboardEvent.side_effect = Exception("Quartz failed")

        with patch.dict(sys.modules, {"Quartz": mock_quartz}):
            from vox.insert_darwin import TextInserter

            inserter = TextInserter()
            result = inserter.insert("test")

            assert result is False

    def test_insert_clipboard_empty_original(
        self, mock_accessibility_disabled, mock_clipboard
    ) -> None:
        """Should handle empty original clipboard."""
        from vox.insert_darwin import TextInserter

        mock_clipboard["pyperclip"].paste.side_effect = Exception("Empty clipboard")

        inserter = TextInserter()
        result = inserter.insert("new text")

        # Should still succeed
        assert result is True


@pytest.mark.unit
@pytest.mark.skipif(sys.platform != "linux", reason="Linux-specific tests")
class TestTextInserterLinux:
    """Tests for TextInserter class on Linux."""

    @pytest.fixture
    def mock_x11_environment(self):
        """Mock X11 environment."""
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}):
            with patch("shutil.which") as mock_which:
                mock_which.side_effect = lambda cmd: f"/usr/bin/{cmd}" if cmd in ["xdotool", "xclip"] else None
                yield mock_which

    @pytest.fixture
    def mock_wayland_environment(self):
        """Mock Wayland environment."""
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "wayland"}):
            with patch("shutil.which") as mock_which:
                mock_which.side_effect = lambda cmd: f"/usr/bin/{cmd}" if cmd in ["wtype", "wl-copy", "wl-paste"] else None
                yield mock_which

    @pytest.fixture
    def mock_no_tools_environment(self):
        """Mock environment with no tools available."""
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}):
            with patch("shutil.which") as mock_which:
                mock_which.return_value = None
                yield mock_which

    def test_init_x11_with_xdotool(self, mock_x11_environment) -> None:
        """Should use xdotool on X11 when available."""
        from vox.insert_linux import TextInserter

        inserter = TextInserter()
        assert inserter._primary_method == "xdotool"
        assert not inserter._is_wayland

    def test_init_wayland_with_wtype(self, mock_wayland_environment) -> None:
        """Should use wtype on Wayland when available."""
        from vox.insert_linux import TextInserter

        inserter = TextInserter()
        assert inserter._primary_method == "wtype"
        assert inserter._is_wayland

    def test_init_falls_back_to_clipboard(self, mock_no_tools_environment) -> None:
        """Should fall back to clipboard when no tools available."""
        from vox.insert_linux import TextInserter

        inserter = TextInserter()
        assert inserter._primary_method == "clipboard"

    def test_insert_empty_string(self, mock_x11_environment) -> None:
        """Should handle empty string insertion."""
        from vox.insert_linux import TextInserter

        inserter = TextInserter()
        result = inserter.insert("")
        assert result is True

    def test_insert_xdotool_success(self, mock_x11_environment) -> None:
        """Should insert text via xdotool."""
        from vox.insert_linux import TextInserter

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr=b"")
            inserter = TextInserter()
            result = inserter.insert("hello world")

        assert result is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "xdotool"
        assert "type" in call_args
        assert "hello world" in call_args

    def test_insert_wtype_success(self, mock_wayland_environment) -> None:
        """Should insert text via wtype."""
        from vox.insert_linux import TextInserter

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr=b"")
            inserter = TextInserter()
            result = inserter.insert("hello world")

        assert result is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "wtype"
        assert "hello world" in call_args

    def test_insert_xdotool_failure_falls_back(self, mock_x11_environment) -> None:
        """Should fall back to clipboard when xdotool fails."""
        from vox.insert_linux import TextInserter

        with patch("subprocess.run") as mock_run:
            # First call (xdotool) fails, second call (xdotool key for paste) succeeds
            mock_run.side_effect = [
                MagicMock(returncode=1, stderr=b"error"),
                MagicMock(returncode=0, stderr=b""),
            ]
            with patch("vox.insert_linux.pyperclip") as mock_clip:
                mock_clip.paste.return_value = "original"
                inserter = TextInserter()
                result = inserter.insert("test text")

        assert result is True
        mock_clip.copy.assert_called()

    def test_insert_clipboard_preserves_original(self, mock_no_tools_environment) -> None:
        """Should preserve and restore original clipboard contents."""
        from vox.insert_linux import TextInserter

        with patch("vox.insert_linux.pyperclip") as mock_clip:
            mock_clip.paste.return_value = "original clipboard"
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr=b"")
                with patch("vox.insert_linux.time"):
                    inserter = TextInserter()
                    # Force clipboard method by making primary method fail
                    inserter._primary_method = "clipboard"
                    result = inserter.insert("new text")

        # Should have read original clipboard
        mock_clip.paste.assert_called()
        # Check copy calls
        calls = mock_clip.copy.call_args_list
        assert len(calls) >= 2
        assert calls[0][0][0] == "new text"  # First: set new text
        assert calls[-1][0][0] == "original clipboard"  # Last: restore

    def test_check_dependencies(self, mock_x11_environment) -> None:
        """Should return dict of available tools."""
        from vox.insert_linux import TextInserter

        deps = TextInserter.check_dependencies()
        assert isinstance(deps, dict)
        assert "xdotool" in deps
        assert "wtype" in deps
        assert deps["xdotool"] is True
        assert deps["wtype"] is False
