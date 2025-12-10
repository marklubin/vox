"""Unit tests for text insertion module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestTextInserter:
    """Tests for TextInserter class."""

    @pytest.fixture
    def mock_accessibility_enabled(self):
        """Mock accessibility APIs when accessibility is enabled."""
        with patch("vox.insert.AXIsProcessTrusted") as mock_trusted:
            with patch("vox.insert.HAS_TRUSTED_OPTIONS", False):
                mock_trusted.return_value = True
                with patch("vox.insert.AXUIElementCreateSystemWide") as mock_system:
                    with patch("vox.insert.AXUIElementCopyAttributeValue") as mock_copy:
                        with patch("vox.insert.AXUIElementSetAttributeValue") as mock_set:
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
        with patch("vox.insert.AXIsProcessTrusted") as mock_trusted:
            with patch("vox.insert.HAS_TRUSTED_OPTIONS", False):
                mock_trusted.return_value = False
                yield mock_trusted

    @pytest.fixture
    def mock_clipboard(self):
        """Mock clipboard operations."""
        with patch("vox.insert.pyperclip") as mock_clip:
            mock_clip.paste.return_value = "original clipboard"
            with patch("vox.insert.subprocess") as mock_subprocess:
                mock_subprocess.run.return_value = MagicMock()
                with patch("vox.insert.time"):
                    yield {
                        "pyperclip": mock_clip,
                        "subprocess": mock_subprocess,
                    }

    def test_init_with_accessibility(self, mock_accessibility_enabled) -> None:
        """Should use accessibility when available."""
        from vox.insert import TextInserter

        inserter = TextInserter()
        assert inserter.use_accessibility is True

    def test_init_without_accessibility(self, mock_accessibility_disabled) -> None:
        """Should fall back to clipboard when accessibility unavailable."""
        from vox.insert import TextInserter

        inserter = TextInserter()
        assert inserter.use_accessibility is False

    def test_insert_empty_string(self, mock_accessibility_enabled) -> None:
        """Should handle empty string insertion."""
        from vox.insert import TextInserter

        inserter = TextInserter()
        result = inserter.insert("")

        assert result is True
        # Should not call any accessibility APIs
        mock_accessibility_enabled["system"].assert_not_called()

    def test_insert_accessibility_success(self, mock_accessibility_enabled) -> None:
        """Should insert text via accessibility API."""
        from vox.insert import TextInserter

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
        from vox.insert import TextInserter

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
        from vox.insert import TextInserter

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
        """Should send Cmd+V via AppleScript."""
        from vox.insert import TextInserter

        inserter = TextInserter()
        inserter.insert("test")

        mock_clipboard["subprocess"].run.assert_called_once()
        call_args = mock_clipboard["subprocess"].run.call_args[0][0]
        assert "osascript" in call_args
        assert 'keystroke "v" using command down' in call_args[-1]

    def test_check_accessibility_static(self, mock_accessibility_enabled) -> None:
        """Static method should check accessibility status."""
        from vox.insert import TextInserter

        result = TextInserter.check_accessibility()
        assert result is True

    def test_refresh_accessibility_status(self) -> None:
        """Should update accessibility status on refresh."""
        with patch("vox.insert.AXIsProcessTrusted") as mock_trusted:
            with patch("vox.insert.HAS_TRUSTED_OPTIONS", False):
                mock_trusted.return_value = False
                from vox.insert import TextInserter

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
        from vox.insert import TextInserter

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
        from vox.insert import TextInserter

        mock_clipboard["subprocess"].run.side_effect = Exception("Paste failed")

        inserter = TextInserter()
        result = inserter.insert("test")

        assert result is False

    def test_insert_clipboard_empty_original(
        self, mock_accessibility_disabled, mock_clipboard
    ) -> None:
        """Should handle empty original clipboard."""
        from vox.insert import TextInserter

        mock_clipboard["pyperclip"].paste.side_effect = Exception("Empty clipboard")

        inserter = TextInserter()
        result = inserter.insert("new text")

        # Should still succeed
        assert result is True
