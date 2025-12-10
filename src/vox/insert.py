"""Text insertion using macOS Accessibility API with clipboard fallback."""

from __future__ import annotations

import subprocess
import time

import pyperclip
from ApplicationServices import (
    AXIsProcessTrusted,
    AXUIElementCopyAttributeValue,
    AXUIElementCreateSystemWide,
    AXUIElementSetAttributeValue,
)
from CoreFoundation import kCFRunLoopDefaultMode

# For prompting accessibility permission
try:
    from ApplicationServices import AXIsProcessTrustedWithOptions
    from CoreFoundation import CFDictionaryCreate, kCFTypeDictionaryKeyCallBacks, kCFTypeDictionaryValueCallBacks, kCFBooleanTrue

    # kAXTrustedCheckOptionPrompt key
    kAXTrustedCheckOptionPrompt = "AXTrustedCheckOptionPrompt"
    HAS_TRUSTED_OPTIONS = True
except ImportError:
    HAS_TRUSTED_OPTIONS = False


# Accessibility attribute constants
kAXFocusedUIElementAttribute = "AXFocusedUIElement"
kAXSelectedTextAttribute = "AXSelectedText"
kAXValueAttribute = "AXValue"


class TextInserter:
    """Inserts text at the current cursor position.

    Uses macOS Accessibility API as the primary method, with clipboard
    paste as a fallback if accessibility is not available.
    """

    def __init__(self, prompt_for_access: bool = True) -> None:
        """Initialize the text inserter.

        Args:
            prompt_for_access: If True, prompt user to grant accessibility
                permissions if not already granted.
        """
        if prompt_for_access:
            self.use_accessibility = self._check_accessibility_with_prompt()
        else:
            self.use_accessibility = AXIsProcessTrusted()

        if not self.use_accessibility:
            print("Warning: Accessibility not enabled, using clipboard fallback")

    def _check_accessibility_with_prompt(self) -> bool:
        """Check accessibility and prompt user if not enabled.

        Returns:
            True if accessibility is enabled.
        """
        if HAS_TRUSTED_OPTIONS:
            try:
                # Create options dict with prompt=True
                options = {kAXTrustedCheckOptionPrompt: kCFBooleanTrue}
                return AXIsProcessTrustedWithOptions(options)
            except Exception:
                pass

        # Fallback to simple check
        return AXIsProcessTrusted()

    def insert(self, text: str) -> bool:
        """Insert text at cursor. Returns True if successful.

        Args:
            text: The text to insert at the current cursor position.

        Returns:
            True if insertion succeeded, False otherwise.
        """
        if not text:
            return True  # Nothing to insert

        if self.use_accessibility:
            success = self._insert_accessibility(text)
            if success:
                return True

        # Fallback to clipboard
        return self._insert_clipboard(text)

    def _insert_accessibility(self, text: str) -> bool:
        """Insert text using Accessibility API.

        This replaces the currently selected text (if any) with the new text.
        If no text is selected, it inserts at the cursor position.

        Args:
            text: The text to insert.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get the system-wide accessibility element
            system_wide = AXUIElementCreateSystemWide()

            # Get the currently focused element
            err, focused = AXUIElementCopyAttributeValue(
                system_wide,
                kAXFocusedUIElementAttribute,
                None,
            )

            if err != 0 or focused is None:
                return False

            # Set the selected text attribute to insert our text
            # This replaces selected text or inserts at cursor if nothing selected
            err = AXUIElementSetAttributeValue(
                focused,
                kAXSelectedTextAttribute,
                text,
            )

            return err == 0

        except Exception as e:
            print(f"Accessibility insert failed: {e}")
            return False

    def _insert_clipboard(self, text: str) -> bool:
        """Insert text using clipboard and paste.

        This is the fallback method when accessibility is not available.
        It preserves the original clipboard contents and restores them after.

        Args:
            text: The text to insert.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Save current clipboard
            old_clipboard = ""
            try:
                old_clipboard = pyperclip.paste()
            except Exception:
                pass  # Clipboard may be empty or contain non-text

            # Set our text and paste
            pyperclip.copy(text)

            # Use AppleScript to send Cmd+V
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to keystroke "v" using command down',
                ],
                check=True,
                capture_output=True,
            )

            # Wait a bit for paste to complete
            time.sleep(0.1)

            # Restore original clipboard
            try:
                pyperclip.copy(old_clipboard)
            except Exception:
                pass

            return True

        except Exception as e:
            print(f"Clipboard insert failed: {e}")
            return False

    @staticmethod
    def check_accessibility() -> bool:
        """Check if accessibility permissions are enabled.

        Returns:
            True if the app has accessibility permissions.
        """
        return AXIsProcessTrusted()

    def refresh_accessibility_status(self) -> None:
        """Refresh the accessibility permission status.

        Call this after the user enables accessibility permissions
        to update the internal state.
        """
        self.use_accessibility = AXIsProcessTrusted()
