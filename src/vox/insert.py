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

from .config import get_logger

log = get_logger("insert")

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
        log.info("Initializing TextInserter (prompt_for_access=%s)", prompt_for_access)
        if prompt_for_access:
            self.use_accessibility = self._check_accessibility_with_prompt()
        else:
            self.use_accessibility = AXIsProcessTrusted()

        if not self.use_accessibility:
            log.warning("Accessibility not enabled, using clipboard fallback")
        else:
            log.info("Accessibility enabled")

    def _check_accessibility_with_prompt(self) -> bool:
        """Check accessibility and prompt user if not enabled.

        Returns:
            True if accessibility is enabled.
        """
        log.debug("Checking accessibility with prompt...")
        if HAS_TRUSTED_OPTIONS:
            try:
                # Create options dict with prompt=True
                options = {kAXTrustedCheckOptionPrompt: kCFBooleanTrue}
                result = AXIsProcessTrustedWithOptions(options)
                log.debug("AXIsProcessTrustedWithOptions returned: %s", result)
                return result
            except Exception as e:
                log.warning("Error checking accessibility with options: %s", e)

        # Fallback to simple check
        result = AXIsProcessTrusted()
        log.debug("AXIsProcessTrusted returned: %s", result)
        return result

    def insert(self, text: str) -> bool:
        """Insert text at cursor. Returns True if successful.

        Args:
            text: The text to insert at the current cursor position.

        Returns:
            True if insertion succeeded, False otherwise.
        """
        if not text:
            log.debug("Empty text, nothing to insert")
            return True  # Nothing to insert

        log.info("Inserting text: '%s' (%d chars)", text[:50] + "..." if len(text) > 50 else text, len(text))

        # Try accessibility first - it works well for native apps and doesn't mess with clipboard
        if self.use_accessibility:
            log.debug("Trying accessibility API...")
            success = self._insert_accessibility(text)
            if success:
                log.info("Accessibility insert succeeded")
                return True
            log.warning("Accessibility insert failed, falling back to clipboard")

        # Fallback to clipboard
        log.debug("Using clipboard fallback...")
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
            log.debug("Getting system-wide accessibility element...")
            system_wide = AXUIElementCreateSystemWide()

            # Get the currently focused element
            log.debug("Getting focused element...")
            err, focused = AXUIElementCopyAttributeValue(
                system_wide,
                kAXFocusedUIElementAttribute,
                None,
            )

            if err != 0 or focused is None:
                log.warning("Failed to get focused element (err=%d, focused=%s)", err, focused)
                return False

            # Log details about the focused element
            log.debug("Got focused element: %s", focused)
            try:
                # Try to get role of focused element for debugging
                from ApplicationServices import kAXRoleAttribute
                role_err, role = AXUIElementCopyAttributeValue(focused, "AXRole", None)
                title_err, title = AXUIElementCopyAttributeValue(focused, "AXTitle", None)
                log.debug("Focused element role=%s (err=%d), title=%s (err=%d)", role, role_err, title, title_err)
            except Exception as e:
                log.debug("Could not get focused element details: %s", e)

            # Set the selected text attribute to insert our text
            # This replaces selected text or inserts at cursor if nothing selected
            log.debug("Setting selected text attribute...")
            err = AXUIElementSetAttributeValue(
                focused,
                kAXSelectedTextAttribute,
                text,
            )

            if err == 0:
                # Verify the text was actually inserted by reading it back
                # Some apps (Electron-based like Claude, Slack) return success but don't actually insert
                time.sleep(0.05)  # Small delay to let the UI update
                verify_err, current_text = AXUIElementCopyAttributeValue(
                    focused, kAXValueAttribute, None
                )
                if verify_err == 0 and current_text and text in str(current_text):
                    log.info("Successfully inserted and verified text")
                    return True
                else:
                    log.warning("Set attribute succeeded but text not found in element - app may not support AX insertion")
                    return False
            else:
                # Common error codes:
                # -25204 = kAXErrorAttributeUnsupported (element doesn't support this attribute)
                # -25205 = kAXErrorActionUnsupported
                # -25212 = kAXErrorCannotComplete
                log.warning("Failed to set selected text attribute (err=%d) - element may not support text insertion", err)
                return False

        except Exception as e:
            log.error("Accessibility insert failed: %s", e, exc_info=True)
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
                log.debug("Saved original clipboard (%d chars)", len(old_clipboard))
            except Exception as e:
                log.debug("Could not read clipboard (may be empty/non-text): %s", e)

            # Set our text and paste
            log.debug("Setting clipboard to new text...")
            pyperclip.copy(text)

            # Use Quartz to send Cmd+V - more reliable than AppleScript across all apps
            log.debug("Sending Cmd+V via Quartz CGEvent...")
            import Quartz

            # Key code for 'v' is 9 on macOS
            v_keycode = 9

            # Create key down event with Command modifier
            event_down = Quartz.CGEventCreateKeyboardEvent(None, v_keycode, True)
            Quartz.CGEventSetFlags(event_down, Quartz.kCGEventFlagMaskCommand)

            # Create key up event with Command modifier
            event_up = Quartz.CGEventCreateKeyboardEvent(None, v_keycode, False)
            Quartz.CGEventSetFlags(event_up, Quartz.kCGEventFlagMaskCommand)

            # Post the events
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_down)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)

            # Wait a bit for paste to complete
            time.sleep(0.1)
            log.debug("Paste command sent")

            # Restore original clipboard
            try:
                pyperclip.copy(old_clipboard)
                log.debug("Restored original clipboard")
            except Exception as e:
                log.warning("Could not restore clipboard: %s", e)

            log.info("Clipboard insert succeeded")
            return True

        except Exception as e:
            log.error("Clipboard insert failed: %s", e, exc_info=True)
            return False

    @staticmethod
    def check_accessibility() -> bool:
        """Check if accessibility permissions are enabled.

        Returns:
            True if the app has accessibility permissions.
        """
        result = AXIsProcessTrusted()
        log.debug("check_accessibility() returned: %s", result)
        return result

    def refresh_accessibility_status(self) -> None:
        """Refresh the accessibility permission status.

        Call this after the user enables accessibility permissions
        to update the internal state.
        """
        old_status = self.use_accessibility
        self.use_accessibility = AXIsProcessTrusted()
        log.info(
            "Refreshed accessibility status: %s -> %s",
            old_status,
            self.use_accessibility,
        )
