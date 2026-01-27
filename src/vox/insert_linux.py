"""Text insertion for Linux using xdotool (X11) / wtype (Wayland) with clipboard fallback."""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import pyperclip

from .config import get_logger

log = get_logger("insert")


def _is_wayland() -> bool:
    """Check if running under Wayland.

    Returns:
        True if running under Wayland, False for X11.
    """
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session_type == "wayland":
        return True
    if session_type == "x11":
        return False
    # Fallback: check for WAYLAND_DISPLAY
    return bool(os.environ.get("WAYLAND_DISPLAY"))


def _has_command(cmd: str) -> bool:
    """Check if a command is available.

    Args:
        cmd: Command name to check.

    Returns:
        True if command exists in PATH.
    """
    return shutil.which(cmd) is not None


class TextInserter:
    """Inserts text at the current cursor position on Linux.

    Uses xdotool (X11) or wtype (Wayland) for direct text input,
    with clipboard paste as a fallback.
    """

    def __init__(self) -> None:
        """Initialize the text inserter."""
        log.info("Initializing TextInserter (Linux)")

        self._is_wayland = _is_wayland()
        log.info("Display server: %s", "Wayland" if self._is_wayland else "X11")

        # Check available tools
        self._has_xdotool = _has_command("xdotool")
        self._has_wtype = _has_command("wtype")
        self._has_xclip = _has_command("xclip")
        self._has_xdotool_key = self._has_xdotool  # xdotool can send keys too
        self._has_wl_copy = _has_command("wl-copy")
        self._has_wl_paste = _has_command("wl-paste")
        self._has_ydotool = _has_command("ydotool")

        log.info(
            "Available tools: xdotool=%s, wtype=%s, xclip=%s, wl-copy=%s, ydotool=%s",
            self._has_xdotool,
            self._has_wtype,
            self._has_xclip,
            self._has_wl_copy,
            self._has_ydotool,
        )

        # Determine best insertion method
        if self._is_wayland:
            if self._has_wtype:
                self._primary_method = "wtype"
            elif self._has_ydotool:
                self._primary_method = "ydotool"
            else:
                self._primary_method = "clipboard"
                log.warning(
                    "No Wayland text input tool found (wtype or ydotool). "
                    "Install wtype for best results: sudo apt install wtype"
                )
        else:
            if self._has_xdotool:
                self._primary_method = "xdotool"
            else:
                self._primary_method = "clipboard"
                log.warning(
                    "xdotool not found. Install for best results: sudo apt install xdotool"
                )

        log.info("Primary insertion method: %s", self._primary_method)

    def insert(self, text: str) -> bool:
        """Insert text at cursor. Returns True if successful.

        Args:
            text: The text to insert at the current cursor position.

        Returns:
            True if insertion succeeded, False otherwise.
        """
        if not text:
            log.debug("Empty text, nothing to insert")
            return True

        log.info(
            "Inserting text: '%s' (%d chars)",
            text[:50] + "..." if len(text) > 50 else text,
            len(text),
        )

        # Try primary method first
        if self._primary_method == "xdotool":
            success = self._insert_xdotool(text)
            if success:
                return True
            log.warning("xdotool failed, falling back to clipboard")

        elif self._primary_method == "wtype":
            success = self._insert_wtype(text)
            if success:
                return True
            log.warning("wtype failed, falling back to clipboard")

        elif self._primary_method == "ydotool":
            success = self._insert_ydotool(text)
            if success:
                return True
            log.warning("ydotool failed, falling back to clipboard")

        # Fallback to clipboard
        return self._insert_clipboard(text)

    def _insert_xdotool(self, text: str) -> bool:
        """Insert text using xdotool (X11).

        Args:
            text: The text to insert.

        Returns:
            True if successful, False otherwise.
        """
        try:
            log.debug("Using xdotool to type text...")
            # --clearmodifiers releases any held modifiers (like Alt) before typing
            result = subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--", text],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                log.info("xdotool insert succeeded")
                return True
            else:
                log.warning(
                    "xdotool failed (returncode=%d): %s",
                    result.returncode,
                    result.stderr.decode(),
                )
                return False
        except subprocess.TimeoutExpired:
            log.error("xdotool timed out")
            return False
        except Exception as e:
            log.error("xdotool insert failed: %s", e, exc_info=True)
            return False

    def _insert_wtype(self, text: str) -> bool:
        """Insert text using wtype (Wayland).

        Args:
            text: The text to insert.

        Returns:
            True if successful, False otherwise.
        """
        try:
            log.debug("Using wtype to type text...")
            result = subprocess.run(
                ["wtype", "--", text],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                log.info("wtype insert succeeded")
                return True
            else:
                log.warning(
                    "wtype failed (returncode=%d): %s",
                    result.returncode,
                    result.stderr.decode(),
                )
                return False
        except subprocess.TimeoutExpired:
            log.error("wtype timed out")
            return False
        except Exception as e:
            log.error("wtype insert failed: %s", e, exc_info=True)
            return False

    def _insert_ydotool(self, text: str) -> bool:
        """Insert text using ydotool (Wayland, requires ydotoold).

        Args:
            text: The text to insert.

        Returns:
            True if successful, False otherwise.
        """
        try:
            log.debug("Using ydotool to type text...")
            result = subprocess.run(
                ["ydotool", "type", "--", text],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                log.info("ydotool insert succeeded")
                return True
            else:
                log.warning(
                    "ydotool failed (returncode=%d): %s",
                    result.returncode,
                    result.stderr.decode(),
                )
                return False
        except subprocess.TimeoutExpired:
            log.error("ydotool timed out")
            return False
        except Exception as e:
            log.error("ydotool insert failed: %s", e, exc_info=True)
            return False

    def _insert_clipboard(self, text: str) -> bool:
        """Insert text using clipboard and paste.

        This is the fallback method. It preserves the original clipboard
        contents and restores them after.

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

            # Set our text
            log.debug("Setting clipboard to new text...")
            pyperclip.copy(text)

            # Send Ctrl+V to paste
            success = self._send_paste_shortcut()
            if not success:
                log.warning("Failed to send paste shortcut")
                # Still try to restore clipboard
                try:
                    pyperclip.copy(old_clipboard)
                except Exception:
                    pass
                return False

            # Wait for paste to complete
            time.sleep(0.25)
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

    def _send_paste_shortcut(self) -> bool:
        """Send Ctrl+V keystroke to paste from clipboard.

        Returns:
            True if successful, False otherwise.
        """
        if self._is_wayland:
            # On Wayland, try ydotool or wtype for key simulation
            if self._has_ydotool:
                try:
                    # ydotool key syntax: 29 = left ctrl, 47 = v
                    result = subprocess.run(
                        ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"],
                        capture_output=True,
                        timeout=5,
                    )
                    return result.returncode == 0
                except Exception as e:
                    log.warning("ydotool key failed: %s", e)

            if self._has_wtype:
                try:
                    # wtype -M ctrl -P v -p v -m ctrl sends Ctrl+V
                    result = subprocess.run(
                        ["wtype", "-M", "ctrl", "-P", "v", "-p", "v", "-m", "ctrl"],
                        capture_output=True,
                        timeout=5,
                    )
                    return result.returncode == 0
                except Exception as e:
                    log.warning("wtype key failed: %s", e)

            log.warning("No Wayland tool available for Ctrl+V")
            return False
        else:
            # X11: use xdotool
            if self._has_xdotool:
                try:
                    result = subprocess.run(
                        ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
                        capture_output=True,
                        timeout=5,
                    )
                    return result.returncode == 0
                except Exception as e:
                    log.warning("xdotool key failed: %s", e)
                    return False

            log.warning("xdotool not available for Ctrl+V")
            return False

    @staticmethod
    def check_dependencies() -> dict[str, bool]:
        """Check which text insertion dependencies are available.

        Returns:
            Dict mapping tool names to availability.
        """
        return {
            "xdotool": _has_command("xdotool"),
            "wtype": _has_command("wtype"),
            "ydotool": _has_command("ydotool"),
            "xclip": _has_command("xclip"),
            "wl-copy": _has_command("wl-copy"),
            "wl-paste": _has_command("wl-paste"),
        }
