# Systemd User Service for Vox

## Setup

1. Create your environment file with your Deepgram API key:
   ```bash
   echo "DEEPGRAM_API_KEY=your_key_here" > ~/.config/vox.env
   chmod 600 ~/.config/vox.env
   ```

2. Install the service:
   ```bash
   mkdir -p ~/.config/systemd/user
   cp systemd/vox.service ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable vox
   systemctl --user start vox
   ```

## Commands

- Start: `systemctl --user start vox`
- Stop: `systemctl --user stop vox`
- Restart: `systemctl --user restart vox`
- Status: `systemctl --user status vox`
- Logs: `journalctl --user -u vox -f`

## Notes

- The service starts automatically on login
- Logs are also written to `~/.local/share/vox/vox.log`
- Use "Open Logs" from the tray menu to view logs
