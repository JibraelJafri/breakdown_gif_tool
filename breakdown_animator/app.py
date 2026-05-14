"""breakdown_animator.app
======================
Unified entrypoint for animforge (breakdown-animator).
Routes execution between interactive Textual TUI and headless CLI mode.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from breakdown_animator.cli import run_cli
from breakdown_animator.tui import AnimForgeApp
from breakdown_animator.wizard import run_wizard


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Main entrypoint router for animforge.
    
    - No arguments: Launches fast interactive terminal wizard.
    - Single folder argument without flags: Launches interactive wizard pre-loaded with folder.
    - '--tui': Launches full-screen Textual dark-mode TUI.
    - CLI arguments provided: Runs headless batch processor.
    
    Returns:
        Process exit code (0 on success, non-zero on error).
    """
    if argv is None:
        argv = sys.argv[1:]

    # 1. Check for explicit TUI flag
    if "--tui" in argv or "-t" in argv:
        remaining = [arg for arg in argv if arg not in ("--tui", "-t")]
        init_path = remaining[0] if remaining else None
        app = AnimForgeApp(initial_path=init_path)
        app.run()
        return 0

    # 2. Check for explicit wizard flag or no arguments
    if len(argv) == 0 or "--wizard" in argv or "-w" in argv:
        remaining = [arg for arg in argv if arg not in ("--wizard", "-w")]
        init_path = remaining[0] if remaining else None
        return run_wizard(initial_path=init_path)

    # 3. If single positional argument with no options and stdin is interactive TTY, run wizard with folder
    has_flags = any(arg.startswith("-") for arg in argv)
    if not has_flags and len(argv) == 1 and sys.stdin.isatty():
        return run_wizard(initial_path=argv[0])

    # 4. Otherwise run headless CLI
    return run_cli(argv)


if __name__ == "__main__":
    sys.exit(main())
