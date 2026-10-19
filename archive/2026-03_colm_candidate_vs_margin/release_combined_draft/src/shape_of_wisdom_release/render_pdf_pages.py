from __future__ import annotations

from .build_assets import inspect_latex_log, render_pages, scan_text, write_render_report


def main() -> None:
    text_failures = scan_text()
    log_failures = inspect_latex_log()
    render_pages()
    write_render_report(text_failures, log_failures)
    if text_failures or log_failures:
        failures = text_failures + log_failures
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
