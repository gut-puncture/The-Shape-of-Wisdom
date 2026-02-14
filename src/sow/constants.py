from __future__ import annotations

BASELINE_WRAPPER_ID = "plain_exam"

EXPECTED_ROBUSTNESS_WRAPPER_IDS_V2 = [
    "academic_abstract",
    "ascii_box",
    "changelog_entry",
    "csv_inline",
    "graphql_query",
    "haiku_riddle",
    "html_form",
    "ini_file",
    "irc_log",
    "key_equals",
    "legal_clause",
    "meeting_minutes",
    "protobuf_msg",
    "quest_briefing",
    "recipe_instruction",
    "regex_match",
    "s_expression",
    "shell_heredoc",
    "toml_config",
    "tweet_thread",
]

ANSWER_SUFFIX = "Answer: "

ROBUSTNESS_SUFFIX = "\n\nReturn only the letter (A, B, C, or D).\n" + ANSWER_SUFFIX

