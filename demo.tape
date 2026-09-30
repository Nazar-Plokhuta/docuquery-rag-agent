Output docs/assets/demo.gif

Set Shell "bash"
Set Theme "Catppuccin Mocha"
Set FontSize 14
Set Width 1100
Set Height 700
Set Padding 20
Set TypingSpeed 85ms

Hide
Type@0 "source .venv/bin/activate"
Enter
Type@0 "export PS1='$ '"
Enter
Type@0 "clear"
Enter
Sleep 500ms
Show

Type "python scripts/demo_runner.py"
Sleep 500ms
Enter
# Live ingest and query latency sits on top of the script's paced pauses.
# 30s covers a ~21s run with enough hold for the refusal panel to finish.
Sleep 30s
