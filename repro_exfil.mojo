from terminal import TerminalReader


comptime YOU_PROMPT = "\n\x1b[32myou> \x1b[0m"
comptime MODEL_PROMPT = "\x1b[38;5;211mmodel> \x1b[0m"


def main():
    print("paste then Enter — the model parrots exactly what it received. /quit to exit")
    var console = TerminalReader()
    while True:
        var line_opt = console.read_message(YOU_PROMPT)
        if not line_opt:
            print()
            break
        var s = String(line_opt.value().strip())
        if s == "/quit":
            break
        print(MODEL_PROMPT + s)
    print("bye")
