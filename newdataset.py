
from pathlib import Path
import json
import io

SRC  = Path(r"C:\Users\Patrick Myers\Downloads\c4_sample.jsonl")
DEST = Path("new_output.txt")

# Larger read buffer + fast json parse = faster overall
with SRC.open("rb", buffering=2**20) as bin_in, \
     io.TextIOWrapper(bin_in, encoding="utf-8", errors="replace") as jsonl_in, \
     DEST.open("w", encoding="utf-8", newline="\n") as txt_out:

    for ln, raw in enumerate(jsonl_in, 1):
        try:
            payload = json.loads(raw)
            text    = payload.get("text")
            if text:
                # Normalise newlines so every example is exactly one line
                txt_out.write(text.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n")
        except json.JSONDecodeError as e:
            # Wonky lines are common in huge web crawls—skip & log
            print(f"[Line {ln:,}] bad JSON → {e.msg!r}")

        # Flush every ~1 MB for safety (optional)
        if txt_out.tell() & ((1 << 20) - 1) == 0:
            txt_out.flush()

print("✅  Extraction complete →", DEST)
