#!/usr/bin/env python
"""Point-and-click labeller for the gold sheet (data/gold/gold_labels_v1.csv).
Shows one image at a time; tick the defect classes you can see (or none)
and press Save & next. Every save is written to the sheet immediately, so
closing the window never loses work. Starts at the first unlabelled row.

Usage:
    python scripts/label_gold.py --labeller RT

Keys: 1-4 toggle classes, Enter = save & next, Left/Right = previous/next.
"""

from __future__ import annotations

import argparse
import csv
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

CLASSES = ["crack", "vegetation", "corrosion", "dampness"]
FIELDS = ["order", "relpath", "content_hash", "defect_positive", "classes", "labeller", "notes"]
MAX_SIDE = 640


class Labeller:
    def __init__(self, root: tk.Tk, sheet: Path, image_root: Path, labeller: str) -> None:
        self.root, self.sheet, self.image_root, self.labeller = root, sheet, image_root, labeller
        with sheet.open(newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.i = next((k for k, r in enumerate(self.rows) if not r["defect_positive"]), 0)

        self.title = tk.Label(root, font=("Segoe UI", 11))
        self.title.pack(pady=4)
        self.canvas = tk.Label(root)
        self.canvas.pack()

        boxes = tk.Frame(root)
        boxes.pack(pady=6)
        self.vars = {c: tk.BooleanVar() for c in CLASSES}
        for n, c in enumerate(CLASSES, start=1):
            box = tk.Checkbutton(boxes, text=f"{n}. {c}", variable=self.vars[c])
            box.pack(side=tk.LEFT, padx=8)

        notes = tk.Frame(root)
        notes.pack()
        tk.Label(notes, text="notes (optional):").pack(side=tk.LEFT)
        self.notes = tk.Entry(notes, width=60)
        self.notes.pack(side=tk.LEFT)

        btns = tk.Frame(root)
        btns.pack(pady=8)
        tk.Button(btns, text="< Back", command=lambda: self.go(-1)).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Save & next  (Enter)", command=self.save, bg="#cfe8cf").pack(
            side=tk.LEFT, padx=6
        )
        tk.Button(btns, text="Skip >", command=lambda: self.go(1)).pack(side=tk.LEFT, padx=6)
        self.progress = tk.Label(root)
        self.progress.pack(pady=4)

        root.bind("<Return>", lambda e: self.save())
        root.bind("<Left>", lambda e: self.go(-1))
        root.bind("<Right>", lambda e: self.go(1))
        for n, c in enumerate(CLASSES, start=1):
            root.bind(str(n), lambda e, c=c: self.vars[c].set(not self.vars[c].get()))
        self.show()

    def show(self) -> None:
        row = self.rows[self.i]
        img = Image.open(self.image_root / row["relpath"])
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self.photo)
        self.title.configure(text=f"Row {row['order']} / {len(self.rows)}   {row['relpath']}")
        chosen = {c for c in row["classes"].split(";") if c}
        for c in CLASSES:
            self.vars[c].set(c in chosen)
        self.notes.delete(0, tk.END)
        self.notes.insert(0, row["notes"])
        done = sum(1 for r in self.rows if r["defect_positive"])
        state = f"already labelled: {row['defect_positive']}" if row["defect_positive"] else "new"
        self.progress.configure(text=f"{done} labelled   |   this row: {state}")

    def save(self) -> None:
        chosen = [c for c in CLASSES if self.vars[c].get()]
        row = self.rows[self.i]
        row["defect_positive"] = "yes" if chosen else "no"
        row["classes"] = ";".join(chosen)
        row["labeller"] = self.labeller
        row["notes"] = self.notes.get().strip()
        tmp = self.sheet.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        tmp.replace(self.sheet)
        self.go(1)

    def go(self, step: int) -> None:
        self.i = max(0, min(len(self.rows) - 1, self.i + step))
        self.show()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeller", required=True, help="your initials")
    parser.add_argument("--sheet", type=Path, default=Path("data/gold/gold_labels_v1.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/cbri"))
    args = parser.parse_args()

    root = tk.Tk()
    root.title("Certus gold labelling")
    Labeller(root, args.sheet, args.image_root, args.labeller)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
