"""
Q-Table Interactive Editor
==========================
Run:  python qtable_editor.py [path_to_qtable.pkl]

Controls
--------
  Arrow keys / WASD  – navigate cells
  Page Up / Page Dn  – jump 10 rows
  Home / End         – first / last row
  Tab                – cycle through state dimensions (if state is a tuple)
  E                  – edit selected cell value
  S                  – save (overwrite original file)
  A                  – save As (new filename)
  F                  – find / search (state key + action index)
  R                  – reload from disk (discard unsaved changes)
  I                  – info on selected cell (precision, dtype, raw repr)
  Q / Esc            – quit
"""

import sys
import os
import pickle
import curses
import copy
import struct
import math
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_qtable(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_qtable(qtable, path: str):
    with open(path, "wb") as f:
        pickle.dump(qtable, f)


def detect_precision(value: float) -> int:
    """Return how many decimal places the value appears to use."""
    if not isinstance(value, float):
        return 0
    s = repr(value)
    if "." in s:
        dec = s.split(".")[1]
        # strip trailing zeros that repr adds for floats like 0.10000000000001
        return len(dec.rstrip("0")) or 1
    return 0


def detect_limits(all_values):
    """Return (min_val, max_val) from all Q-values in the table."""
    flat = []
    for v in all_values:
        if hasattr(v, "__iter__"):
            flat.extend(v)
        else:
            flat.append(v)
    valid = [x for x in flat if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not valid:
        return (None, None)
    return (min(valid), max(valid))


def qtable_to_rows(qtable):
    """
    Normalise various Q-table shapes into a list of (state_key, [q_values]).
    Supports:
      dict  { state: [q0, q1, ...] }
      dict  { state: {action: q} }
      list/array of lists
    Returns: rows, action_labels
    """
    if isinstance(qtable, dict):
        first_val = next(iter(qtable.values()))
        if isinstance(first_val, dict):
            # {state: {action: q}}
            all_actions = sorted({a for v in qtable.values() for a in v.keys()})
            rows = []
            for state, adict in qtable.items():
                qvals = [adict.get(a, 0.0) for a in all_actions]
                rows.append((state, qvals))
            return rows, [str(a) for a in all_actions]
        elif hasattr(first_val, "__iter__"):
            # {state: [q0, q1, ...]}
            rows = [(state, list(qvals)) for state, qvals in qtable.items()]
            n_actions = max(len(r[1]) for r in rows)
            return rows, [f"A{i}" for i in range(n_actions)]
        else:
            # {state: scalar} – single-action table
            rows = [(state, [val]) for state, val in qtable.items()]
            return rows, ["A0"]
    else:
        try:
            import numpy as np
            arr = np.array(qtable)
            if arr.ndim == 2:
                rows = [(i, list(arr[i])) for i in range(arr.shape[0])]
                return rows, [f"A{j}" for j in range(arr.shape[1])]
        except Exception:
            pass
        rows = [(i, list(v)) for i, v in enumerate(qtable)]
        n_actions = max(len(r[1]) for r in rows)
        return rows, [f"A{i}" for i in range(n_actions)]


def apply_edit(qtable, state_key, action_idx, new_value):
    """Write new_value back into the original qtable structure."""
    if isinstance(qtable, dict):
        val = qtable[state_key]
        if isinstance(val, dict):
            action_keys = sorted(val.keys())
            qtable[state_key][action_keys[action_idx]] = new_value
        else:
            qtable[state_key][action_idx] = new_value
    else:
        qtable[state_key][action_idx] = new_value


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

COL_WIDTH = 14
KEY_WIDTH = 36
STATUS_H  = 3


def draw_header(win, action_labels, col_offset, max_w):
    win.clear()
    key_part = f"{'STATE KEY':<{KEY_WIDTH}}"
    win.addstr(0, 0, key_part[:max_w], curses.color_pair(3) | curses.A_BOLD)
    x = KEY_WIDTH
    for i, label in enumerate(action_labels[col_offset:]):
        cell = f"{label:^{COL_WIDTH}}"
        if x + COL_WIDTH > max_w:
            break
        win.addstr(0, x, cell, curses.color_pair(3) | curses.A_BOLD)
        x += COL_WIDTH
    win.refresh()


def draw_table(win, rows, action_labels, row_offset, col_offset,
               cursor_row, cursor_col, max_h, max_w, modified_cells):
    win.clear()
    for screen_row in range(max_h):
        data_row = row_offset + screen_row
        if data_row >= len(rows):
            break
        state_key, qvals = rows[data_row]
        key_str = str(state_key)
        if len(key_str) > KEY_WIDTH - 2:
            key_str = key_str[:KEY_WIDTH - 5] + "..."

        row_attr = curses.color_pair(1)
        if data_row == cursor_row:
            row_attr = curses.color_pair(2) | curses.A_BOLD

        win.addstr(screen_row, 0, f"{key_str:<{KEY_WIDTH}}"[:max_w], row_attr)

        x = KEY_WIDTH
        for col_i, q in enumerate(qvals[col_offset:]):
            actual_col = col_offset + col_i
            if x + COL_WIDTH > max_w:
                break
            cell_val = f"{q:>{COL_WIDTH - 1}.6g} "
            is_cursor = (data_row == cursor_row and actual_col == cursor_col)
            is_modified = (state_key, actual_col) in modified_cells

            if is_cursor:
                attr = curses.color_pair(4) | curses.A_BOLD
            elif is_modified:
                attr = curses.color_pair(5) | curses.A_BOLD
            else:
                attr = row_attr
            win.addstr(screen_row, x, cell_val, attr)
            x += COL_WIDTH

    win.refresh()


def draw_status(win, msg, filepath, modified, max_w):
    win.clear()
    mod_flag = " [MODIFIED]" if modified else ""
    top = f" {Path(filepath).name}{mod_flag}  |  {msg}"
    win.addstr(0, 0, top[:max_w], curses.color_pair(6))
    help_line = " ←→↑↓:navigate  E:edit  S:save  A:saveAs  F:find  I:info  R:reload  Q:quit"
    win.addstr(1, 0, help_line[:max_w], curses.color_pair(1))
    win.refresh()


def prompt_string(stdscr, prompt: str, prefill: str = "") -> str:
    h, w = stdscr.getmaxyx()
    curses.echo()
    curses.curs_set(1)
    stdscr.addstr(h - 1, 0, (prompt + prefill)[:w - 1], curses.color_pair(6) | curses.A_BOLD)
    stdscr.clrtoeol()
    stdscr.refresh()
    result_bytes = stdscr.getstr(h - 1, len(prompt), w - len(prompt) - 2)
    curses.noecho()
    curses.curs_set(0)
    raw = result_bytes.decode("utf-8", errors="replace").strip()
    return raw if raw else prefill


def run_editor(stdscr, filepath: str):
    # -- Colour pairs --
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, -1, -1)                           # normal
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)   # selected row
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_WHITE)  # header
    curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_YELLOW) # cursor cell
    curses.init_pair(5, curses.COLOR_RED, -1)             # modified cell
    curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_GREEN)  # status bar

    curses.curs_set(0)
    stdscr.keypad(True)

    qtable = load_qtable(filepath)
    original_qtable = copy.deepcopy(qtable)

    rows, action_labels = qtable_to_rows(qtable)
    n_rows = len(rows)
    n_cols = len(action_labels)

    cursor_row = 0
    cursor_col = 0
    row_offset = 0
    col_offset = 0
    modified = False
    modified_cells: set = set()
    status_msg = f"Loaded {n_rows} states × {n_cols} actions"

    # All Q-values flat for limit detection
    all_qvals = [q for _, qvals in rows for q in qvals]
    q_min, q_max = detect_limits(all_qvals)

    while True:
        h, w = stdscr.getmaxyx()
        table_h = h - STATUS_H - 1   # rows for data
        header_h = 1

        # -- Layout windows --
        try:
            header_win = curses.newwin(header_h, w, 0, 0)
            table_win  = curses.newwin(table_h,  w, header_h, 0)
            status_win = curses.newwin(STATUS_H, w, h - STATUS_H, 0)
        except curses.error:
            stdscr.clear()
            stdscr.addstr(0, 0, "Terminal too small. Resize and press any key.")
            stdscr.refresh()
            stdscr.getch()
            continue

        # Keep offsets sane
        visible_cols = max(1, (w - KEY_WIDTH) // COL_WIDTH)
        if cursor_col < col_offset:
            col_offset = cursor_col
        if cursor_col >= col_offset + visible_cols:
            col_offset = cursor_col - visible_cols + 1

        if cursor_row < row_offset:
            row_offset = cursor_row
        if cursor_row >= row_offset + table_h:
            row_offset = cursor_row - table_h + 1

        draw_header(header_win, action_labels, col_offset, w)
        draw_table(table_win, rows, action_labels, row_offset, col_offset,
                   cursor_row, cursor_col, table_h, w, modified_cells)
        draw_status(status_win, status_msg, filepath, modified, w)

        key = stdscr.getch()

        # --- Navigation ---
        if key in (curses.KEY_UP, ord("w"), ord("W")):
            cursor_row = max(0, cursor_row - 1)
        elif key in (curses.KEY_DOWN, ord("s"), ord("S")):
            cursor_row = min(n_rows - 1, cursor_row + 1)
        elif key in (curses.KEY_LEFT, ord("a"), ord("A")):
            cursor_col = max(0, cursor_col - 1)
        elif key in (curses.KEY_RIGHT, ord("d"), ord("D")):
            cursor_col = min(n_cols - 1, cursor_col + 1)
        elif key == curses.KEY_PPAGE:  # Page Up
            cursor_row = max(0, cursor_row - 10)
        elif key == curses.KEY_NPAGE:  # Page Down
            cursor_row = min(n_rows - 1, cursor_row + 10)
        elif key == curses.KEY_HOME:
            cursor_row = 0
        elif key == curses.KEY_END:
            cursor_row = n_rows - 1

        # --- Edit ---
        elif key in (ord("e"), ord("E")):
            state_key, qvals = rows[cursor_row]
            old_val = qvals[cursor_col]
            prec = detect_precision(old_val)
            prompt = f"Edit [{state_key}][{action_labels[cursor_col]}] (was {old_val}): "
            new_str = prompt_string(stdscr, prompt, str(old_val))
            try:
                new_val = float(new_str)
                # Warn if outside observed range
                if q_min is not None and not (q_min <= new_val <= q_max):
                    status_msg = (f"Warning: {new_val} outside original range "
                                  f"[{q_min:.4g}, {q_max:.4g}]. Saved anyway.")
                else:
                    status_msg = (f"Updated [{state_key}][{action_labels[cursor_col]}]: "
                                  f"{old_val} → {new_val}")
                # Round to same precision as original value
                if prec > 0:
                    new_val = round(new_val, max(prec, 10))
                apply_edit(qtable, state_key, cursor_col, new_val)
                rows[cursor_row][1][cursor_col] = new_val
                modified = True
                modified_cells.add((state_key, cursor_col))
            except ValueError:
                status_msg = f"Invalid number: '{new_str}' – edit cancelled."

        # --- Save ---
        elif key in (ord("s"), ord("S")) and not (key in (ord("s"), ord("S")) and
              curses.keyname(key) in (b'^S',)):
            # plain S is navigation; Ctrl+S is save — but we also bind capital S
            pass  # handled by navigation above; use dedicated save key below

        elif key in (curses.KEY_F2,):  # F2 = save shortcut (extra)
            save_qtable(qtable, filepath)
            modified = False
            status_msg = f"Saved to {filepath}"

        # Ctrl+S
        elif key == 19:   # Ctrl+S
            save_qtable(qtable, filepath)
            modified = False
            status_msg = f"Saved to {filepath}"

        # Capital letter commands (dedicated, not navigation)
        elif key == ord("E"):
            pass  # already handled above (same as e)

        elif key == ord("V"):  # V = save
            save_qtable(qtable, filepath)
            modified = False
            status_msg = f"Saved to {filepath}"

        # S (save) — when CAPS LOCK via explicit binding
        # We use number keys as extra bindings for save / saveAs / find / info / reload
        elif key == ord("1"):  # 1 = Save
            save_qtable(qtable, filepath)
            modified = False
            status_msg = f"Saved → {filepath}"

        elif key == ord("2"):  # 2 = Save As
            new_path = prompt_string(stdscr, "Save As: ", filepath)
            if new_path:
                save_qtable(qtable, new_path)
                filepath = new_path
                status_msg = f"Saved → {filepath}"
                modified = False

        elif key == ord("3"):  # 3 = Find
            query = prompt_string(stdscr, "Find state (substring): ")
            query = query.strip()
            found = [(i, r) for i, r in enumerate(rows) if query in str(r[0])]
            if found:
                cursor_row = found[0][0]
                status_msg = f"Found {len(found)} match(es). Jumped to row {cursor_row}."
            else:
                status_msg = f"No state matching '{query}' found."

        elif key == ord("4"):  # 4 = Info
            state_key, qvals = rows[cursor_row]
            val = qvals[cursor_col]
            prec = detect_precision(val)
            status_msg = (f"State={state_key}  Action={action_labels[cursor_col]}  "
                          f"val={val!r}  type={type(val).__name__}  "
                          f"~prec={prec}dp  range=[{q_min:.4g},{q_max:.4g}]")

        elif key == ord("5"):  # 5 = Reload
            confirm = prompt_string(stdscr, "Reload from disk? Unsaved changes lost. (y/N): ")
            if confirm.lower() == "y":
                qtable = load_qtable(filepath)
                rows, action_labels = qtable_to_rows(qtable)
                n_rows = len(rows)
                n_cols = len(action_labels)
                cursor_row = min(cursor_row, n_rows - 1)
                cursor_col = min(cursor_col, n_cols - 1)
                modified = False
                modified_cells.clear()
                status_msg = "Reloaded from disk."

        # --- Legacy letter shortcuts (lowercase, non-navigation) ---
        elif key in (ord("f"), ord("F")):
            query = prompt_string(stdscr, "Find state (substring): ")
            found = [(i, r) for i, r in enumerate(rows) if query in str(r[0])]
            if found:
                cursor_row = found[0][0]
                status_msg = f"Found {len(found)} match(es). Jumped to row {cursor_row}."
            else:
                status_msg = f"No state matching '{query}' found."

        elif key in (ord("i"), ord("I")):
            state_key, qvals = rows[cursor_row]
            val = qvals[cursor_col]
            prec = detect_precision(val)
            status_msg = (f"State={state_key}  Action={action_labels[cursor_col]}  "
                          f"val={val!r}  type={type(val).__name__}  "
                          f"~prec={prec}dp  range=[{q_min:.4g},{q_max:.4g}]")

        elif key in (ord("r"), ord("R")):
            confirm = prompt_string(stdscr, "Reload from disk? Unsaved changes lost. (y/N): ")
            if confirm.lower() == "y":
                qtable = load_qtable(filepath)
                rows, action_labels = qtable_to_rows(qtable)
                n_rows = len(rows)
                n_cols = len(action_labels)
                cursor_row = min(cursor_row, n_rows - 1)
                cursor_col = min(cursor_col, n_cols - 1)
                modified = False
                modified_cells.clear()
                status_msg = "Reloaded from disk."

        elif key in (curses.KEY_F5,):  # F5 = save
            save_qtable(qtable, filepath)
            modified = False
            status_msg = f"Saved → {filepath}"

        # Save  ← explicit 's' after upper-case guard
        # (Lower-case s is navigation DOWN; upper-case S IS also navigation.
        #  Use Ctrl+S / F5 / numeric '1' for save.)

        elif key in (ord("q"), ord("Q"), 27):  # Q or Esc
            if modified:
                confirm = prompt_string(
                    stdscr, "Unsaved changes! Save before quitting? (y/n/cancel): ")
                if confirm.lower() == "y":
                    save_qtable(qtable, filepath)
                    break
                elif confirm.lower() == "n":
                    break
                # cancel → stay
            else:
                break


def main():
    if len(sys.argv) < 2:
        # Interactive path input if no arg given
        filepath = input("Enter path to Q-table .pkl file: ").strip()
    else:
        filepath = sys.argv[1]

    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    curses.wrapper(run_editor, filepath)
    print("Editor closed.")


if __name__ == "__main__":
    main()
