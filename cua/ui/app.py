"""Buddy: voice-first, macOS-inspired desktop assistant.   Run:  python -m cua.ui   (or start_buddy.bat)

tkinter ships with Python, so this adds no dependencies and only a few MB of RAM. Task execution lives in the
shared AgentService (cua/agent/service.py) -- this window is one client of it, exactly like the phone. The UI
thread only draws and polls queues; a background thread pumps the service's event stream into it.
Follows the Windows light/dark setting.
"""
from __future__ import annotations

import ctypes
import math
import os
import queue
import threading
import tkinter as tk
from tkinter import font as tkfont

import win32con
import win32gui

from cua.agent.router import RouterPlanner
from cua.agent.service import AgentService
from cua.input.voice import VoiceEvent, VoiceInput, VoiceState, build_voice
from cua.types import Step, TaskResult

LIGHT = dict(bg="#F5F5F7", text="#1D1D1F", sub="#86868B", bot="#E9E9EB", me="#0A84FF", field="#FFFFFF",
             border="#D2D2D7", chip="#FFFFFF", off="#D1D1D6", disabled="#C7C7CC", card="#FFFFFF",
             voice_bg="#E9E9EB", voice_active="#FFE5E5")
DARK = dict(bg="#1C1C1E", text="#F5F5F7", sub="#98989D", bot="#2C2C2E", me="#0A84FF", field="#2C2C2E",
            border="#3A3A3C", chip="#2C2C2E", off="#48484A", disabled="#48484A", card="#2C2C2E",
            voice_bg="#2C2C2E", voice_active="#3A1C1C")
GREEN, RED, AMBER = "#30D158", "#FF453A", "#FF9F0A"

EXAMPLES = ["Open Chrome", "Downloads folder", "Bluetooth settings", "Search SpaceX"]
S = 1.0


def px(v):
    return int(round(v * S))


def system_is_dark() -> bool:
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
        return winreg.QueryValueEx(k, "AppsUseLightTheme")[0] == 0
    except Exception:
        return False


def rrect(c: tk.Canvas, x1, y1, x2, y2, r, **kw):
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2, x2 - r, y2,
           x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return c.create_polygon(pts, smooth=True, **kw)


def describe(step: Step) -> str:
    a, g = step.action, step.args
    named = g.get("name") or g.get("title") or g.get("path") or g.get("query") or g.get("url") or g.get("folder") \
            or g.get("page") or g.get("keys") or g.get("key") or ""
    special = {
        "open_app": f"Open {g.get('name')}", "navigate_url": "Open the website", "web_search": "Search the web",
        "open_settings": f"Open {g.get('page')} settings", "find_file": "Find the file", "open_file": "Open the file",
        "create_file": f"Create {g.get('path')}", "type_text": f"Type “{str(g.get('text'))[:28]}”",
        "type_into_ui": f"Type into “{g.get('name')}”", "hotkey": f"Press {g.get('keys')}",
        "press_key": f"Press {g.get('key')}", "click_ui": f"Click “{g.get('name') or g.get('control_type')}”",
        "focus_app": f"Switch to {g.get('name')}", "wait": "Wait a moment",
    }
    return special.get(a) or f"{a.replace('_', ' ').capitalize()} {named}".strip()


# ---- widgets ---------------------------------------------------------------
class Bubble(tk.Canvas):
    def __init__(self, parent, lines, fill, fg, font, page_bg, max_w):
        pad, gap = px(12), px(3)
        lh = font.metrics("linespace")
        laid = []
        for mark, mc, text, tc in lines:
            indent = font.measure(mark + " ") if mark else 0
            for i, seg in enumerate(self._wrap(text, font, max_w - 2 * pad - indent)):
                laid.append((mark if i == 0 else "", mc, seg, tc, indent))
        w = max((indent + font.measure(t) for _, _, t, _, indent in laid), default=0) + 2 * pad
        h = len(laid) * lh + (len(laid) - 1) * gap + 2 * pad - px(2)
        super().__init__(parent, width=w, height=h, bg=page_bg, highlightthickness=0)
        rrect(self, 0, 0, w - 1, h - 1, px(17), fill=fill, outline="")
        y = pad - px(2)
        for mark, mc, t, tc, indent in laid:
            if mark:
                self.create_text(pad, y, text=mark, anchor="nw", fill=mc or fg, font=font)
            self.create_text(pad + indent, y, text=t, anchor="nw", fill=tc or fg, font=font)
            y += lh + gap

    @staticmethod
    def _wrap(text, font, width):
        out = []
        for para in text.split("\n"):
            cur = ""
            for word in para.split(" "):
                trial = (cur + " " + word).strip()
                if font.measure(trial) <= width or not cur:
                    cur = trial
                else:
                    out.append(cur)
                    cur = word
            out.append(cur)
        return out


class MicButton(tk.Canvas):
    """Large circular mic button — the visual centre of the voice-first interface."""

    def __init__(self, parent, theme, command, size=56):
        s = px(size)
        super().__init__(parent, width=s, height=s, bg=theme["bg"], highlightthickness=0, cursor="hand2")
        self.s, self.cmd, self.t = s, command, theme
        self._state = "idle"
        self.bind("<Button-1>", lambda e: self.cmd())
        self.redraw()

    def set_state(self, state: str):
        self._state = state
        self.redraw()

    def redraw(self):
        self.delete("all")
        s = self.s
        c = s / 2
        fills = {"idle": self.t["voice_bg"], "listening": RED, "processing": AMBER,
                 "executing": self.t["me"], "success": GREEN, "error": RED}
        fill = fills.get(self._state, self.t["voice_bg"])
        self.create_oval(2, 2, s - 2, s - 2, fill=fill, outline="")
        fg = "white" if self._state != "idle" else self.t["text"]

        if self._state == "listening":
            # pulsing waves
            for i, r in enumerate([px(6), px(9), px(12)]):
                self.create_oval(c - r, c - r, c + r, c + r, outline=fg, width=px(1.5))
        elif self._state in ("processing", "executing"):
            # spinning dots
            for i in range(3):
                angle = i * 2.094
                dx, dy = math.cos(angle) * px(10), math.sin(angle) * px(10)
                self.create_oval(c + dx - px(3), c + dy - px(3), c + dx + px(3), c + dy + px(3), fill=fg, outline="")
        elif self._state == "success":
            # checkmark
            self.create_line(c - px(8), c, c - px(2), c + px(7), c + px(10), c - px(7),
                             fill=fg, width=px(3), capstyle="round", joinstyle="round")
        elif self._state == "error":
            # X
            self.create_line(c - px(7), c - px(7), c + px(7), c + px(7), fill=fg, width=px(3), capstyle="round")
            self.create_line(c - px(7), c + px(7), c + px(7), c - px(7), fill=fg, width=px(3), capstyle="round")
        else:
            # mic icon
            self.create_oval(c - px(5), c - px(11), c + px(5), c + px(3), fill=fg, outline="")
            self.create_arc(c - px(9), c - px(6), c + px(9), c + px(10), start=180, extent=180,
                            outline=fg, style="arc", width=px(2))
            self.create_line(c, c + px(10), c, c + px(14), fill=fg, width=px(2), capstyle="round")
            self.create_line(c - px(5), c + px(14), c + px(5), c + px(14), fill=fg, width=px(2), capstyle="round")


class Switch(tk.Canvas):
    def __init__(self, parent, theme, label, var, font, on_change=None):
        w, h = px(34), px(20)
        super().__init__(parent, width=w + font.measure(label) + px(12), height=h, bg=theme["bg"],
                         highlightthickness=0, cursor="hand2")
        self.t, self.var, self.w, self.h = theme, var, w, h
        self.label, self.font, self.on_change = label, font, on_change
        self.bind("<Button-1>", self._toggle)
        self.redraw()

    def _toggle(self, _):
        self.var.set(not self.var.get())
        self.redraw()
        if self.on_change:
            self.on_change()

    def redraw(self):
        self.delete("all"); on = self.var.get(); w, h = self.w, self.h
        rrect(self, 0, 0, w, h, h // 2, fill=GREEN if on else self.t["off"], outline="")
        kx = w - h + px(2) if on else px(2)
        self.create_oval(kx, px(2), kx + h - px(4), h - px(2), fill="white", outline="")
        self.create_text(w + px(8), h // 2, text=self.label, anchor="w", fill=self.t["sub"], font=self.font)


class Chip(tk.Canvas):
    def __init__(self, parent, theme, text, command, font, w):
        w, h = px(w), px(30)
        super().__init__(parent, width=w, height=h, bg=theme["bg"], highlightthickness=0, cursor="hand2")
        self.shape = rrect(self, 1, 1, w - 1, h - 1, h // 2, fill=theme["chip"], outline=theme["border"])
        self.create_text(w // 2, h // 2, text=text, fill=theme["text"], font=font)
        self.bind("<Button-1>", lambda e: command())
        self.bind("<Enter>", lambda e: self.itemconfig(self.shape, outline=theme["me"]))
        self.bind("<Leave>", lambda e: self.itemconfig(self.shape, outline=theme["border"]))


class ActivityItem(tk.Frame):
    """One line in the recent activity list."""

    def __init__(self, parent, theme, text, ok, font, timing=""):
        super().__init__(parent, bg=theme["bg"])
        mark = "✓" if ok else "✗"
        color = GREEN if ok else RED
        tk.Label(self, text=mark, font=font, fg=color, bg=theme["bg"], width=2).pack(side="left")
        tk.Label(self, text=text, font=font, fg=theme["text"], bg=theme["bg"], anchor="w").pack(side="left", fill="x", expand=True)
        if timing:
            tk.Label(self, text=timing, font=font, fg=theme["sub"], bg=theme["bg"]).pack(side="right")


# ---- app -------------------------------------------------------------------
class BuddyApp:
    def __init__(self, service: AgentService | None = None):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        self.root = root = tk.Tk()
        global S
        S = max(1.0, root.winfo_fpixels("1i") / 96.0)
        self.T = T = DARK if system_is_dark() else LIGHT
        fams = set(tkfont.families())
        face = "Segoe UI Variable Text" if "Segoe UI Variable Text" in fams else "Segoe UI"
        self.f_body = tkfont.Font(family=face, size=10)
        self.f_small = tkfont.Font(family=face, size=8)
        self.f_title = tkfont.Font(family=face, size=14, weight="bold")
        self.f_sub = tkfont.Font(family=face, size=10)
        self.f_in = tkfont.Font(family=face, size=11)
        self.f_transcript = tkfont.Font(family=face, size=12)

        root.title("Buddy")
        w, h = px(400), min(px(620), root.winfo_screenheight() - px(90))
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{w}x{h}+{sw - w - px(16)}+{max(0, (sh - h) // 2 - px(24))}")
        root.minsize(px(340), px(440))
        root.configure(bg=T["bg"])

        self.events: queue.Queue = queue.Queue()   # local UI signaling only: confirm requests, voice events
        self.busy, self.history, self.hidx, self.t = False, [], 0, 0
        self.preview = tk.BooleanVar(value=False)
        self.top = tk.BooleanVar(value=False)
        self.ai = tk.BooleanVar(value=True)
        self.voice_enabled = tk.BooleanVar(value=True)
        self._nsteps = 0
        self.hidden = False
        self.voice = build_voice(on_event=self._voice_event)
        self.voice.warm()          # pre-load the STT model now, not on the user's first mic click
        self._voice_state = VoiceState.IDLE

        # One shared Agent for every interface: this window submits to it exactly like the phone does.
        self.service = service or AgentService(confirm=self._confirm_from_worker)
        self.kill = self.service.kill
        self._llm = self.service.agent.planner.llm
        self._task_id: str | None = None
        self._task_q = self.service.subscribe()

        self._build()
        threading.Thread(target=self._event_pump, daemon=True, name="service-event-pump").start()
        root.protocol("WM_DELETE_WINDOW", self._quit)
        root.after(50, self._poll)
        root.after(70, self._pulse)
        if self.kill.error:
            self._add_activity(f"Kill switch: {self.kill.error}", False)

    # ---- layout ------------------------------------------------------------
    def _build(self):
        r, T = self.root, self.T

        # Title
        head = tk.Frame(r, bg=T["bg"])
        head.pack(side="top", fill="x", padx=px(24), pady=(px(28), px(4)))
        self.dot = tk.Canvas(head, width=px(10), height=px(10), bg=T["bg"], highlightthickness=0)
        self.dot.pack(side="left", padx=(0, px(8)), pady=(px(4), 0))
        tk.Label(head, text="Buddy", font=self.f_title, fg=T["text"], bg=T["bg"]).pack(side="left")
        self.status = tk.Label(head, text="Ready", font=self.f_small, fg=T["sub"], bg=T["bg"])
        self.status.pack(side="right", pady=(px(4), 0))

        # Subtitle
        self.subtitle = tk.Label(r, text="What can I help with?", font=self.f_sub, fg=T["sub"], bg=T["bg"])
        self.subtitle.pack(side="top", pady=(px(2), px(16)))

        # Mic button (centre of the interface)
        mic_frame = tk.Frame(r, bg=T["bg"])
        mic_frame.pack(side="top", pady=(px(4), px(8)))
        self.mic_btn = MicButton(mic_frame, T, self._mic_pressed, size=60)
        self.mic_btn.pack()

        # Transcript display
        self.transcript = tk.Label(r, text="", font=self.f_transcript, fg=T["text"], bg=T["bg"],
                                    wraplength=px(340))
        self.transcript.pack(side="top", pady=(px(4), px(8)))

        # Text input bar
        bar = tk.Frame(r, bg=T["bg"])
        bar.pack(side="top", fill="x", padx=px(20), pady=(px(4), px(8)))
        self.box = tk.Canvas(bar, height=px(40), bg=T["bg"], highlightthickness=0)
        self.box.pack(side="left", fill="x", expand=True)
        self.entry = tk.Entry(self.box, relief="flat", bg=T["field"], fg=T["text"], insertbackground=T["text"],
                              font=self.f_in, bd=0, highlightthickness=0)
        self._box_win = self.box.create_window(px(16), px(20), window=self.entry, anchor="w")
        self.box.bind("<Configure>", self._draw_box)
        self.entry.bind("<Return>", lambda e: self._send())
        self.entry.bind("<KeyRelease>", lambda e: self._sync_button())
        self.entry.bind("<Up>", lambda e: self._hist(-1))
        self.entry.bind("<Down>", lambda e: self._hist(1))

        self.send_cv = tk.Canvas(bar, width=px(36), height=px(36), bg=T["bg"], highlightthickness=0, cursor="hand2")
        self.send_cv.pack(side="left", padx=(px(6), 0))
        self.send_cv.bind("<Button-1>", lambda e: self._send())
        self._draw_send()

        # Quick chips
        chips = tk.Frame(r, bg=T["bg"])
        chips.pack(side="top", fill="x", padx=px(20), pady=(0, px(8)))
        for i, ex in enumerate(EXAMPLES):
            Chip(chips, T, ex, lambda e=ex: self.submit(e), self.f_small, 90).grid(
                row=0, column=i, padx=px(2), pady=px(2), sticky="w")

        # Separator
        sep = tk.Canvas(r, height=1, bg=T["border"], highlightthickness=0)
        sep.pack(fill="x", padx=px(20), pady=(px(4), px(4)))

        # Bottom: kill switch label
        tk.Label(r, text=f"Stop: {self.kill.hotkey.upper().replace('+', ' + ')}", font=self.f_small,
                 fg=T["sub"], bg=T["bg"]).pack(side="bottom", pady=(px(2), px(10)))

        # Settings row
        opts = tk.Frame(r, bg=T["bg"])
        opts.pack(side="bottom", fill="x", padx=px(20), pady=(0, px(2)))
        Switch(opts, T, "Preview", self.preview, self.f_small).pack(side="left", padx=(0, px(10)))
        Switch(opts, T, "On top", self.top, self.f_small, self._apply_top).pack(side="left", padx=(0, px(10)))
        Switch(opts, T, "AI", self.ai, self.f_small, self._apply_ai).pack(side="left", padx=(0, px(10)))
        Switch(opts, T, "Voice", self.voice_enabled, self.f_small).pack(side="left")

        # Confirmation slot
        self.confirm_slot = tk.Frame(r, bg=T["bg"])
        self.confirm_slot.pack(side="bottom", fill="x", padx=px(16))

        # Recent activity (scrollable)
        act_label = tk.Label(r, text="Recent activity", font=self.f_small, fg=T["sub"], bg=T["bg"], anchor="w")
        act_label.pack(side="top", fill="x", padx=px(24), pady=(px(4), px(2)))
        self.activity_frame = tk.Frame(r, bg=T["bg"])
        self.activity_frame.pack(side="top", fill="both", expand=True, padx=px(20))

        self.entry.focus_set()
        self._draw_dot()

    def _draw_box(self, e):
        self.box.delete("bg")
        rrect(self.box, 1, 1, e.width - 1, px(39), px(19), fill=self.T["field"], outline=self.T["border"], tags="bg")
        self.box.tag_lower("bg")
        self.box.coords(self._box_win, px(16), px(20))
        self.entry.configure(width=max(6, (e.width - px(36)) // px(9)))

    def _draw_send(self, active=False):
        self.send_cv.delete("all")
        s = px(36)
        fill = self.T["me"] if active else self.T["disabled"]
        self.send_cv.create_oval(1, 1, s - 1, s - 1, fill=fill, outline="")
        c = s / 2
        a = px(7)
        self.send_cv.create_line(c, c + a, c, c - a, fill="white", width=px(2.5), capstyle="round")
        self.send_cv.create_line(c - a, c - px(1), c, c - a - px(1), c + a, c - px(1),
                                 fill="white", width=px(2.5), capstyle="round", joinstyle="round")

    def _sync_button(self):
        if not self.busy:
            self._draw_send(bool(self.entry.get().strip()))

    # ---- voice integration -------------------------------------------------
    def _mic_pressed(self):
        if self.busy:
            self.kill.trigger()
            return
        if not self.voice_enabled.get():
            self.entry.focus_set()
            return
        if self._voice_state == VoiceState.IDLE:
            if hasattr(self.voice, 'available') and not self.voice.available:
                self._mood("Voice not available", AMBER)
                self.entry.focus_set()
                return
            self.voice.start_listening()
        elif self._voice_state == VoiceState.LISTENING:
            text = self.voice.stop_listening()
            if text:
                self.submit(text)

    def _voice_event(self, event: VoiceEvent, data):
        self.events.put(("voice", event, data))

    def _on_voice(self, event: VoiceEvent, data):
        if event == VoiceEvent.STATE_CHANGED:
            self._voice_state = data
            if data == VoiceState.LISTENING:
                self.mic_btn.set_state("listening")
                self.subtitle.config(text="Listening...")
                self.transcript.config(text="")
                self._mood("Listening")
            elif data == VoiceState.PROCESSING:
                self.mic_btn.set_state("processing")
                self.subtitle.config(text="Processing...")
                self._mood("Processing")
            elif data == VoiceState.IDLE:
                if not self.busy:
                    self.mic_btn.set_state("idle")
                    self.subtitle.config(text="What can I help with?")
        elif event == VoiceEvent.PARTIAL_TRANSCRIPT:
            self.transcript.config(text=data)
        elif event == VoiceEvent.FINAL_TRANSCRIPT:
            self.transcript.config(text=f"Buddy heard: “{data}”")
            self.submit(data)
        elif event == VoiceEvent.METRICS:
            m = data.to_dict()
            if m.get("rejected"):
                self.transcript.config(text="Didn't catch that — say it again?")
                self._mood("Didn't catch that", AMBER)
            self._add_activity(
                f"heard on {m['device']} ({m['sample_rate']}Hz): raw='{m['raw_transcript']}' "
                f"final='{m['final_text']}' conf={m['confidence']:.2f}"
                f"{' REJECTED' if m.get('rejected') else ''} [{m['audio_duration_s']:.1f}s audio, "
                f"stt {m['stt_ms']:.0f}ms]", not m.get("rejected"))
        elif event == VoiceEvent.ERROR:
            self.mic_btn.set_state("error")
            self._mood(f"Voice error: {data}", RED)
            self.root.after(2000, lambda: self.mic_btn.set_state("idle"))

    # ---- window handling ---------------------------------------------------
    def _hwnd(self) -> int:
        return int(self.root.wm_frame(), 16)

    def _apply_top(self):
        self.root.attributes("-topmost", bool(self.top.get()))

    def _apply_ai(self):
        if self._llm is not None:
            agent = self.service.agent
            agent.planner.llm = self._llm if self.ai.get() else None
            if not self.ai.get() and agent.runtime is not None:
                agent.runtime.unload()

    def _hide(self):
        win32gui.ShowWindow(self._hwnd(), win32con.SW_MINIMIZE)
        self.hidden = True

    def _show_quietly(self, on_top: bool):
        h = self._hwnd()
        win32gui.ShowWindow(h, win32con.SW_SHOWNOACTIVATE)
        pos = win32con.HWND_TOPMOST if on_top or self.top.get() else win32con.HWND_BOTTOM
        win32gui.SetWindowPos(h, pos, 0, 0, 0, 0,
                              win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)
        self.hidden = False

    # ---- status dot --------------------------------------------------------
    def _draw_dot(self, color=None):
        self.dot.delete("all")
        c = color or (self.T["me"] if self.busy else GREEN)
        r = px(4)
        if self.busy:
            r = px(3) + round(px(2) * (1 + math.sin(self.t / 2)))
        m = px(5)
        self.dot.create_oval(m - r, m - r, m + r, m + r, fill=c, outline="")

    def _pulse(self):
        self.t += 1
        if self.busy:
            self._draw_dot()
        self.root.after(70, self._pulse)

    def _mood(self, text, color=None):
        self.status.config(text=text)
        self._draw_dot(color)

    # ---- activity list -----------------------------------------------------
    def _add_activity(self, text, ok, timing=""):
        item = ActivityItem(self.activity_frame, self.T, text, ok, self.f_small, timing)
        item.pack(side="top", fill="x", pady=px(1))
        children = self.activity_frame.winfo_children()
        if len(children) > 12:
            children[0].destroy()

    # ---- actions -----------------------------------------------------------
    def _hist(self, d):
        if not self.history:
            return
        self.hidx = max(0, min(len(self.history), self.hidx + d))
        self.entry.delete(0, "end")
        if self.hidx < len(self.history):
            self.entry.insert(0, self.history[self.hidx])
        self._sync_button()

    def _send(self):
        if self.busy:
            self.kill.trigger()
        else:
            self.submit()

    def submit(self, text: str | None = None):
        text = (text or self.entry.get()).strip()
        if not text or self.busy:
            return
        self.entry.delete(0, "end")
        self.history.append(text)
        self.hidx = len(self.history)
        self.transcript.config(text=text)

        if self.preview.get():
            return self._show_plan(text)

        self._set_busy(True)
        self.mic_btn.set_state("executing")
        busy_elsewhere = self.service.is_busy()       # e.g. the phone already has a task running
        rec = self.service.submit_task(text, source="laptop")
        self._task_id = rec.id
        if busy_elsewhere:
            self._mood("Buddy is busy — queued", AMBER)
            self.subtitle.config(text="Waiting for the current task to finish…")
        else:
            self._mood("Working…")
            self.subtitle.config(text=f"“{text[:40]}”")

        plan = RouterPlanner().plan(text)
        if plan and any(s.action in ("type_text", "hotkey", "press_key", "click", "double_click", "right_click",
                                     "scroll", "click_ui", "type_into_ui") and not s.args.get("app")
                        for s in plan.steps):
            self._hide()

    def _show_plan(self, text):
        plan = RouterPlanner().plan(text)
        if plan is None:
            self._add_activity("Can't plan that yet", False)
            self._mood("Not sure how", AMBER)
        else:
            for s in plan.steps:
                self._add_activity(describe(s), True)
            self._mood("Plan ready")

    def _set_busy(self, on: bool):
        self.busy = on
        self._draw_send(False if on else bool(self.entry.get().strip()))

    # ---- shared-service event pump ------------------------------------------
    def _event_pump(self):
        """Drains AgentService's broadcast queue -- the same stream the phone's WebSocket reads -- on a
        background thread, and hands events for OUR task onto the local UI queue for _poll() to render."""
        while True:
            try:
                evt = self._task_q.get(timeout=1)
            except queue.Empty:
                continue
            if evt.get("id") != self._task_id:
                continue                  # someone else's task (e.g. the phone submitted one); not ours to show
            self.events.put(("service_event", evt))

    def _confirm_from_worker(self, why: str) -> bool:
        ev, box = threading.Event(), {"ok": False}
        self.events.put(("confirm", why, box, ev))
        while not ev.wait(0.1):
            self.kill.check()
        return box["ok"]

    def _poll(self):
        try:
            while True:
                msg = self.events.get_nowait()
                kind = msg[0]
                if kind == "voice":
                    self._on_voice(msg[1], msg[2])
                else:
                    getattr(self, "_on_" + kind)(*msg[1:])
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def _on_service_event(self, evt: dict):
        kind = evt["type"]
        if kind == "planning":
            self._mood("Planning…")
        elif kind == "executing":
            rec = self.service.get_task(self._task_id)
            if rec and rec.plan:
                self._nsteps = len(rec.plan.steps)
                if rec.plan.source == "llm":
                    for s in rec.plan.steps:
                        self._add_activity(f"Plan: {describe(s)}", True)
        elif kind == "step_start":
            step = Step(evt["action"], evt.get("args") or {})
            self._mood(f"Step {evt['index'] + 1}/{self._nsteps}: {describe(step)}")
            self.subtitle.config(text=describe(step))
        elif kind in ("completed", "failed", "cancelled"):
            rec = self.service.get_task(self._task_id)
            if rec and rec.result is not None:
                self._on_result(rec.result)
            else:
                self._on_crash(evt.get("reason", "unknown error"))

    def _on_confirm(self, why, box, ev):
        T = self.T
        self._show_quietly(on_top=True)
        self._clear_confirm()
        card = tk.Canvas(self.confirm_slot, height=px(92), bg=T["bg"], highlightthickness=0)
        card.pack(fill="x", pady=(0, px(8)))

        def draw(_=None):
            card.delete("bgcard")
            rrect(card, 1, 1, card.winfo_width() - 1, px(90), px(14), fill=T["card"], outline=T["border"], tags="bgcard")
            card.tag_lower("bgcard")
        card.bind("<Configure>", draw)
        card.create_text(px(16), px(20), text=f"Allow: {why}?", anchor="w", fill=T["text"], font=self.f_sub)
        card.create_text(px(16), px(40), text="This changes something on your computer.", anchor="w",
                         fill=T["sub"], font=self.f_small)

        def answer(ok):
            box["ok"] = ok
            card.destroy()
            self._apply_top()
            ev.set()

        def pill(x, text, fill, fg, cmd, w):
            cv = tk.Canvas(card, width=px(w), height=px(28), bg=T["card"], highlightthickness=0, cursor="hand2")
            rrect(cv, 0, 0, px(w) - 1, px(27), px(14), fill=fill, outline="")
            cv.create_text(px(w) // 2, px(14), text=text, fill=fg, font=self.f_small)
            cv.bind("<Button-1>", lambda e: cmd())
            card.create_window(px(x), px(66), window=cv, anchor="w")
        pill(16, "Allow", T["me"], "white", lambda: answer(True), 78)
        pill(102, "Cancel", T["border"], T["text"], lambda: answer(False), 78)
        self._mood("Waiting for your OK", AMBER)

    def _clear_confirm(self):
        for w in self.confirm_slot.winfo_children():
            w.destroy()

    def _on_crash(self, err):
        self._finish()
        self._add_activity(f"Error: {err[:60]}", False)
        self._mood("Error", RED)

    def _on_result(self, res: TaskResult):
        self._finish()
        self._clear_confirm()

        if res.reason == "no_plan":
            # Say WHY. "Don't know how to do that yet" was shown even when the real cause was that the
            # planner model couldn't load for lack of RAM -- indistinguishable, from the outside, from the
            # assistant simply not understanding, and impossible for the user to act on.
            why = (res.metrics or {}).get("planner_error") or ""
            if "RAM" in why:
                self._add_activity(f"AI planner unavailable: {why}", False)
                self._mood("Low memory — close some apps", AMBER)
            else:
                self._add_activity("Don't know how to do that yet", False)
                self._mood("Not sure how", AMBER)
            return

        for s in res.steps:
            self._add_activity(describe(s.step), s.ok, f"{s.ms:.0f}ms" if s.ms else "")

        if res.ok:
            self.mic_btn.set_state("success")
            self._mood(f"Done in {res.total_ms / 1000:.1f}s")
            self.subtitle.config(text="Done!")
            self.root.after(2500, self._reset_idle)
        elif res.reason.startswith("aborted"):
            self._add_activity("Stopped", False)
            self._mood("Stopped", AMBER)
            self.mic_btn.set_state("idle")
        elif res.reason == "declined":
            self._add_activity("Cancelled by user", False)
            self._mood("Ready")
            self.mic_btn.set_state("idle")
        else:
            self.mic_btn.set_state("error")
            self._mood("That didn't work", RED)
            self.root.after(2500, self._reset_idle)

    def _reset_idle(self):
        if not self.busy:
            self.mic_btn.set_state("idle")
            self.subtitle.config(text="What can I help with?")
            self._mood("Ready")

    def _finish(self):
        self._set_busy(False)
        if self.hidden:
            self._show_quietly(on_top=False)

    def _quit(self):
        self.service.unsubscribe(self._task_q)
        self.kill.stop()
        if self.voice:
            self.voice.unload()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

    # ---- phone access (same shared service, reachable over the LAN) --------
    def start_phone_server(self, host: str = "0.0.0.0", port: int = 8420):
        """Exposes this window's AgentService to the phone PWA over the network, in a background thread.
        Because it is the SAME AgentService the text/voice input above submits to, the laptop and any
        connected phone share one Agent and one model runtime -- never two."""
        def _run():
            try:
                from cua.server.api import run_server
                run_server(self.service, voice=self.voice, host=host, port=port)
            except OSError as e:
                self._add_activity(f"Phone access unavailable: {e}", False)
        threading.Thread(target=_run, daemon=True, name="phone-server").start()


def main():
    app = BuddyApp()
    if os.environ.get("CUA_NO_PHONE_SERVER") != "1":
        app.start_phone_server()
    app.run()


if __name__ == "__main__":
    main()
