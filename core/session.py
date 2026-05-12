"""
MediCareGuide Session Module
=========================
Tracks conversation state and decides whether the session is in WELCOME
or SELECT mode.

Profile values are written exclusively through set_intake_field() —
structured UI selections for steps 1–4 and ZIP text input for step 0.
process_turn() is a pure Q&A turn handler and never writes to the profile.

Usage:
    from core.lookup import MediCareGuideLookup
    from core.session import MediCareGuideSession

    lookup = MediCareGuideLookup("CY2026_Landscape_202603.csv")
    session = MediCareGuideSession(lookup)

    # structured intake
    session.set_intake_field(0, "zip", "24073")

    # free-form Q&A
    state = session.process_turn("What is a MOOP?")
    # ... call Ollama ...
    session.close_turn(answer)
"""

from __future__ import annotations

from core.lookup import MediCareGuideLookup


class MediCareGuideSession:
    """
    Manages per-conversation state for a MediCareGuide session.

    Maintains a profile (zip, county, state, track, snp_flags, budget_max),
    context hints (has_rx, keep_doctors, wants_dental, prefers_ppo),
    conversation history, and the current mode (WELCOME | SELECT).
    """

    def __init__(self, lookup: MediCareGuideLookup):
        self.lookup = lookup
        self.state: dict = {
            "mode": "WELCOME",
            "intake_step": 0,
            "sort_key": None,        # None = auto-derive from profile/context
            "profile": {
                "zip": None,
                "county": None,
                "state": None,
                "track": None,
                "snp_flags": [],
                "budget_max": None,
            },
            "context": {
                "has_rx": False,
                "keep_doctors": False,
                "wants_dental": False,
                "prefers_ppo": False,
            },
            "history": [],
        }

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process_turn(self, text: str) -> dict:
        """
        Called at the start of every Q&A turn.

        Does NOT write to the session profile — profile values come
        exclusively from set_intake_field(). This method is a pure
        Q&A handler: it resolves the current mode, appends the user
        message to history, and returns the full state dict.

        1. Resolves the session mode.
        2. Appends the user message to history.
        3. Returns the full state dict.
        """
        self._resolve_mode()
        self.state["history"].append({"role": "user", "content": text})
        return self.state

    def set_intake_field(
        self,
        step: int,
        field: str,
        value,
        action: str = "set",
    ) -> dict:
        """
        Called by the UI when the user submits an intake step.

        step:   the current intake step (0–4)
        field:  "zip"        → profile["zip"] / county / state  (step 0, string)
                "track"      → profile["track"]                 (step 1, string)
                "snp_flags"  → profile["snp_flags"]             (step 2, list of flag strings)
                "budget_max" → profile["budget_max"]            (step 3, int or None)
                "context"    → context flags                    (step 4, list of flag names)
        value:  the value to write; ignored for action="skip"
        action: "set"  → write value, advance intake_step
                "skip" → advance intake_step, no value written

        For field="zip", the ZIP is resolved to county + state via _resolve_zip().
        If the ZIP is not found, profile["zip"] stays None — the caller should
        check state["profile"]["zip"] to determine whether resolution succeeded.
        """
        if action == "set":
            if field == "zip":
                locs = self.lookup._resolve_zip(value)
                if locs:
                    self.state["profile"]["zip"]    = value
                    self.state["profile"]["county"] = locs[0]["county"]
                    self.state["profile"]["state"]  = locs[0]["state"]
                # if ZIP not found, profile["zip"] stays None — caller checks this
            elif field == "snp_flags":
                for flag in (value or []):
                    if flag not in self.state["profile"]["snp_flags"]:
                        self.state["profile"]["snp_flags"].append(flag)
            elif field == "context":
                for flag in (value or []):
                    if flag in self.state["context"]:
                        self.state["context"][flag] = True
            elif field in self.state["profile"]:
                self.state["profile"][field] = value

        # Both "set" and "skip" advance the step
        self.state["intake_step"] = max(self.state["intake_step"], step + 1)
        self._resolve_mode()

        # Synthetic history entry so conversation context stays coherent
        label = str(value) if action == "set" else "(skipped)"
        self.state["history"].append({"role": "user", "content": f"[selected: {label}]"})

        return self.state

    def close_turn(self, answer: str) -> None:
        """
        Called after Ollama returns an answer.

        Appends the assistant reply to history and trims to the last 12
        entries (6 user+assistant pairs).
        """
        self.state["history"].append({"role": "assistant", "content": answer})
        self.state["history"] = self.state["history"][-12:]

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_intake_complete(self) -> bool:
        return self.state["intake_step"] >= 5

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _resolve_mode(self) -> None:
        """
        Two-mode resolution:

        WELCOME  → default; active until all 5 intake steps are complete
        SELECT   → all 5 intake steps done, ZIP and non-MEDIGAP track known
        """
        if (self.state["intake_step"] >= 5
                and self.state["profile"]["zip"] is not None
                and self.state["profile"]["track"] is not None
                and self.state["profile"]["track"] != "MEDIGAP"):
            self.state["mode"] = "SELECT"
        else:
            self.state["mode"] = "WELCOME"
