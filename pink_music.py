#!/usr/bin/env python3
"""
pink_music.py

A modern Python reproduction of the musical core of John Simonton's
"PINK TUNES", Polyphony, July/August 1978.

The original program used Richard Voss's "pink" random-number algorithm:
five independent 2-bit values ("four-sided dice") are summed to produce
an index 0..15.  A 5-bit counter decides which dice are re-rolled.
Because only the dice whose counter bits changed are re-rolled, successive
indices usually change only a little, while occasional larger changes occur.

This version:
  * implements that pink-number generator
  * keeps the original 16-entry candidate-note concept
  * uses one pink index compositionally for four voices
  * renders the result as a standard MIDI file

It is a musical reproduction, not a cycle-for-cycle 6502 port of the
PAiA 8700 / MUS-1 / P4700-J hardware program.
"""

from __future__ import annotations

import argparse
import random
import struct
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# The Voss / "pinking" random-number generator
# ---------------------------------------------------------------------------

class PinkRandom:
    """
    Reproduce the core of Simonton's five-dice Voss algorithm.

    There are five 2-bit "dice", each holding 0..3.
    A 5-bit counter is incremented for every new value.
    Only dice corresponding to bits which changed are re-rolled.

    The sum is therefore always 0..15.
    """

    def __init__(self, seed: int | None = None, counter: int = 0):
        self.rng = random.Random(seed)
        self.dice = [self.rng.randrange(4) for _ in range(5)]
        self.counter = counter & 0x1F

    def next(self) -> int:
        old = self.counter
        self.counter = (self.counter + 1) & 0x1F
        changed = old ^ self.counter

        for bit in range(5):
            if changed & (1 << bit):
                self.dice[bit] = self.rng.randrange(4)

        return sum(self.dice)

    def sequence(self, count: int) -> list[int]:
        return [self.next() for _ in range(count)]


# ---------------------------------------------------------------------------
# Candidate notes
# ---------------------------------------------------------------------------

# This is the 16-note example printed in the article:
# C1 E1 G1 C2 E2 G2 C3 G2 A2 F2 D2 A1 F1 D1 F1 A1
#
# MIDI note numbers use C4 = 60.
ARTICLE_EXAMPLE = [
    24, 28, 31, 36, 40, 43, 48, 43,
    45, 41, 38, 33, 29, 26, 29, 33,
]


@dataclass
class Voice:
    name: str
    transpose: int
    channel: int
    velocity: int
    duration_weights: tuple[tuple[int, int], ...]
    program: int = 0


# The original program had independent timing characteristics for its
# four channels.  These are deliberately simple musical equivalents:
# (duration in eighth-notes, weight).
VOICES = [
    Voice("A", +12, 0, 92, ((1, 5), (2, 3), (4, 1))),
    Voice("B",   0, 1, 76, ((1, 6), (2, 3), (3, 1))),
    Voice("C", -12, 2, 82, ((2, 5), (4, 3), (8, 1))),
    Voice("D", +24, 3, 60, ((1, 8), (2, 2))),
]


def weighted_choice(rng: random.Random, choices):
    total = sum(weight for _, weight in choices)
    n = rng.randrange(total)
    for value, weight in choices:
        if n < weight:
            return value
        n -= weight
    return choices[-1][0]


def make_events(
    bars: int = 16,
    seed: int | None = None,
    candidate_notes: list[int] | None = None,
):
    """
    Produce four voice event lists.

    The important Pink Tunes characteristic is that all four voices use the
    same pink index at a composition event.  Thus the harmony moves through
    a locally clustered part of the 16-note candidate table instead of each
    voice independently choosing a random note.
    """
    if candidate_notes is None:
        candidate_notes = ARTICLE_EXAMPLE

    if len(candidate_notes) != 16:
        raise ValueError("The candidate table must contain exactly 16 notes.")

    rng = random.Random(seed)
    pink = PinkRandom(seed=rng.randrange(2**32))

    # 32 eighth-notes per 4/4 bar.
    total_eighths = bars * 8
    positions = [0] * len(VOICES)
    events = [[] for _ in VOICES]

    while min(positions) < total_eighths:
        idx = pink.next()
        base = candidate_notes[idx]

        for v, voice in enumerate(VOICES):
            if positions[v] >= total_eighths:
                continue

            duration = weighted_choice(rng, voice.duration_weights)
            duration = min(duration, total_eighths - positions[v])

            events[v].append(
                (positions[v], duration, base + voice.transpose, voice.velocity)
            )
            positions[v] += duration

    return events


# ---------------------------------------------------------------------------
# Minimal Standard MIDI File writer -- no external packages required.
# ---------------------------------------------------------------------------

def vlq(n: int) -> bytes:
    """Encode a MIDI variable-length quantity."""
    if n < 0:
        raise ValueError("VLQ cannot encode a negative value")

    buffer = n & 0x7F
    out = bytearray()

    while True:
        n >>= 7
        if n:
            buffer <<= 8
            buffer |= (n & 0x7F) | 0x80
        else:
            break

    while True:
        out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break

    return bytes(out)


def midi_track(events, channel: int, program: int, ticks_per_eighth: int) -> bytes:
    """
    Convert (position, duration, pitch, velocity) events into a MIDI track.
    """
    messages = []

    # Program change at time zero.
    messages.append((0, bytes([0xC0 | channel, program])))

    for position, duration, pitch, velocity in events:
        pitch = max(0, min(127, pitch))
        velocity = max(1, min(127, velocity))
        start = position * ticks_per_eighth
        end = (position + duration) * ticks_per_eighth
        messages.append((start, bytes([0x90 | channel, pitch, velocity])))
        messages.append((end, bytes([0x80 | channel, pitch, 0])))

    # Sort simultaneous note-offs before note-ons.
    messages.sort(key=lambda x: (x[0], 0 if (x[1][0] & 0xF0) == 0x80 else 1))

    data = bytearray()
    last_time = 0
    for time, msg in messages:
        data += vlq(time - last_time)
        data += msg
        last_time = time

    data += b"\x00\xFF\x2F\x00"  # End of track
    return b"MTrk" + struct.pack(">I", len(data)) + data


def write_midi(path: Path, events, bpm: int = 108):
    # 240 ticks per eighth note makes the file easy to inspect.
    ticks_per_eighth = 240
    division = ticks_per_eighth * 2  # 480 ticks per quarter note

    # Tempo track.
    micros_per_quarter = round(60_000_000 / bpm)
    tempo = (
        b"\x00\xFF\x51\x03"
        + micros_per_quarter.to_bytes(3, "big")
        + b"\x00\xFF\x58\x04\x04\x02\x18\x08"
        + b"\x00\xFF\x2F\x00"
    )
    tempo_track = b"MTrk" + struct.pack(">I", len(tempo)) + tempo

    tracks = [tempo_track]
    for voice, voice_events in zip(VOICES, events):
        tracks.append(
            midi_track(
                voice_events,
                voice.channel,
                voice.program,
                ticks_per_eighth,
            )
        )

    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), division)
    path.write_bytes(header + b"".join(tracks))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a MIDI approximation of John Simonton's Pink Tunes."
    )
    parser.add_argument("--bars", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bpm", type=int, default=108)
    parser.add_argument(
        "--print-indices",
        action="store_true",
        help="print the first 64 pink indices before writing the MIDI",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("pink_music.mid"),
    )
    args = parser.parse_args()

    if args.bars < 1:
        parser.error("--bars must be at least 1")

    if args.print_indices:
        p = PinkRandom(seed=args.seed)
        print("pink indices:", p.sequence(64))

    events = make_events(bars=args.bars, seed=args.seed)
    write_midi(args.output, events, bpm=args.bpm)

    print(f"Wrote {args.output}")
    print("Candidate notes:")
    print("  " + " ".join(f"{n:02d}" for n in ARTICLE_EXAMPLE))


if __name__ == "__main__":
    main()
