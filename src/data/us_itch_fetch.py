#!/usr/bin/env python3
"""
us_itch_fetch.py - build the external United States level-one panel (Appendix F).

Reads the free Nasdaq TotalView-ITCH 5.0 sample and writes, for each target
instrument and sample day, a compact event-time top-of-book panel. The sample
days that Nasdaq serves without a key or a fee span 2019 to 2026, each one walk-
forward day-block; see DAYS for the exact list and the two file namings that
carry the same ITCH 5.0 format. The message feed is order-by-order (level three):
every resting order
carries a reference, and each execution, cancellation, and replacement cites it.
This module reconstructs only the top of book (best bid and ask price and the
size resting there) from that stream, which is all the level-one state the
transfer test consumes.

Storage discipline. The raw feed is large (about five gigabytes per day once
decompressed) and is never written to disk in full. The public endpoint throttles
each connection, so a remote day is pulled with a pool of ordered byte-range
requests, reassembled in order, and decompressed in flight; only a bounded window
of the raw stream is ever held, and only the collapsed level-one panel is kept.
Each output holds one row per change of the top of book, capped at MAX_EVENTS and
restricted to the regular session, so a day-symbol file is a few megabytes of
compressed columnar data rather than the multi-gigabyte message log. The panels
are git-ignored and are not redistributed with the package; a replicator rebuilds
them from the public sample with this script.

This step needs network access and is therefore kept out of the deterministic
reproduction chain: analysis/us_transfer.py self-skips when the panel is absent.
Provenance and the sample-day list are documented in data/README.md.

Usage
    python src/data/us_itch_fetch.py                       # all configured sample days
    python src/data/us_itch_fetch.py --days 2020-01-30     # one day
    python src/data/us_itch_fetch.py --source /path/to.gz --days 2020-01-30
    MICRODIFFUSION_US_DIR=/path python src/data/us_itch_fetch.py
"""
from __future__ import annotations
import argparse
import gzip
import os
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from sortedcontainers import SortedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import US_DIR, ensure
from data.us_panel import MAX_EVENTS, PANEL

# The Nasdaq TotalView-ITCH 5.0 sample days served free from the public endpoint,
# keyed by their calendar day. Each is one walk-forward day-block. Two file
# namings coexist on the endpoint for the same ITCH 5.0 message format: the older
# MMDDYYYY.NASDAQ_ITCH50.gz and the SMMDDYY-v50.txt.gz series; both decode with
# the same parser. Availability verified by HEAD on 2026-08-30.
BASE = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/"
DAYS = {
    "2019-01-30": BASE + "01302019.NASDAQ_ITCH50.gz",
    "2019-03-27": BASE + "03272019.NASDAQ_ITCH50.gz",
    "2019-07-30": BASE + "07302019.NASDAQ_ITCH50.gz",
    "2019-08-30": BASE + "08302019.NASDAQ_ITCH50.gz",
    "2019-10-18": BASE + "S101819-v50.txt.gz",
    "2019-10-30": BASE + "10302019.NASDAQ_ITCH50.gz",
    "2019-12-30": BASE + "12302019.NASDAQ_ITCH50.gz",
    "2020-01-30": BASE + "01302020.NASDAQ_ITCH50.gz",
    "2021-07-13": BASE + "S071321-v50.txt.gz",
    "2021-08-13": BASE + "S081321-v50.txt.gz",
    "2025-11-28": BASE + "S112825-v50.txt.gz",
    "2025-12-08": BASE + "S120825-v50.txt.gz",
    "2025-12-09": BASE + "S120925-v50.txt.gz",
    "2025-12-10": BASE + "S121025-v50.txt.gz",
    "2025-12-11": BASE + "S121125-v50.txt.gz",
    "2025-12-12": BASE + "S121225-v50.txt.gz",
    "2026-06-12": BASE + "S061226-v50.txt.gz",
}

# Corporate actions over the extended sample. The panel keys an instrument by a
# canonical symbol, but the ticker in force can change across the years the sample
# now spans; on a given day the stream carries the ticker traded that day. Meta
# Platforms (Facebook) changed its ticker from FB to META effective 2022-06-09, so
# the canonical instrument "FB" is carried as FB before that date and as META on or
# after it. traded_ticker() resolves the canonical symbol to the ticker in force on
# a day; extract() maps that back to the canonical symbol so the instrument is
# continuous across the whole sample. (On the post-change days a security also
# trades under a reassigned "FB" ticker; the mapping targets META, so that
# unrelated issuer is correctly ignored.)
TICKER_HISTORY = {
    "FB": [("2022-06-09", "META")],   # canonical -> [(effective_date, ticker), ...]
}


def traded_ticker(symbol: str, day: str) -> str:
    """Return the ticker under which a canonical panel symbol traded on `day`."""
    ticker = symbol
    for effective, renamed in sorted(TICKER_HISTORY.get(symbol, [])):
        if day >= effective:
            ticker = renamed
    return ticker

_PRICE_SCALE = 10_000.0                 # ITCH price is an integer in units of $0.0001
_NS = 1_000_000_000
OPEN_NS = int(9.5 * 3600) * _NS         # 09:30:00 ET, nanoseconds since midnight
CLOSE_NS = int(16 * 3600) * _NS         # 16:00:00 ET
_U16 = struct.Struct(">H")

# Parallel range-request transport. The public endpoint throttles each
# connection to a fraction of a megabyte per second, so a single stream of a
# multi-gigabyte day is impractical; the throttle is per-connection, so a fixed
# pool of ordered byte-range requests recovers usable throughput. Chunks are
# reassembled in order and fed to the decompressor, and only a bounded window is
# ever held, so the raw feed is never materialised in full.
_N_CONN = 16                            # concurrent range requests
_CHUNK = 16 * 1024 * 1024               # bytes per range request
_WINDOW = _N_CONN + 8                   # chunks submitted ahead of the write cursor
_RANGE_TRIES = 4                        # retries for a single range request


def _u16(b, o): return _U16.unpack_from(b, o)[0]
def _u32(b, o): return int.from_bytes(b[o:o + 4], "big")
def _u48(b, o): return int.from_bytes(b[o:o + 6], "big")
def _u64(b, o): return int.from_bytes(b[o:o + 8], "big")


def _remote_size(url: str) -> int | None:
    """Content length of a remote resource, or None if it cannot be read."""
    r = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True)
    size = None
    for line in r.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            try:
                size = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return size


def _get_range(url: str, off: int, length: int) -> bytes:
    """Fetch one byte range, retrying until the full length arrives."""
    end = off + length - 1
    data = b""
    for _ in range(_RANGE_TRIES):
        r = subprocess.run(
            ["curl", "-s", "-H", f"Range: bytes={off}-{end}", url],
            capture_output=True,
        )
        data = r.stdout
        if len(data) == length:
            return data
    raise OSError(f"range {off}-{end} returned {len(data)} of {length} bytes")


class _ParallelFetch:
    """Ordered parallel byte-range download feeding a pipe read by a decompressor.

    A pool of range requests runs ahead of a write cursor; each chunk is written
    to the pipe only once every earlier chunk has been written, so the consumer
    sees the exact byte stream of the file. At most a bounded window of chunks is
    in flight, so the raw feed is never held in full. When the consumer closes the
    read end early (all target books have filled, or the session has closed) the
    feeder stops on the resulting broken pipe.
    """

    def __init__(self, url: str, total: int):
        self.url = url
        self.total = total
        self._stop = threading.Event()
        r_fd, w_fd = os.pipe()
        self._w = os.fdopen(w_fd, "wb", buffering=0)
        self.reader = os.fdopen(r_fd, "rb")
        self._thread = threading.Thread(target=self._feed, daemon=True)
        self._thread.start()

    def _feed(self) -> None:
        nchunks = (self.total + _CHUNK - 1) // _CHUNK
        pool = ThreadPoolExecutor(max_workers=_N_CONN)
        pending: dict[int, object] = {}
        submitted = 0
        try:
            for cursor in range(nchunks):
                while (submitted < nchunks and (submitted - cursor) < _WINDOW
                       and not self._stop.is_set()):
                    off = submitted * _CHUNK
                    length = min(_CHUNK, self.total - off)
                    pending[submitted] = pool.submit(_get_range, self.url, off, length)
                    submitted += 1
                if self._stop.is_set():
                    break
                chunk = pending.pop(cursor).result()
                if self._stop.is_set():
                    break
                self._w.write(chunk)
        except (BrokenPipeError, OSError, ValueError, RuntimeError):
            pass                                 # consumer closed the pipe early
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            try:
                self._w.close()
            except OSError:
                pass

    def terminate(self) -> None:
        self._stop.set()
        try:
            self.reader.close()
        except OSError:
            pass
        self._thread.join(timeout=60)


def _open_itch(source: str):
    """Open an ITCH gzip stream from a local file OR a URL.

    A local path is opened directly. A URL is pulled with a pool of ordered
    byte-range requests (the public endpoint throttles each connection, so a
    single stream of a multi-gigabyte day is impractical), reassembled in order
    and decompressed in flight, so the raw file never touches the disk in full and
    only the per-symbol level-one panel is written. curl is used for transport
    because it succeeds where urllib is at times refused. If the resource size is
    unavailable the transport falls back to a single stream.
    """
    if source.startswith(("http://", "https://")):
        total = _remote_size(source)
        if total:
            fetch = _ParallelFetch(source, total)
            return gzip.GzipFile(fileobj=fetch.reader), fetch
        proc = subprocess.Popen(["curl", "-sL", source], stdout=subprocess.PIPE)
        return gzip.GzipFile(fileobj=proc.stdout), proc
    return gzip.open(source, "rb"), None


class _Book:
    """Minimal top-of-book for one instrument: aggregate size at each price.

    Only the best bid and ask are ever read, but the whole resting book must be
    tracked because a cancellation at the touch exposes the next price. Bids and
    asks are kept as price-keyed sorted maps of aggregate size, so the best level
    is the last (highest bid) or first (lowest ask) key.
    """
    __slots__ = ("bids", "asks")

    def __init__(self):
        self.bids = SortedDict()
        self.asks = SortedDict()

    def add(self, side, price, shares):
        book = self.bids if side == "B" else self.asks
        book[price] = book.get(price, 0) + shares

    def remove(self, side, price, shares):
        book = self.bids if side == "B" else self.asks
        left = book.get(price, 0) - shares
        if left > 0:
            book[price] = left
        else:
            book.pop(price, None)

    def top(self):
        """Return (bid, bid_sz, ask, ask_sz) in integer price, or None if a side is empty."""
        if not self.bids or not self.asks:
            return None
        bp, bsz = self.bids.peekitem(-1)
        ap, asz = self.asks.peekitem(0)
        return bp, bsz, ap, asz


def extract(source: str, day: str, symbols, out_dir: Path,
            cap: int = MAX_EVENTS) -> dict[str, Path]:
    """Stream one ITCH day, write one level-one parquet per target symbol.

    Reconstructs the top of book for every target instrument and records a row
    each time its best bid or ask (price or size) changes within the regular
    session, up to cap rows per instrument.
    """
    # Map the ticker traded on this day (8 bytes) back to the canonical panel
    # symbol, so a mid-sample ticker change (e.g. FB -> META) stays one instrument.
    targets = {traded_ticker(s, day).encode().ljust(8): s for s in symbols}
    locate_of: dict[int, str] = {}                          # locate -> target symbol
    order_sym: dict[int, str] = {}                          # order ref -> symbol
    order_ref: dict[int, tuple] = {}                        # order ref -> (side, price, shares)
    books: dict[str, _Book] = {s: _Book() for s in symbols}
    rows: dict[str, list] = {s: [] for s in symbols}
    last: dict[str, tuple] = {s: None for s in symbols}
    capped: set[str] = set()

    def touch(sym, ts):
        if sym in capped:
            return
        t = books[sym].top()
        if t is None:
            return
        bp, bsz, ap, asz = t
        if ap <= bp:                    # crossed book is a sequencing artefact
            return
        key = (bp, bsz, ap, asz)
        if key == last[sym]:
            return
        last[sym] = key
        rows[sym].append((ts, bp / _PRICE_SCALE, bsz, ap / _PRICE_SCALE, asz))
        if len(rows[sym]) >= cap:
            capped.add(sym)

    stream, proc = _open_itch(source)
    try:
        read = stream.read
        while len(capped) < len(symbols):
            head = read(2)
            if len(head) < 2:
                break
            length = _U16.unpack(head)[0]
            msg = read(length)
            if len(msg) < length:
                break
            typ = msg[0:1]
            body = msg[1:]

            # Every ITCH 5.0 message shares the header stock_locate (2), tracking
            # number (2), timestamp (6), so the event clock is readable without
            # decoding the body. The stream is globally time-ordered, so once it
            # passes the regular-session close no target row can be added again and
            # the remaining post-close bytes need not be read.
            ts = _u48(body, 4)
            if ts > CLOSE_NS:
                break

            if typ == b"R":                     # stock directory: locate -> symbol
                stock = body[10:18]
                if stock in targets:
                    locate_of[_u16(body, 0)] = targets[stock]
                continue

            sym = locate_of.get(_u16(body, 0))
            if sym is None:
                continue

            if typ == b"A" or typ == b"F":
                ref = _u64(body, 10)
                side = chr(body[18])
                shares = _u32(body, 19)
                price = _u32(body, 31)
                order_ref[ref] = (side, price, shares)
                order_sym[ref] = sym
                books[sym].add(side, price, shares)
                if OPEN_NS <= ts <= CLOSE_NS:
                    touch(sym, ts)
            elif typ == b"E" or typ == b"C":
                ref = _u64(body, 10)
                ex = _u32(body, 18)
                o = order_ref.get(ref)
                if o is not None:
                    side, price, shares = o
                    shares -= ex
                    books[sym].remove(side, price, ex)
                    if shares <= 0:
                        order_ref.pop(ref, None)
                        order_sym.pop(ref, None)
                    else:
                        order_ref[ref] = (side, price, shares)
                    if OPEN_NS <= ts <= CLOSE_NS:
                        touch(sym, ts)
            elif typ == b"X":
                ref = _u64(body, 10)
                canceled = _u32(body, 18)
                o = order_ref.get(ref)
                if o is not None:
                    side, price, shares = o
                    shares -= canceled
                    books[sym].remove(side, price, canceled)
                    if shares <= 0:
                        order_ref.pop(ref, None)
                        order_sym.pop(ref, None)
                    else:
                        order_ref[ref] = (side, price, shares)
                    if OPEN_NS <= ts <= CLOSE_NS:
                        touch(sym, ts)
            elif typ == b"D":
                ref = _u64(body, 10)
                o = order_ref.pop(ref, None)
                order_sym.pop(ref, None)
                if o is not None:
                    side, price, shares = o
                    books[sym].remove(side, price, shares)
                    if OPEN_NS <= ts <= CLOSE_NS:
                        touch(sym, ts)
            elif typ == b"U":
                orig = _u64(body, 10)
                new_ref = _u64(body, 18)
                shares = _u32(body, 26)
                price = _u32(body, 30)
                o = order_ref.pop(orig, None)
                order_sym.pop(orig, None)
                if o is None:
                    continue                  # U carries no side; the old reference is required
                old_side, old_price, old_shares = o
                books[sym].remove(old_side, old_price, old_shares)
                side = old_side
                order_ref[new_ref] = (side, price, shares)
                order_sym[new_ref] = sym
                books[sym].add(side, price, shares)
                if OPEN_NS <= ts <= CLOSE_NS:
                    touch(sym, ts)
    except (EOFError, OSError):
        pass                                     # truncated or interrupted stream
    finally:
        stream.close()
        if proc is not None:
            proc.terminate()

    ensure(out_dir)
    paths: dict[str, Path] = {}
    for sym in symbols:
        if not rows[sym]:
            continue
        df = pd.DataFrame(rows[sym], columns=["ts", "bid", "bid_sz", "ask", "ask_sz"])
        p = out_dir / f"{sym}_{day}.parquet"
        df.to_parquet(p, index=False)
        paths[sym] = p
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", default=",".join(DAYS),
                    help="comma-separated sample days (YYYY-MM-DD)")
    ap.add_argument("--symbols", default=",".join(PANEL),
                    help="comma-separated Nasdaq symbols")
    ap.add_argument("--source", default=None,
                    help="local ITCH .gz for a single --days value (else the day is streamed)")
    ap.add_argument("--out", default=str(US_DIR), help="destination directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    days = [d.strip() for d in args.days.split(",") if d.strip()]
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"{len(days)} day(s) x {len(syms)} symbol(s) -> {out_dir}")
    for day in days:
        source = args.source if args.source else DAYS.get(day)
        if source is None:
            print(f"  [{day}] no source URL known; skipping", flush=True)
            continue
        origin = "local" if not source.startswith("http") else "stream"
        print(f"  [{day}] {origin} {source}", flush=True)
        paths = extract(source, day, syms, out_dir)
        for sym in syms:
            if sym in paths:
                n = len(pd.read_parquet(paths[sym]))
                print(f"      {sym}: {n} top-of-book rows -> {paths[sym].name}", flush=True)
            else:
                print(f"      {sym}: no rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
