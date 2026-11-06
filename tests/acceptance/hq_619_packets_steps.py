#!/usr/bin/env python3
"""hq#619 round 3: hq_619_packets.py, plus a 10 s time step on every count. Headers only.

Usage (root): hq_619_packets.py <iface> <phase file> <seconds> <output file>
Output: <step start, s from counter start> <phase> <way> <proto> <peer> <port> x<count>
"""
import ipaddress
import socket
import struct
import sys
import time
from collections import Counter

iface, phase_file, seconds, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))
s.bind((iface, 0))
s.settimeout(1.0)
OUTGOING = 4
counts = Counter()
changes = []
start = time.time()
phase, read_at = "start", 0.0


def scope(addr):
    return "internet" if ipaddress.ip_address(addr).is_global else "local"


while time.time() - start < seconds:
    if time.time() - read_at > 0.5:
        try:
            new = open(phase_file).read().strip() or phase
            if new != phase:
                changes.append((round(time.time() - start, 1), new))
            phase = new
        except OSError:
            pass
        read_at = time.time()
    try:
        data, meta = s.recvfrom(65535)
    except socket.timeout:
        continue
    kind = struct.unpack("!H", data[12:14])[0]
    if kind == 0x0800:
        ihl = (data[14] & 0x0F) * 4
        proto = data[23]
        src, dst = socket.inet_ntoa(data[26:30]), socket.inet_ntoa(data[30:34])
        l4 = 14 + ihl
    elif kind == 0x86DD:
        proto = data[20]
        src = socket.inet_ntop(socket.AF_INET6, data[22:38])
        dst = socket.inet_ntop(socket.AF_INET6, data[38:54])
        l4 = 54
    else:
        continue
    way = "out" if meta[2] == OUTGOING else "in"
    peer = dst if way == "out" else src
    sport, dport = struct.unpack("!HH", data[l4:l4 + 4]) if proto in (6, 17) else (0, 0)
    port = dport if way == "out" else sport
    name = {6: "tcp", 17: "udp", 1: "icmp", 58: "icmp6"}.get(proto, str(proto))
    if scope(peer) == "internet":
        step = int((time.time() - start) // 10) * 10
        counts[(step, phase, way, name, peer, port)] += 1

with open(out, "w") as f:
    f.write(f"counter started at epoch {start:.1f}\n")
    for at, name in changes:
        f.write(f"PHASE +{at}s {name}\n")
    for key, n in sorted(counts.items()):
        f.write(" ".join(str(k) for k in key) + f" x{n}\n")
    f.write("END\n")
