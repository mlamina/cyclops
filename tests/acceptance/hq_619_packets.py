#!/usr/bin/env python3
"""hq#619: count packets on one interface by phase. Headers only, no contents are kept or printed.

Usage (root): capture.py <iface> <phase file> <seconds> <output file>
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
start = time.time()
phase, read_at = "start", 0.0


def scope(addr):
    return "internet" if ipaddress.ip_address(addr).is_global else "local"


while time.time() - start < seconds:
    if time.time() - read_at > 0.5:
        try:
            phase = open(phase_file).read().strip() or phase
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
        counts[(phase, way, name, peer, port)] += 1
    else:
        counts[(phase, "local", way, "", 0)] += 1

with open(out, "w") as f:
    for key, n in sorted(counts.items(), key=lambda kv: (kv[0][0], -kv[1])):
        f.write(" ".join(str(k) for k in key if k != "") + f" x{n}\n")
    f.write("END\n")
