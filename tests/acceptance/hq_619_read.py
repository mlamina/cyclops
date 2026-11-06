"""hq#619: from one strace log, every non-loopback internet socket: type, where it connects, bytes sent."""
import re
import sys
from collections import Counter

sockets = {}  # (pid, fd) -> kind; pids in one process share fds, but the network service keeps to one
kinds = {}
connects = Counter()
sends = Counter()
for line in open(sys.argv[1], errors="replace"):
    m = re.match(r"(\d+) (\w+)\((.*)", line)
    if not m:
        continue
    pid, call, rest = m.groups()
    if call == "socket":
        r = re.search(r"= (\d+)$", line.strip())
        if r and "AF_INET" in rest:
            sockets[(pid, r.group(1))] = "udp" if "SOCK_DGRAM" in rest else "tcp"
        continue
    fd = rest.split(",", 1)[0]
    key = (pid, fd)
    if call == "connect":
        a = re.search(r'htons\((\d+)\).*?(?:inet_addr|inet_pton\(AF_INET6, )\(?"([^"]+)"', rest)
        if a and not a.group(2).startswith(("127.", "::1", "0.0.0.0")):
            kind = sockets.get(key, "?")
            kinds[key] = (kind, f"{a.group(2)}:{a.group(1)}")
            status = "ok" if re.search(r"= 0$", line.strip()) else line.strip().rsplit("= ", 1)[-1][:24]
            connects[(kind, f"{a.group(2)}:{a.group(1)}", status)] += 1
    elif key in kinds and call in ("write", "sendto", "sendmsg", "sendmmsg"):
        sends[kinds[key]] += 1

print("  non-loopback connects (type, destination, result, count):")
for (kind, dest, status), n in connects.most_common() or []:
    print(f"    {kind} {dest} {status} x{n}")
if not connects:
    print("    none")
print("  sends on those sockets (type, destination, send calls):")
for (kind, dest), n in sends.most_common() or []:
    print(f"    {kind} {dest} x{n}")
if not sends:
    print("    none")
