"""Spike: a deterministic Pico-on-breadboard wiring picture. USB left, chip up, pin 1 in column 60."""
from PIL import Image, ImageDraw, ImageFont

PINS = {1: "GP0", 2: "GP1", 3: "GND", 4: "GP2", 5: "GP3", 6: "GP4", 7: "GP5", 8: "GND", 9: "GP6",
        10: "GP7", 11: "GP8", 12: "GP9", 13: "GND", 14: "GP10", 15: "GP11", 16: "GP12", 17: "GP13",
        18: "GND", 19: "GP14", 20: "GP15", 21: "GP16", 22: "GP17", 23: "GND", 24: "GP18", 25: "GP19",
        26: "GP20", 27: "GP21", 28: "GND", 29: "GP22", 30: "RUN", 31: "GP26", 32: "GP27", 33: "GND",
        34: "GP28", 35: "VREF", 36: "3V3", 37: "3V3_EN", 38: "GND", 39: "VSYS", 40: "VBUS"}

W, H = 1600, 960
P = 40                      # hole pitch
COLS = list(range(60, 26, -1))  # breadboard columns, left to right (60 at the left, as on his board)
X0, Y0 = 170, 250           # first hole
ROWS = "abcde" + "fghij"
FIRST = 60                  # column pin 1 sits in

A = "/System/Library/Fonts/Supplemental/Arial.ttf"
AB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
f = lambda s, b=False: ImageFont.truetype(AB if b else A, s)


def hole(col, row):
    r = ROWS.index(row)
    y = Y0 + r * P + (P * 1 if r >= 5 else 0)      # the channel is one extra pitch here
    return X0 + COLS.index(col) * P, y


def pin_hole(n):
    """USB left, chip up: pins 1-20 along the bottom row (row h), 40-21 along the top (row c)."""
    return (FIRST - (n - 1), "h") if n <= 20 else (FIRST - (40 - n), "c")


img = Image.new("RGB", (W, H), "#101418")
d = ImageDraw.Draw(img)

# breadboard
bx0, by0 = X0 - 50, Y0 - 90
bx1, by1 = X0 + (len(COLS) - 1) * P + 50, Y0 + 10 * P + P + 60
d.rounded_rectangle((bx0, by0, bx1, by1), 18, fill="#eceae4")
d.rectangle((bx0, Y0 + 4 * P + 12, bx1, Y0 + 5 * P + P - 12), fill="#d6d3cb")  # centre channel
for c in COLS:
    x, _ = hole(c, "a")
    if c % 5 == 0:
        d.text((x, by0 + 30), str(c), font=f(20), fill="#555", anchor="mm")
        d.text((x, by1 - 25), str(c), font=f(20), fill="#555", anchor="mm")
    for r in ROWS:
        hx, hy = hole(c, r)
        d.rectangle((hx - 6, hy - 6, hx + 6, hy + 6), fill="#3a3a3a")
for r in ROWS:
    _, y = hole(COLS[0], r)
    d.text((bx0 + 22, y), r, font=f(20), fill="#555", anchor="mm")

# the Pico
lx, ty = hole(FIRST, "c")
rx, bty = hole(FIRST - 19, "h")
d.rounded_rectangle((lx - 30, ty - 28, rx + 30, bty + 28), 10, fill="#1f7a3a", outline="#155c2a", width=3)
d.rounded_rectangle((lx - 70, (ty + bty) / 2 - 38, lx - 12, (ty + bty) / 2 + 38), 6, fill="#b8bcc2")  # USB
d.text((lx - 110, (ty + bty) / 2), "USB", font=f(22, True), fill="#b8bcc2", anchor="mm")
cx = (lx + rx) / 2
d.rectangle((cx - 45, (ty + bty) / 2 - 45, cx + 45, (ty + bty) / 2 + 45), fill="#222")
d.text((cx, (ty + bty) / 2), "RP2040", font=f(16), fill="#888", anchor="mm")
d.text((cx + 250, (ty + bty) / 2), "Raspberry Pi Pico", font=f(22), fill="#cfe8d4", anchor="mm")

LIT = {20: "#ffcc00", 18: "#4db8ff"}
for n in range(1, 41):
    c, r = pin_hole(n)
    x, y = hole(c, r)
    col = LIT.get(n, "#d9b44a")
    rad = 12 if n in LIT else 8
    d.ellipse((x - rad, y - rad, x + rad, y + rad), fill=col, outline="#222", width=2)
    label = PINS[n]
    ly = y - 22 if r == "h" else y + 22
    d.text((x, ly), label, font=f(13, n in LIT), fill="#fff" if n in LIT else "#cfe8d4", anchor="mm")

# wiring: GP15 (col 41) -> resistor -> LED anode col 37, cathode col 36 -> jumper -> GND (col 43)
def dot(c, r, colr):
    x, y = hole(c, r)
    d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=colr)
    return x, y

a = dot(41, "j", "#ffcc00"); b = dot(37, "j", "#ffcc00")
d.line((a, (a[0], a[1] + 40), (b[0], b[1] + 40), b), fill="#999", width=5)
mx = (a[0] + b[0]) / 2
d.rounded_rectangle((mx - 38, a[1] + 28, mx + 38, a[1] + 52), 8, fill="#d8b98a", outline="#8a6a3a", width=2)
for i, band in enumerate(["#ff8c00", "#ff8c00", "#6b3a1a", "#c8a24a"]):
    d.rectangle((mx - 26 + i * 15, a[1] + 29, mx - 19 + i * 15, a[1] + 51), fill=band)
an = dot(37, "i", "#ff4040"); ca = dot(36, "i", "#4db8ff")
d.ellipse((an[0] - 6, an[1] - 70, ca[0] + 6, an[1] - 20), fill="#ff3030", outline="#a00", width=3)
d.line((an, (an[0], an[1] - 30)), fill="#aaa", width=4)
d.line((ca, (ca[0], ca[1] - 25)), fill="#aaa", width=4)
g = dot(43, "j", "#4db8ff"); k = dot(36, "j", "#4db8ff")
d.line((k, (k[0], k[1] + 95), (g[0], g[1] + 95), g), fill="#1a1a1a", width=7)

# legend
L = [("#ffcc00", "GP15  ·  pin 20  →  column 41, row j", "330 Ω resistor to column 37"),
     ("#ff4040", "LED long leg (+)  →  column 37, row i", "short leg (−) to column 36"),
     ("#4db8ff", "GND  ·  pin 18  →  column 43, row j", "black wire to column 36, row j")]
y = by1 + 40
for colr, main, sub in L:
    d.ellipse((X0 - 40, y + 4, X0 - 16, y + 28), fill=colr)
    d.text((X0, y), main, font=f(28, True), fill="#fff")
    d.text((X0 + 720, y + 4), sub, font=f(24), fill="#aab")
    y += 48
d.text((W / 2, 40), "USB to your left, chip up  ·  pin 1 in column 60", font=f(30, True), fill="#fff", anchor="mm")

img.save("/tmp/pico-show-example.png")
