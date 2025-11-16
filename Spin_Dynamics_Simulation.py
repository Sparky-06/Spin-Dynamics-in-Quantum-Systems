"""
Larmor Precession Visualization with beat-based continuous audio
- Produces audio by generating two sine waves at fL (Larmor) and fr (reference),
  multiplying them, low-pass filtering to keep the beat (fL-fr), normalizing,
  and playing the result looped. Audio updates as B0 changes.
- Rest of the UI: blue rotating arrow, red static arrow, decoherence/T2, grid coloring
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider, Button, TextBox
import pygame
import threading
import time
from collections import OrderedDict
import math

# -----------------------------
# Audio configuration & helpers
# -----------------------------
SAMPLE_RATE = 44100
SOUND_BUFFER_MS = 300        # buffer duration in ms for each loop chunk
SOUND_DURATION = SOUND_BUFFER_MS / 1000.0
_CACHE_MAX = 256
_FREQ_QUANT = 0.5            # quantize frequencies slightly to reduce churn
RAMP_MS = 8

_mixer_available = True
try:
    pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
    pygame.mixer.set_num_channels(max(pygame.mixer.get_num_channels(), 4))
except Exception as e:
    print("Warning: pygame.mixer.init() failed:", e)
    _mixer_available = False

_sound_cache = OrderedDict()
_cache_lock = threading.Lock()
_reserved_channel = None
if _mixer_available:
    try:
        _reserved_channel = pygame.mixer.Channel(0)
    except Exception:
        _reserved_channel = None

def _lowpass_fft(signal, sr, cutoff_hz):
    """
    Simple FFT low-pass filter: zero frequency bins > cutoff_hz and inverse FFT.
    Signal is real 1D array.
    """
    N = len(signal)
    # FFT
    S = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=1.0/sr)
    # zero above cutoff
    S[freqs > cutoff_hz] = 0
    out = np.fft.irfft(S, n=N)
    return out

def _make_beat_sound(fL, fr, duration=SOUND_DURATION, volume=0.28):
    """
    Generate a stereo pygame Sound that is the low-passed product of two sines
    at fL and fr. Returns pygame.mixer.Sound or None.
    """
    if not _mixer_available:
        return None

    n_samples = int(SAMPLE_RATE * duration)
    if n_samples <= 2:
        return None

    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Generate the two sines
    s1 = np.sin(2.0 * np.pi * fL * t)
    s2 = np.sin(2.0 * np.pi * fr * t)

    # Multiply -> contains cos((fL-fr)t) and cos((fL+fr)t) terms
    prod = s1 * s2

    # Low-pass cutoff: keep the difference (beat). Choose cutoff slightly above |fL-fr|
    beat_freq = abs(fL - fr)
    # guard values
    if beat_freq < 0.5:
        cutoff = max(30.0, 4.0 * beat_freq + 30.0)  # ensure some audible content
    else:
        cutoff = max(beat_freq * 3.0 + 10.0, 60.0)

    # Apply low-pass to remove the high (sum) component ~ fL+fr
    filtered = _lowpass_fft(prod, SAMPLE_RATE, cutoff_hz=cutoff)

    # Optionally apply a short fade-in/out to reduce loop clicks
    ramp_samps = int(SAMPLE_RATE * (RAMP_MS / 1000.0))
    if ramp_samps * 2 < n_samples:
        ramp = np.linspace(0.0, 1.0, ramp_samps)
        filtered[:ramp_samps] *= ramp
        filtered[-ramp_samps:] *= ramp[::-1]

    # Normalize to int16 range with requested volume
    max_val = np.max(np.abs(filtered)) + 1e-12
    pcm = np.int16(np.clip(filtered / max_val * (32767.0 * volume), -32767, 32767))

    stereo = np.column_stack((pcm, pcm))
    stereo = np.ascontiguousarray(stereo)
    try:
        snd = pygame.sndarray.make_sound(stereo)
        return snd
    except Exception as e:
        print("Warning: couldn't create pygame Sound:", e)
        return None

def get_cached_beat(fL, fr, duration=SOUND_DURATION, volume=0.28):
    """
    Cache beat sounds by quantized fL and fr to avoid regenerating too often.
    Key quantizes both frequencies to _FREQ_QUANT.
    """
    if not _mixer_available:
        return None
    qfL = round(fL / _FREQ_QUANT) * _FREQ_QUANT
    qfr = round(fr / _FREQ_QUANT) * _FREQ_QUANT
    key = (qfL, qfr, round(duration,3), float(volume))
    with _cache_lock:
        snd = _sound_cache.get(key)
        if snd is not None:
            _sound_cache.move_to_end(key)
            return snd
        snd = _make_beat_sound(qfL, qfr, duration=duration, volume=volume)
        if snd is None:
            return None
        _sound_cache[key] = snd
        _sound_cache.move_to_end(key)
        if len(_sound_cache) > _CACHE_MAX:
            _sound_cache.popitem(last=False)
        return snd

def play_looped_sound_obj(snd, fade_ms=80):
    """Play a pygame Sound object looped (-1) on reserved channel."""
    if not _mixer_available or _reserved_channel is None or snd is None:
        return
    try:
        if _reserved_channel.get_busy():
            _reserved_channel.fadeout(int(fade_ms * 0.6))
        _reserved_channel.play(snd, loops=-1, fade_ms=fade_ms)
    except Exception:
        pass

# -----------------------------
# Simulation parameters
# -----------------------------
gamma_effective = 1.0  # rad/s per unit B (user scale)
t_max = 6.0
fps = 30
n_frames = int(t_max * fps)

# initial UI state
theta0 = np.pi / 4
phi0 = np.pi / 3
B0 = 0.7

# audio reference frequency fr (close to fL)
# We'll pick fr slightly offset from fL by a small reference delta (can be tuned)
ref_offset_hz = 2.0   # default offset in Hz (fr = fL + ref_offset_hz)

# temperature & magnetic noise sliders
temperature = 0.0
mag_noise = 0.0

# phenomenological constants for T2 model (tweakable)
alpha = 0.01   # 1/(K*s)
beta = 0.8     # 1/(B^2 * s)

# -----------------------------
# Phase integration state
# -----------------------------
spin_phase = 0.0
last_time = time.time()
start_time = last_time

# collapse/reset handling
collapse_threshold = 0.06   # coherence below which we consider collapsed
reset_delay_after_collapse = 2.0  # seconds after collapse to reset
_reset_scheduled_time = None  # timestamp at which a reset will occur

# -----------------------------
# Math helpers
# -----------------------------
def sph_to_cart(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def compute_T2(T, sigmaB):
    if (abs(T) < 1e-12) and (abs(sigmaB) < 1e-12):
        return math.inf
    denom = alpha * max(T, 0.0) + beta * (sigmaB ** 2)
    if denom <= 0:
        return math.inf
    return 1.0 / denom

def larmor_hz_from_B(B):
    """Return Larmor frequency in Hz from B using gamma_effective (rad/s per unit B)."""
    omega = abs(gamma_effective * B)  # rad/s
    return omega / (2.0 * np.pi)

def map_larmor_to_audible(omega):
    base_hz = abs(omega) / (2.0 * np.pi)
    return float(np.clip(base_hz, 0.1, 20000.0))  # allow <30 to keep beat math ok

# -----------------------------
# Matplotlib figure & widgets
# -----------------------------
plt.style.use('default')
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])
ax.axis('off')

# draw grid lines as individual Line3D objects so we can recolor them in animation
Nu = 36   # number of longitude lines
Nv = 18   # number of latitude lines
_u = np.linspace(0, 2*np.pi, Nu+1)
_v = np.linspace(0, np.pi, Nv)

grid_lines = []
line_width = 0.6
base_alpha = 0.28
for i, ui in enumerate(_u):
    xs = np.cos(ui) * np.sin(_v)
    ys = np.sin(ui) * np.sin(_v)
    zs = np.cos(_v)
    l, = ax.plot(xs, ys, zs, lw=line_width, color=(0.6,0.6,0.6,base_alpha))
    grid_lines.append(l)
for j, vj in enumerate(_v):
    xs = np.cos(_u) * np.sin(vj)
    ys = np.sin(_u) * np.sin(vj)
    zs = np.full_like(_u, np.cos(vj))
    l, = ax.plot(xs, ys, zs, lw=line_width, color=(0.6,0.6,0.6,base_alpha))
    grid_lines.append(l)

# axes labels
L = 1.2
ax.plot([-L, L], [0,0], [0,0], color='black', lw=1)
ax.plot([0,0], [-L, L], [0,0], color='black', lw=1)
ax.plot([0,0], [0,0], [-L, L], color='black', lw=1)
ax.text(L, 0, 0, 'x', fontsize=12)
ax.text(0, L, 0, 'y', fontsize=12)
ax.text(0, 0, L, 'z', fontsize=12)
ax.view_init(elev=30, azim=30)

# initial arrows
blue_vec0 = sph_to_cart(theta0, phi0)
red_vec0 = sph_to_cart(theta0, phi0)
blue_quiver = ax.quiver(0,0,0, blue_vec0[0], blue_vec0[1], blue_vec0[2], color='blue', linewidth=2)
red_quiver  = ax.quiver(0,0,0, red_vec0[0],  red_vec0[1],  red_vec0[2],  color='red', linewidth=1.5)


# -----------------------------
# UI layout (new, non-overlapping)
# -----------------------------
plt.subplots_adjust(left=0.05, bottom=0.25, right=0.95, top=0.95)

# Axes for controls (x0, y0, width, height)
ax_B = plt.axes([0.15, 0.23, 0.70, 0.035])         # 🔴 main B0 slider (longest)
ax_temp = plt.axes([0.45, 0.18, 0.33, 0.03])       # 🔴 smaller temperature slider
ax_noise = plt.axes([0.45, 0.13, 0.33, 0.03])      # 🔴 smaller magnetic noise slider

# 🔵 text boxes for theta and phi (below sliders)
ax_theta = plt.axes([0.25, 0.07, 0.18, 0.05])
ax_phi   = plt.axes([0.55, 0.07, 0.18, 0.05])

# 🟨 audio control buttons together on the left
ax_play  = plt.axes([0.17, 0.15, 0.13, 0.07])
ax_reset = plt.axes([0.80, 0.06, 0.10, 0.07])      # 🟣 reset button bottom-right

# Create controls
slider_B = Slider(ax_B, 'B₀', 0.0, 10.0, valinit=B0, valstep=0.01)
slider_temp = Slider(ax_temp, 'Temperature (K)', 0.0, 600.0, valinit=temperature, valstep=1.0)
slider_noise = Slider(ax_noise, 'Mag Noise', 0.0, 1.0, valinit=mag_noise, valstep=0.01)

text_theta = TextBox(ax_theta, 'θ (rad)', initial=f"{theta0:.3f}")
text_phi   = TextBox(ax_phi, 'φ (rad)', initial=f"{phi0:.3f}")

button_play  = Button(ax_play, 'Pause', color='yellow', hovercolor='lightgreen')
button_reset = Button(ax_reset, 'Reset', color='lightgray', hovercolor='lightblue')

# audio play state
audio_playing = True

# -----------------------------
# Animation update
# -----------------------------
last_quiver = None
cmap = plt.cm.plasma
vmax_color = 200.0

def larmor_color_base(larmor_hz):
    frac = np.clip(larmor_hz / vmax_color, 0.0, 1.0)
    return frac


# -----------------------------
# Bloch sphere colour change
# -----------------------------------------------------------------------------------------------------------------------------------

import matplotlib.cm as cm

def color_from_B(B, B_min=0.0, B_max=10.0):
    """
    Map B field to rainbow colors: red → green → violet.
    """
    t = np.clip((B - B_min) / (B_max - B_min), 0.0, 1.0)
    cmap = cm.get_cmap('rainbow')
    rgba = cmap(t)  # returns (r,g,b,a)
    return (rgba[0], rgba[1], rgba[2], 0.5)

#-------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------



def update(frame):
    global last_quiver, spin_phase, last_time, _reset_scheduled_time, start_time

    now = time.time()
    dt = now - last_time if last_time is not None else 0.0
    last_time = now

    omega_now = -gamma_effective * B0
    spin_phase += omega_now * dt

    elapsed = now - start_time
    T2 = compute_T2(temperature, mag_noise)
    if math.isinf(T2):
        C = 1.0
    else:
        C = math.exp(-elapsed / T2)

    # detect collapse and schedule reset if needed
    if (not math.isinf(T2)) and (temperature > 1e-12 or mag_noise > 1e-12):
        if C < collapse_threshold:
            if _reset_scheduled_time is None:
                _reset_scheduled_time = now + reset_delay_after_collapse
        else:
            _reset_scheduled_time = None
    else:
        _reset_scheduled_time = None

    # perform scheduled reset
    if _reset_scheduled_time is not None and now >= _reset_scheduled_time:
        spin_phase = 0.0
        start_time = time.time()
        last_time = start_time
        _reset_scheduled_time = None

    # compute arrow vectors
    ca, sa = np.cos(spin_phase), np.sin(spin_phase)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0,0,1]])
    blue_dir_unit = Rz @ sph_to_cart(theta0, phi0)
    blue_vec = blue_dir_unit * C
    red_vec = sph_to_cart(theta0, phi0)

    try:
        blue_quiver.remove()
    except Exception:
        pass
    try:
        red_quiver.remove()
    except Exception:
        pass

    # recolor grid lines
    larmor_hz = abs(omega_now) / (2.0 * np.pi)
    base_frac = larmor_color_base(larmor_hz)
    Nu_local = Nu
    Nv_local = Nv
    for idx, line in enumerate(grid_lines):
        offset = (idx % (Nu_local if idx < Nu_local else Nv_local)) / float(max(1, (Nu_local if idx < Nu_local else Nv_local)))
        #frac = np.clip(base_frac * 0.6 + offset * 0.4, 0.0, 1.0)
        #rgba = cmap(frac)
        #alpha_dynamic = base_alpha * (0.6 + 0.4 * (0.5 + 0.5 * math.sin(idx)))
        #line.set_color((rgba[0], rgba[1], rgba[2], alpha_dynamic))

    # recolor grid lines based on current B0 (new)
    grid_color = color_from_B(B0)
    for line in grid_lines:
        line.set_color(grid_color)


    

    color_blue = cmap(np.clip(base_frac, 0.0, 1.0))

    # draw updated quivers
    blue_q = ax.quiver(0,0,0, blue_vec[0], blue_vec[1], blue_vec[2], color=color_blue, linewidth=2)
    red_q  = ax.quiver(0,0,0, red_vec[0],  red_vec[1],  red_vec[2],  color='red', linewidth=1.5)

    globals()['blue_quiver'] = blue_q
    globals()['red_quiver'] = red_q

    ax.set_title(f"B0={B0:.3f}   Larmor={larmor_hz:.2f} Hz   T2={'inf' if math.isinf(T2) else f'{T2:.3e}'} s   Coherence={C:.3f}")
    return []

ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False, repeat=True)

# -----------------------------
# UI callbacks
# -----------------------------
def slider_B_changed(val):
    global B0
    B0 = float(val)

def slider_temp_changed(val):
    global temperature
    temperature = float(val)

def slider_noise_changed(val):
    global mag_noise
    mag_noise = float(val)

def theta_submit(txt):
    global theta0, spin_phase, start_time, last_time
    try:
        theta_new = float(txt)
        theta_new = float(np.clip(theta_new, 0.0, np.pi))
        theta0 = theta_new
        spin_phase = 0.0
        start_time = time.time()
        last_time = start_time
    except Exception:
        pass

def phi_submit(txt):
    global phi0, spin_phase, start_time, last_time
    try:
        phi_new = float(txt)
        phi_new = float(phi_new % (2*np.pi))
        phi0 = phi_new
        spin_phase = 0.0
        start_time = time.time()
        last_time = start_time
    except Exception:
        pass

def reset(event=None):
    global B0, temperature, mag_noise, theta0, phi0, spin_phase, start_time, last_time, audio_playing
    B0 = 0.0
    temperature = 0.0
    mag_noise = 0.0
    theta0 = np.pi / 4
    phi0 = np.pi / 3
    spin_phase = 0.0
    start_time = time.time()
    last_time = start_time
    slider_B.set_val(B0)
    slider_temp.set_val(temperature)
    slider_noise.set_val(mag_noise)
    text_theta.set_val(f"{theta0:.3f}")
    text_phi.set_val(f"{phi0:.3f}")

slider_B.on_changed(slider_B_changed)
slider_temp.on_changed(slider_temp_changed)
slider_noise.on_changed(slider_noise_changed)
text_theta.on_submit(theta_submit)
text_phi.on_submit(phi_submit)
button_reset.on_clicked(reset)

# play/pause for audio
def toggle_audio(event):
    global audio_playing
    audio_playing = not audio_playing
    if audio_playing:
        button_play.label.set_text('Pause')
    else:
        button_play.label.set_text('Play')
        if _mixer_available and _reserved_channel is not None:
            try:
                _reserved_channel.fadeout(80)
            except Exception:
                pass

button_play.on_clicked(toggle_audio)

# keyboard bindings
def on_key(event):
    global B0
    if event.key in ('+','=' ):
        B0 = min(B0 + 0.1, 50.0)
        slider_B.set_val(B0)
    elif event.key == '-':
        B0 = max(B0 - 0.1, 0.0)
        slider_B.set_val(B0)
    elif event.key == 'r':
        reset()

fig.canvas.mpl_connect('key_press_event', on_key)

# -----------------------------
# Continuous audio thread (beat-based looped playback)
# -----------------------------
_audio_thread_running = False
_last_audio_pair = (None, None)

def continuous_audio_thread(poll_interval=0.08):
    """
    Runs in background:
    - computes instantaneous fL (from B0)
    - sets reference fr = fL + ref_offset_hz
    - generates cached beat sound for (fL, fr) and plays looped
    - updates when significant change in fL/fr
    - honors audio_playing flag
    """
    global _audio_thread_running, _last_audio_pair
    _audio_thread_running = True
    _last_audio_pair = (None, None)
    while _audio_thread_running:
        if audio_playing and _mixer_available and _reserved_channel is not None:
            # instantaneous Larmor frequency (Hz)
            fL = larmor_hz_from_B(B0)
            # choose reference frequency close to fL
            fr = fL + ref_offset_hz
            # quantized check to avoid constant regeneration
            qfL = round(fL / _FREQ_QUANT) * _FREQ_QUANT
            qfr = round(fr / _FREQ_QUANT) * _FREQ_QUANT
            if (_last_audio_pair[0] != qfL) or (_last_audio_pair[1] != qfr):
                snd = get_cached_beat(fL, fr, duration=SOUND_DURATION, volume=0.30)
                if snd is not None:
                    # play looped sound (smooth fade)
                    play_looped_sound_obj(snd, fade_ms=80)
                    _last_audio_pair = (qfL, qfr)
        else:
            # pause or mixer not available -> fade out channel
            if _mixer_available and _reserved_channel is not None and _reserved_channel.get_busy():
                try:
                    _reserved_channel.fadeout(60)
                except Exception:
                    pass
            _last_audio_pair = (None, None)
        time.sleep(poll_interval)

# start audio thread
if _mixer_available:
    t_audio = threading.Thread(target=continuous_audio_thread, daemon=True)
    t_audio.start()
else:
    print("Audio disabled: pygame mixer not available.")

# -----------------------------
# Shutdown handler
# -----------------------------
def on_close(event):
    global _audio_thread_running
    _audio_thread_running = False
    if _mixer_available:
        try:
            pygame.mixer.stop()
            pygame.mixer.quit()
        except Exception:
            pass

fig.canvas.mpl_connect('close_event', on_close)

# -----------------------------
# Start
# -----------------------------
print("Larmor Precession (beat-based continuous audio) — ready")
print("Controls: B0, Temperature, Mag Noise. Type θ/φ to set red arrow. Reset to defaults. Use Play/Pause button for audio.")
plt.show()
