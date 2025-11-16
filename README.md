# Spin-Dynamics-in-Quantum-Systems

## **SUMMERY AND WORKFLOW:**

# LARMOR PRECESSION VISUALIZATION USING CONTINUOUS BEAT-BASED AUDIO

# Overview
A Python-based interactive visualization that demonstrates Larmor precession in magnetic fields with real-time audio feedback. Application visualizes quantum spin dynamics on the Bloch sphere in 3D and produces audible beat frequencies that dynamically shift with magnetic field strength.
Features

# Core Visualization
1.	3D Bloch Sphere Representation: The simulation can Interactively visualize the spin precession in a Bloch sphere using the blue rotating arrow for the spin state and the red static arrow for reference
2.	Real-Time Animation: Simulation offers 30 FPS animation that smoothly updates the spin orientation as parameters change
3.	Grid Coloring: It contains Dynamic color mapping of the spherical grid based on Larmor frequency, creating a rainbow effect from red to violet

# Audio Features
1.	Beat-Based Sound Synthesis: Generate stereo audio by multiplying two sine waves, corresponding to the Larmor frequency and reference frequency, and low-pass filtering to produce audible beats
2.	Continuous Looped Audio: Real time audio playback that updates without interruption as magnetic field (B0) changes
3.	Smart Audio Caching: Quantizes frequencies to reduce cache churn and regeneration overhead
4.	Fade In/Out Effects: 8ms ramps applied to loop transitions to reduce clicking artifacts
5.	Play/Pause Control: On-demand toggling of audio playback, with feedback provided

# Control Parameters
1.	Magnetic Field (B0): The principle slider that governs the power of the magnetic field (0.0 to 10.0 T)
Temperature: Modulates the decay of coherence by means of T2 relaxation time (0.0 to 600.0 K)
2.	Magnetic Noise: Incorporates the effects of field inhomogeneity (0.0 to 1.0).
3.	Theta and Phi Textboxes: Allow direct input of angular quantities for very precise spinning of state in spherical coordinates.

# Quantum Dynamics
1.	T2 Decoherence Modeling: Phenomenological exponential decay of spin coherence with adjustable parameters
2.	Collapse Detection: Automatically detects coherence below threshold (0.06) and schedules reset
3.	Auto-Reset: System automatically sets back to the initial state after 2-second delay once coherence collapses
4.	Temperature Effects: Magnetic noise contribution affects the computation of T2 relaxation time

# Interactive Controls
1.	Keyboard Shortcuts: The Up/Down arrows adjust B0, R key resets to defaults, P key toggles audio
2.	Reset Button: Resets all the parameters to their initial defaults
3.	Play/Pause Button: Controls continuous audio playback

# Technical Features
1.	FFT-Based Low-Pass Filtering: Frequency domain filtering to extract beat frequencies
2.	Pygame Audio Engine: Caters for efficient mixing and multi-channel audio management
3.  Threading: Background audio thread for non-blocking continuous sound generation
4.	LRU Cache: It limits cached sounds to 256 entries and automatically performs evictions(resets/refresh).
5.	Frequency Quantization: The algorithm optimizes and reduces frequency precision by 0.5 Hz to minimize generating redundant audio.


## Overview of Workflow

# Initialization Phase
1.	Simulator starts off with the default parameters: B0=0.7 T, temperature=0 K, magnetic noise=0
2.	Bloch sphere grid gets created with 36 longitude and 18 latitude lines
3.	UI controls: B0, temperature and noise sliders, theta/phi textboxes and buttons
4.	Initial spin state is set to be: theta=π/4, phi=π/3
5.	Pygame mixer is preinitialized for audio output with reserved channel allocation

# Animation Loop in Real-time
1.	Frequency Calculation: Calculates instantaneous Larmor frequency (ω = γ × B0), where γ is the gyromagnetic ratio
2.	Phase Update: Spin phase is incremented using the formula  - Δφ = ω × Δt
3.	Coherence Computation: T2 relaxation time computed from temperature and magnetic noise; coherence factor - C = exp(-t/ T2)
4.	Spin State Rotation: The blue arrow (spin state) is rotated by spin phase; red arrow fixed as reference as per the provided values of theta and phi from user.
5.	Grid Recoloring: Spherical grid lines gets dynamically coloured depending on instantenous Larmor frequency
6.	Frame Rendering: The updated 3D scene gets rendered at 30 FPS

# Thread of Continuous Audio Generation
1.	Frequency Monitoring: Background thread monitors instantaneous B0 value and calculates Larmor frequency fL
2.	Reference Frequency: Sets fr = fL - offset_hz (default offset = 2 Hz)
3.	Audio Signal Generation: generates stereo sound by creating two sine waves at frequency fL and fr respectively, multiplies them, applies FFT-based low-pass filter, and normalizes to int16 PCM
4.	Intelligent Cache Management: Checks quantized frequencies against cache; reuses existing sounds or generates new ones
5.	Looped Playback: Sound plays infinitely on reserved channel w/80ms fade transitions
6.	Play/Pause Handling: Thread respects the audioplaying flag; fades out channel on pause

# User Interaction Response
1.	Slider Updates: Values of B0, temperature, and magnetic noise
2.	Textbox Submission: Theta/phi entries sets position for red arrow; resets spin phase and start time for state reset.
3.	Reset Action: Resets all parameters to their defaults.
4.	Keyboard Input: Up/Down arrows to adjust B0; R to reset; and P to pause/play audio
5.	Collapse Handling: If coherence drops below 0.06, reset is scheduled after every 2 seconds.
