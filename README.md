# Virtual Drawing App with Hand Gestures
# Project Overview

This is a real-time virtual drawing application built using OpenCV and MediaPipe. It allows users to draw on a virtual canvas using hand gestures, without any physical pen or mouse. The app recognizes hand landmarks and pinch gestures to control drawing, color selection, and tools.

# Features
- Hand Gesture Drawing
- Draw directly using your index finger.
- Pinch gesture (index finger + thumb close) stops drawing.
- Color Palette
- Choose from 8 colors: Purple, Green, Blue, Red, Yellow, Cyan, White, and Eraser.
- Circular buttons with selection highlight.
- Brush & Eraser
- Dynamic brush size displayed near fingertip.
- Adjustable thickness using keyboard (+ / -).
- Eraser functionality included.

# Tools
- Undo / Redo (multi-level) via keyboard or on-screen buttons.
- Clear canvas (c) to start fresh.
- Save canvas (s) to save artwork as PNG.
- On-Screen UI
- Circular color buttons.
- Tool buttons for Undo, Redo, Clear, Save.
- Real-time brush indicator for better visualization.
- High Performance
- Runs fully in OpenCV with real-time drawing.
- Smooth frame rate using MediaPipe hand tracking.

# Control the drawing using your hand gestures:
- Index finger: draw on the canvas.
- Pinch (index + thumb): stop drawing.
- Move over color buttons to select a color.
- Hover over tool buttons to use Undo, Redo, Clear, Save.

# Keyboard shortcuts:
q → Quit the app
c → Clear canvas
u → Undo
r → Redo
s → Save canvas
+ → Increase brush thickness
- → Decrease brush thickness

# Contributing
Contributions are welcome! You can improve the app by:
Adding new colors or brush effects.
Improving gesture recognition.
Adding multi-hand support or shape drawing.
