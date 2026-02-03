# Visual Guide: New Routing Tab Interface

## Tab Layout Overview

```
╔════════════════════════════════════════════════════════════════╗
║                   Routing: Connect Function                     ║
║                  Outputs to Inputs                              ║
╠════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │  Output          LabelFrame                  Input      │  ║
║  │  ┌──────────────┐                    ┌──────────────┐  │  ║
║  │  │ Combobox     │                    │ Combobox     │  │  ║
║  │  │ -- (default) │                    │ -- (default) │  │  ║
║  │  │ Load Data    │                    │ Load Data    │  │  ║
║  │  │ [1] Baseline │                    │ [1] Baseline │  │  ║
║  │  │ [2] Smoothing│                    │ [2] Smoothing│  │  ║
║  │  └──────────────┘                    └──────────────┘  │  ║
║  └─────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │  Connections       LabelFrame                           │  ║
║  │  ┌───────────────────────────────────────────────────┐  │  ║
║  │  │                                                   │  │  ║
║  │  │  Outputs          [Green Lines Here]    Inputs   │  │  ║
║  │  │  ────────                               ──────   │  │  ║
║  │  │                                                   │  │  ║
║  │  │  ◄ X Data ▼                                       │  │  ║
║  │  │           │                                       │  │  ║
║  │  │           ├─────────────────────────► X_cal ►   │  │  ║
║  │  │           │                                       │  │  ║
║  │  │  ◄ Y Data ▼                                       │  │  ║
║  │  │           │                          Y_cal ►    │  │  ║
║  │  │           └─────────────────────────► Method ►  │  │  ║
║  │  │                                                   │  │  ║
║  │  │  ◄ Var Labels                         X_val ►   │  │  ║
║  │  │                                        Direction ►│  │  ║
║  │  │  ◄ Smp Labels                         Window ►   │  │  ║
║  │  │                                                   │  │  ║
║  │  │  [Blue buttons point left] [Red buttons →]      │  │  ║
║  │  │                                                   │  │  ║
║  │  │  [Scrollbar if many connections]                │  │  ║
║  │  └───────────────────────────────────────────────────┘  │  ║
║  │  ▲                                                      │  ║
║  │  └─ Scrollbar for long lists                           │  ║
║  └─────────────────────────────────────────────────────────┘  ║
║                                                                  ║
╚════════════════════════════════════════════════════════════════╝
```

## Workflow Illustration

### Phase 1: Select Functions

```
┌──────────────────────────────────────────────┐
│  Output Dropdown: [Load Data]  ▼             │
│  Input Dropdown:  [--]         ▼             │
│                                              │
│  (Canvas below empty - waiting for inputs)  │
└──────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────┐
│  Output Dropdown: [Load Data]      ▼         │
│  Input Dropdown:  [Baseline Correction] ▼   │
│                                              │
│  (Canvas ready to display buttons)          │
└──────────────────────────────────────────────┘
```

### Phase 2: Display Parameters

```
┌────────────────────────────────────────────────────────┐
│  Outputs          Connections           Inputs         │
│  ────────                               ──────         │
│                                                         │
│  Outputs                                 Inputs        │
│  ──────                                  ──────        │
│                                                         │
│  ◄ X Data (lightblue)                    X_cal ► (red) │
│  ◄ Y Data (lightblue)                    Y_cal ► (red) │
│  ◄ Var Labels (lightblue)                Method ► (red)│
│  ◄ Smp Labels (lightblue)                X_val ► (red) │
│                                                         │
│  (No buttons highlighted yet - just displayed)        │
└────────────────────────────────────────────────────────┘
```

### Phase 3: Click Output Button

```
┌────────────────────────────────────────────────────────┐
│  Outputs          Connections           Inputs         │
│                                                         │
│  ◄ X Data ← HIGHLIGHTED/SELECTED (darker color)       │
│  ◄ Y Data                                X_cal ►      │
│  ◄ Var Labels                           Method ►      │
│  ◄ Smp Labels                           X_val ►      │
│                                                         │
│  Selection state: (0, True, "X", "X Data")            │
│  Waiting for input button click...                     │
└────────────────────────────────────────────────────────┘
```

### Phase 4: Click Input Button → Create Connection

```
┌────────────────────────────────────────────────────────┐
│  Outputs          Connections           Inputs         │
│                                                         │
│  ◄ X Data ════════════════════════════► X_cal ► ✓    │
│       │        (THICK GREEN LINE)           │           │
│       └─────────── Connection Created ──────┘           │
│                                                         │
│  ◄ Y Data                                Y_cal ►      │
│  ◄ Var Labels                           Method ►      │
│  ◄ Smp Labels                           X_val ►      │
│                                                         │
│  Success Message: "Connection created: X Data → X_cal" │
│  Both buttons deselected, ready for next connection   │
└────────────────────────────────────────────────────────┘
```

### Phase 5: View Existing Connections

```
┌────────────────────────────────────────────────────────┐
│  Outputs          Connections           Inputs         │
│                                                         │
│  ◄ X Data ════════════════════════════► X_cal ►      │
│  ◄ Y Data ═════════════┐              ┌─► Y_cal ►    │
│  ◄ Var Labels          │              │               │
│  ◄ Smp Labels ─────────┼──────────────┤─► Method ►   │
│                        │              │               │
│                        └──────────────┼─► X_val ►    │
│                                       │               │
│  (Line #1: X Data → X_cal)            │               │
│  (Line #2: Y Data & Smp Labels → Y_cal & Method)     │
│                                                       │
│  Green lines persist when switching function pairs   │
└────────────────────────────────────────────────────────┘
```

### Phase 6: Remove Connection

```
Step 1: Click X Data button again
┌──────────────────────────────────────┐
│  ◄ X Data ← HIGHLIGHTED (selected)   │
│  ◄ Y Data                             │
└──────────────────────────────────────┘

Step 2: Click X_cal button again
┌──────────────────────────────────────┐
│  ◄ X Data (deselected)                │
│  ◄ Y Data                             │
│                                       │
│  X_cal ► (deselected)                 │
│  Y_cal ►                              │
│                                       │
│  Connection deleted!                  │
│  Line disappears from canvas          │
│  Message: "Connection removed"        │
└──────────────────────────────────────┘
```

## Color Scheme

| Element | Color | Meaning |
|---------|-------|---------|
| Output buttons | Light Blue (#ADD8E6) | Function outputs, point left (◄) |
| Input buttons | Light Red (#F08080) | Function inputs, point right (►) |
| Connection lines | Dark Green (#006400) | Active connection, width=3 |
| Button text | Black | Parameter name |
| Canvas background | White | Clean display area |
| Selected button | Darker shade | Currently selected for connection |

## Button Layout on Canvas

### Coordinate System
```
Canvas: 800px wide × scrollable height

Left side outputs:      Right side inputs:
x = 100              x = 700 (canvas_width - 100)

y = 50 ... 100  (start)
y = 75 ... Title "Outputs"
y = 100 ... First button
y = 140 ... Second button (spacing=40)
y = 180 ... Third button
y = 220 ... Fourth button
...
```

### Button Positioning
```
LEFT (Outputs)                      RIGHT (Inputs)
anchor="w" (west/left)              anchor="e" (east/right)

x=100 ┌─────────────────────────┐ x=700
      │◄ X Data                 │
      │◄ Y Data                 │
      │◄ Var Labels             │
      │◄ Smp Labels             │
      │                         │
      │                         │
      │                    X_cal►│
      │                    Y_cal►│
      │                   Method►│
      │                    X_val►│
      └─────────────────────────┘
```

## State Machine: Button Click Logic

```
START: selected_button = None

Event: Click output button [A]
  → State: selected_button = (func_idx, True, param_key, param_name)
  → Visual: Button [A] highlights
  
Event: Click same button [A] again
  → State: selected_button = None
  → Visual: Button [A] normal, selection cancelled
  
Event: Click different output button [B]
  → State: selected_button = (func_idx, True, param_key, param_name) [B]
  → Visual: Button [A] normal, Button [B] highlights
  
Event: Click input button [C] with output selected [A]
  → Create connection: A (output) → C (input)
  → Check: source_idx < dest_idx (validation)
  → If valid:
     → Store in routing_lines
     → Draw green line A → C
     → Show success message
  → If invalid:
     → Show error message
  → State: selected_button = None
  → Visual: Both buttons normal, line drawn or error shown

Event: Click input button [C] again if already connected to [A]
  → System detects: connection (A→C) exists
  → Delete connection
  → Remove green line
  → Show "Connection removed" message
  → State: selected_button = None
  → Visual: Line disappears
```

## Example Methodology Flow

```
Methodology Order:
[0] Load Data
[1] Create Validation Set
[2] Baseline Correction
[3] Smoothing
[4] Univariate Calibration

Possible Routing:
[0] ─────── X,Y,Labels ────────────────────────────┐
              ├──────────────────────────────────────┼─► [1] → X_cal, Y_cal
              │                                       │
              │        [2] Baseline Correction ◄─────┘
              │               ├─► X_cal, X_val
              │               │
              └───────────────┼─► [3] Smoothing
                              ├─► X_cal, X_val
                              │
                              └─► [4] Univariate Calibration
                                      ├─► Y Predicted
                                      ├─► Metrics

Visual representation in Routing Tab:
When [0] → [1] selected:
  Outputs: X, Y, var_labels, smp_labels
  Inputs:  X, Y, smp_label, createVal, method, calProportion, selection_file

When [2] → [3] selected:
  Outputs: X_cal, X_val
  Inputs:  X_cal, method, X_val, nway_flag, direction, window_size
  
Etc.
```

## Message Examples

### Success Messages (Green/Info)
```
"Connection created: X Data → X Calibration"
"Connection created: Y Data → Y Calibration"
"Connection removed"
```

### Error Messages (Red/Warning)
```
"Cannot select the same function on both sides"
"Output source must come before input destination in methodology order"
```

### Info Messages
```
When tab opens:
  - Dropdowns empty (showing "--")
  - Canvas empty (no buttons)
  
When function selected:
  - Canvas shows parameter buttons
  - Existing connections auto-drawn
  
When connection created:
  - Success message appears
  - Green line drawn
  
When connection removed:
  - Removal message appears
  - Line disappears
```

## Keyboard/Mouse Interactions

| Action | Result |
|--------|--------|
| Dropdown click | Shows list of functions, can select one |
| Button hover | Mouse cursor changes (hover effect optional) |
| Button click | Toggle selection or create/remove connection |
| Canvas click (empty space) | No effect |
| Canvas scroll | Scrolls to see more connections if many |

## Responsive Behavior

```
Canvas width detection:
  canvas.winfo_width() → actual width
  If < 2px (not yet drawn): use 800px default
  
Button placement adjusts:
  Left column: x = 100 (fixed)
  Right column: x = canvas_width - 100 (dynamic)
  
Line drawing uses button coordinates:
  Stores (x, y) for each button
  Draws line from (src_x, src_y) to (dst_x, dst_y)
```

---

This visual guide shows how the new routing tab provides an **intuitive, graphical interface** for connecting function parameters without needing to understand indices or parameter keys manually.
