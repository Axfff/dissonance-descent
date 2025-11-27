# Optimization Progress Tracking

## Overview

The enhanced optimizer now includes **real-time progress tracking** with intermediate saves, allowing you to:
- Monitor optimization progress during long runs
- View convergence curves in real-time
- Access intermediate results if optimization is interrupted
- Analyze the optimization trajectory

## Features

### 1. **Continuous Progress Updates**
Prints status every 5 iterations:
```
Iteration 5: cost=0.123456, elapsed=45.2s
Iteration 10: cost=0.112345, elapsed=92.8s
...
```

### 2. **Convergence Plots**
Auto-generated and updated every 5 iterations:
- **Cost vs Iteration**: Track optimization progress
- **Cost vs Time**: Monitor convergence rate
- Saved as `convergence.png`

### 3. **Detailed Checkpoints**
Saved at configurable intervals (default: every 10 iterations):
- **Frequency spectrum plots**: Bar charts showing amplitude distribution
- **ADSR envelope plots**: Visual comparison of envelope shapes (if ADSR optimization enabled)
- **Parameter JSON**: Complete partial configuration at that iteration
- Organized in timestamped checkpoint directories

### 4. **History JSON**
Complete optimization history saved as `history.json`:
```json
{
  "iterations": [1, 2, 3, ...],
  "costs": [0.15, 0.14, 0.13, ...],
  "timestamps": [1.2, 2.5, 3.9, ...]
}
```

## Configuration

In `config.json`:

```json
{
  "optimization": {
    "checkpoint_interval": 10,
    "progress_dir": "experiments/optimization_progress"
  }
}
```

Parameters:
- `checkpoint_interval`: How often to save detailed checkpoints (default: 10)
- `progress_dir`: Directory for progress files (default: `experiments/optimization_progress`)

## Output Structure

```
experiments/optimization_progress/
├── history.json                    # Complete optimization history
├── latest_partials.json           # Most recent parameter state
├── convergence.png                # Convergence plot (updated continuously)
├── checkpoint_0010/               # Checkpoint at iteration 10
│   ├── partials.json
│   ├── spectrum.png
│   └── envelopes.png              # If ADSR enabled
├── checkpoint_0020/               # Checkpoint at iteration 20
│   └── ...
└── checkpoint_0030/
    └── ...
```

## Usage

### Basic Usage
Progress tracking is **enabled by default**. Just run:
```bash
python run_experiment.py
```

### Monitor Progress During Run
Open another terminal and watch the progress:
```bash
# View convergence plot
open experiments/optimization_progress/convergence.png

# View latest checkpoint
ls -lt experiments/optimization_progress/checkpoint_*

# Tail the history
cat experiments/optimization_progress/history.json
```

### Analyze Results After Run
```python
import json
import matplotlib.pyplot as plt

# Load history
with open('experiments/optimization_progress/history.json') as f:
    history = json.load(f)

# Plot convergence
plt.plot(history['iterations'], history['costs'])
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()

# Load specific checkpoint
with open('experiments/optimization_progress/checkpoint_0050/partials.json') as f:
    partials = json.load(f)
```

## Performance Impact

Progress tracking adds minimal overhead:
- **Print every 5 iterations**: Negligible
- **Save progress every 5 iterations**: ~0.1s (JSON write + plot update)
- **Detailed checkpoint every 10 iterations**: ~0.5-1s (includes visualizations)

For a 200-iteration optimization:
- Total overhead: ~5-10 seconds
- Percentage: <1% of typical optimization time

## Tips

### For Very Long Optimizations
Increase checkpoint interval to reduce I/O:
```json
"checkpoint_interval": 20  // or 50
```

### For Quick Tests
Decrease interval to see more detail:
```json
"checkpoint_interval": 5
```

### Resume After Crash
While we don't have automatic resume yet, you can:
1. Load `latest_partials.json`
2. Use it as initial guess for new optimization
3. Set lower `maxiter` to continue from there

## Example Outputs

### Convergence Plot
Shows how dissonance decreases over time, helping you:
- Verify optimization is working
- Detect convergence
- Decide when to stop early

### Frequency Spectrum Evolution
Checkpoint spectra show how the optimizer redistributes amplitude across frequencies:
- Early iterations: Exploring the space
- Middle iterations: Concentrating energy
- Late iterations: Fine-tuning

### ADSR Evolution
If optimizing envelopes, checkpoint plots show:
- Which partials get faster/slower attacks
- How sustain levels change
- Release time optimization

## Troubleshooting

**Issue**: "Permission denied" when writing to experiments/
**Solution**: Ensure directory is writable or change `progress_dir`

**Issue**: Progress files use too much disk space
**Solution**: Increase `checkpoint_interval` or periodically delete old checkpoints

**Issue**: Convergence plot not updating
**Solution**: Check that matplotlib backend supports file writing, or use a different backend
