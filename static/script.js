document.addEventListener('DOMContentLoaded', () => {
    const partialsList = document.getElementById('partials-list');
    const addPartialBtn = document.getElementById('add-partial-btn');
    const updateBtn = document.getElementById('update-btn');
    const ctx = document.getElementById('dissonance-chart').getContext('2d');

    let chart = null;

    // Default partials (Sawtooth-like)
    let partials = [
        { ratio: 1.0, amplitude: 1.0 },
        { ratio: 2.0, amplitude: 0.5 },
        { ratio: 3.0, amplitude: 0.33 },
        { ratio: 4.0, amplitude: 0.25 },
        { ratio: 5.0, amplitude: 0.2 }
    ];

    function renderPartials() {
        partialsList.innerHTML = '';
        partials.forEach((p, index) => {
            const row = document.createElement('div');
            row.className = 'partial-row';
            row.innerHTML = `
                <label>Ratio:</label>
                <input type="number" step="0.01" value="${p.ratio}" onchange="updatePartial(${index}, 'ratio', this.value)">
                <label>Amp:</label>
                <input type="number" step="0.01" value="${p.amplitude}" onchange="updatePartial(${index}, 'amplitude', this.value)">
                <button class="btn remove" onclick="removePartial(${index})">×</button>
            `;
            partialsList.appendChild(row);
        });
    }

    window.updatePartial = (index, key, value) => {
        partials[index][key] = parseFloat(value);
    };

    window.removePartial = (index) => {
        partials.splice(index, 1);
        renderPartials();
    };

    addPartialBtn.addEventListener('click', () => {
        const lastRatio = partials.length > 0 ? partials[partials.length - 1].ratio : 0;
        partials.push({ ratio: Math.floor(lastRatio) + 1.0, amplitude: 0.1 });
        renderPartials();
    });

    updateBtn.addEventListener('click', fetchData);

    async function fetchData() {
        updateBtn.textContent = 'Calculating...';
        updateBtn.disabled = true;

        try {
            const response = await fetch('/api/calculate_dissonance', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ partials })
            });
            const data = await response.json();
            renderChart(data.alphas, data.dissonance);
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to calculate dissonance.');
        } finally {
            updateBtn.textContent = 'Update & Visualize';
            updateBtn.disabled = false;
        }
    }

    function renderChart(labels, data) {
        if (chart) {
            chart.destroy();
        }

        // Highlight minima (local minima logic could be added here, but visual is enough for now)

        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels.map(l => l.toFixed(2)),
                datasets: [{
                    label: 'Perceptual Roughness',
                    data: data,
                    borderColor: '#bb86fc',
                    backgroundColor: 'rgba(187, 134, 252, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: 'Frequency Ratio (Interval)', color: '#a0a0a0' },
                        ticks: { color: '#a0a0a0', maxTicksLimit: 10 },
                        grid: { color: '#333' }
                    },
                    y: {
                        title: { display: true, text: 'Roughness', color: '#a0a0a0' },
                        ticks: { color: '#a0a0a0' },
                        grid: { color: '#333' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#e0e0e0' } }
                }
            }
        });
    }

    // Audio Playback
    const keys = document.querySelectorAll('.key');
    keys.forEach(key => {
        key.addEventListener('click', () => {
            const freq = parseFloat(key.dataset.freq);
            playNote(freq);
        });
    });

    document.getElementById('play-custom-btn').addEventListener('click', () => {
        const freq = parseFloat(document.getElementById('custom-freq').value);
        playNote(freq);
    });

    async function playNote(freq) {
        try {
            const response = await fetch('/api/generate_audio', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    partials: partials,
                    freqs: [freq],
                    duration: 1.5
                })
            });

            if (!response.ok) throw new Error('Audio generation failed');

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);
            audio.play();
        } catch (error) {
            console.error('Error playing note:', error);
        }
    }

    // Initial render
    renderPartials();
    fetchData(); // Initial chart
});
