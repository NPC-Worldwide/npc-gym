"""
Web visualization server for npc-gym.

Provides:
- Real-time game state visualization
- Training progress dashboard
- Trace replay system
- Gene pool evolution viewer
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional
from dataclasses import asdict

try:
    from flask import Flask, render_template_string, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>npc-gym Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #00d9ff; margin-bottom: 20px; }
        h2 { color: #00ff88; margin: 20px 0 10px; font-size: 1.2em; }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }

        .card {
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #0f3460;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 10px;
        }

        .stat {
            background: #0f3460;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }

        .stat-value {
            font-size: 2em;
            color: #00d9ff;
            font-weight: bold;
        }

        .stat-label {
            font-size: 0.9em;
            color: #888;
            margin-top: 5px;
        }

        .player-list {
            list-style: none;
            margin-top: 10px;
        }

        .player-item {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background: #0f3460;
            margin: 5px 0;
            border-radius: 4px;
        }

        .player-name { color: #00ff88; }
        .player-stack { color: #00d9ff; }
        .player-status { color: #888; }

        .gene-list {
            list-style: none;
            margin-top: 10px;
        }

        .gene-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background: #0f3460;
            margin: 5px 0;
            border-radius: 4px;
        }

        .gene-name { color: #ff6b6b; }
        .gene-fitness {
            background: #00ff88;
            color: #000;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
        }

        .progress-bar {
            height: 20px;
            background: #0f3460;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            transition: width 0.3s ease;
        }

        .log-area {
            background: #0a0a14;
            padding: 15px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 0.9em;
            max-height: 300px;
            overflow-y: auto;
            margin-top: 10px;
        }

        .log-entry { margin: 5px 0; }
        .log-time { color: #666; }
        .log-event { color: #00d9ff; }
        .log-data { color: #00ff88; }

        .hypothesis-card {
            background: #0f3460;
            padding: 15px;
            border-radius: 6px;
            margin: 10px 0;
        }

        .hypothesis-text {
            color: #fff;
            font-style: italic;
            margin-bottom: 10px;
        }

        .hypothesis-score {
            color: #00ff88;
            font-weight: bold;
        }

        .btn {
            background: #00d9ff;
            color: #000;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            margin: 5px;
        }

        .btn:hover { background: #00ff88; }

        #refresh-status {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #00ff88;
            color: #000;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.8em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>npc-gym Dashboard</h1>

        <div class="grid">
            <!-- Training Stats -->
            <div class="card">
                <h2>Training Progress</h2>
                <div class="stat-grid">
                    <div class="stat">
                        <div class="stat-value" id="epoch">0</div>
                        <div class="stat-label">Epoch</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="total-games">0</div>
                        <div class="stat-label">Games Played</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="traces">0</div>
                        <div class="stat-label">Traces</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="best-fitness">0.00</div>
                        <div class="stat-label">Best Fitness</div>
                    </div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress" style="width: 0%"></div>
                </div>
            </div>

            <!-- Current Game -->
            <div class="card">
                <h2>Current Game State</h2>
                <p>Phase: <span id="phase" style="color: #00ff88;">-</span></p>
                <p>Pot: $<span id="pot">0</span></p>
                <p>Step: <span id="step">0</span></p>

                <h3 style="margin-top: 15px; color: #888;">Players</h3>
                <ul class="player-list" id="player-list">
                    <li class="player-item">No game active</li>
                </ul>
            </div>

            <!-- Gene Pool -->
            <div class="card">
                <h2>Model Genome</h2>
                <p>Generation: <span id="generation" style="color: #00d9ff;">0</span></p>
                <ul class="gene-list" id="gene-list">
                    <li class="gene-item">No genes evolved yet</li>
                </ul>
            </div>

            <!-- Hypotheses -->
            <div class="card">
                <h2>Current Hypotheses</h2>
                <div id="hypotheses">
                    <p style="color: #666;">No hypotheses yet</p>
                </div>
            </div>

            <!-- Event Log -->
            <div class="card" style="grid-column: 1 / -1;">
                <h2>Event Log</h2>
                <div class="log-area" id="log-area">
                    <div class="log-entry">
                        <span class="log-time">[--:--:--]</span>
                        <span class="log-event">Waiting for events...</span>
                    </div>
                </div>
            </div>
        </div>

        <div style="margin-top: 20px;">
            <button class="btn" onclick="refresh()">Refresh</button>
            <button class="btn" onclick="toggleAutoRefresh()">Auto Refresh: <span id="auto-status">OFF</span></button>
        </div>
    </div>

    <div id="refresh-status" style="display: none;">Refreshing...</div>

    <script>
        let autoRefresh = false;
        let refreshInterval = null;

        async function refresh() {
            document.getElementById('refresh-status').style.display = 'block';

            try {
                const response = await fetch('/api/state');
                const data = await response.json();
                updateDashboard(data);
            } catch (e) {
                console.error('Refresh failed:', e);
            }

            setTimeout(() => {
                document.getElementById('refresh-status').style.display = 'none';
            }, 500);
        }

        function updateDashboard(data) {
            // Training stats
            document.getElementById('epoch').textContent = data.epoch || 0;
            document.getElementById('total-games').textContent = data.total_games || 0;
            document.getElementById('traces').textContent = data.traces || 0;
            document.getElementById('best-fitness').textContent =
                (data.best_fitness || 0).toFixed(3);

            if (data.max_epochs) {
                const progress = (data.epoch / data.max_epochs) * 100;
                document.getElementById('progress').style.width = progress + '%';
            }

            // Game state
            document.getElementById('phase').textContent = data.phase || '-';
            document.getElementById('pot').textContent = data.pot || 0;
            document.getElementById('step').textContent = data.step || 0;

            // Players
            const playerList = document.getElementById('player-list');
            if (data.players && Object.keys(data.players).length > 0) {
                playerList.innerHTML = Object.entries(data.players).map(([id, p]) => `
                    <li class="player-item">
                        <span class="player-name">${id}</span>
                        <span class="player-stack">$${p.stack || 0}</span>
                        <span class="player-status">${p.status || 'active'}</span>
                    </li>
                `).join('');
            }

            // Gene pool
            document.getElementById('generation').textContent = data.generation || 0;
            const geneList = document.getElementById('gene-list');
            if (data.genes && data.genes.length > 0) {
                geneList.innerHTML = data.genes.map(g => `
                    <li class="gene-item">
                        <span class="gene-name">${g.specialization}</span>
                        <span class="gene-fitness">${(g.fitness || 0).toFixed(3)}</span>
                    </li>
                `).join('');
            }

            // Hypotheses
            const hypothesesDiv = document.getElementById('hypotheses');
            if (data.hypotheses && Object.keys(data.hypotheses).length > 0) {
                hypothesesDiv.innerHTML = Object.entries(data.hypotheses).map(([id, h]) => `
                    <div class="hypothesis-card">
                        <div class="hypothesis-text">"${h.text || 'No hypothesis'}"</div>
                        <span class="hypothesis-score">Score: ${(h.score || 0).toFixed(2)}</span>
                        <span style="color: #888; margin-left: 10px;">
                            Confidence: ${((h.confidence || 0) * 100).toFixed(0)}%
                        </span>
                    </div>
                `).join('');
            }

            // Log
            if (data.events && data.events.length > 0) {
                const logArea = document.getElementById('log-area');
                logArea.innerHTML = data.events.slice(-20).map(e => `
                    <div class="log-entry">
                        <span class="log-time">[${e.time || '--:--:--'}]</span>
                        <span class="log-event">${e.event || ''}</span>
                        <span class="log-data">${e.data || ''}</span>
                    </div>
                `).join('');
                logArea.scrollTop = logArea.scrollHeight;
            }
        }

        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            document.getElementById('auto-status').textContent = autoRefresh ? 'ON' : 'OFF';

            if (autoRefresh) {
                refreshInterval = setInterval(refresh, 2000);
            } else if (refreshInterval) {
                clearInterval(refreshInterval);
            }
        }

        // Initial load
        refresh();
    </script>
</body>
</html>
"""


class VisualizationServer:
    """
    Web server for visualizing npc-gym training and games.

    Usage:
        server = VisualizationServer(training_loop)
        server.run(port=5000)
    """

    def __init__(
        self,
        training_loop: Any = None,
        env: Any = None,
        port: int = 5000
    ):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask required for visualization. Install with: pip install flask")

        self.training_loop = training_loop
        self.env = env
        self.port = port
        self.events: List[Dict] = []

        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            return render_template_string(DASHBOARD_HTML)

        @self.app.route('/api/state')
        def get_state():
            return jsonify(self._get_current_state())

        @self.app.route('/api/events')
        def get_events():
            return jsonify({"events": self.events[-100:]})

    def _get_current_state(self) -> Dict[str, Any]:
        """Get current state for dashboard."""
        state = {
            "epoch": 0,
            "total_games": 0,
            "traces": 0,
            "best_fitness": 0.0,
            "max_epochs": 10,
            "phase": "-",
            "pot": 0,
            "step": 0,
            "players": {},
            "generation": 0,
            "genes": [],
            "hypotheses": {},
            "events": self.events[-20:],
        }

        # From training loop
        if self.training_loop:
            state["epoch"] = self.training_loop.epoch
            state["total_games"] = self.training_loop.total_games
            state["traces"] = len(self.training_loop.trace_collector.buffer)
            state["max_epochs"] = self.training_loop.config.num_epochs

            # Gene pool
            if self.training_loop.model_evolver.gene_pool:
                pool = self.training_loop.model_evolver.gene_pool
                state["generation"] = pool.generation
                state["genes"] = [
                    {"specialization": g.specialization, "fitness": g.fitness}
                    for g in sorted(pool.genes, key=lambda x: x.fitness, reverse=True)[:10]
                ]
                if pool.genes:
                    state["best_fitness"] = max(g.fitness for g in pool.genes)

            # Current env
            if self.training_loop.env:
                self.env = self.training_loop.env

        # From environment
        if self.env and hasattr(self.env, 'state') and self.env.state:
            env_state = self.env.state
            state["phase"] = env_state.phase.value if hasattr(env_state.phase, 'value') else str(env_state.phase)
            state["step"] = env_state.step
            state["pot"] = getattr(self.env, 'pot', 0)

            # Players
            if hasattr(self.env, 'players'):
                for pid, player in self.env.players.items():
                    status = "folded" if getattr(player, 'folded', False) else "active"
                    if getattr(player, 'busted', False):
                        status = "busted"
                    state["players"][pid] = {
                        "stack": getattr(player, 'stack', 0),
                        "status": status,
                    }

            # Hypotheses
            if hasattr(self.env, 'hypotheses'):
                for pid, hyp_data in self.env.hypotheses.items():
                    state["hypotheses"][pid] = {
                        "text": hyp_data.get("current", "")[:200],
                        "score": hyp_data.get("score", 0),
                        "confidence": hyp_data.get("confidence", 0.5),
                    }

        return state

    def add_event(self, event: str, data: str = "") -> None:
        """Add an event to the log."""
        from datetime import datetime
        self.events.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "event": event,
            "data": data,
        })
        # Keep last 1000 events
        if len(self.events) > 1000:
            self.events = self.events[-1000:]

    def run(self, port: int = None, debug: bool = False) -> None:
        """Start the visualization server."""
        if port:
            self.port = port

        print(f"Starting npc-gym visualization server at http://localhost:{self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)


def visualize_training(training_loop: Any, port: int = 5000) -> VisualizationServer:
    """
    Quick helper to visualize a training loop.

    Args:
        training_loop: TrainingLoop instance
        port: Port to run server on

    Returns:
        VisualizationServer instance (call .run() to start)
    """
    server = VisualizationServer(training_loop=training_loop, port=port)
    return server


def visualize_game(env: Any, port: int = 5000) -> VisualizationServer:
    """
    Quick helper to visualize a single game.

    Args:
        env: Environment instance
        port: Port to run server on

    Returns:
        VisualizationServer instance
    """
    server = VisualizationServer(env=env, port=port)
    return server
