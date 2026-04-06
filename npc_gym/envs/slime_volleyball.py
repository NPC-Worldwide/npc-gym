"""
Slime Volleyball environment for npc-gym.

Direct port of hardmaru/slimevolleygym physics and baseline policy.
Same constants, same coordinate system, same collision, same observations.

Usage:
    env = SlimeVolleyEnv()                          # vs original baseline RNN
    env = SlimeVolleyEnv(opponent=BuiltInAI("hard")) # vs heuristic AI
    env = SlimeVolleyEnv(self_play=True)             # self-play mode

    obs = env.reset()
    obs, reward, done, info = env.step(action)
    # self-play: obs, reward, done, info = env.step(action1, action2)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np


# ---------------------------------------------------------------------------
# Physics constants — EXACT from hardmaru/slimevolleygym
# ---------------------------------------------------------------------------

REF_W = 24 * 2            # 48 — court width
REF_H = REF_W             # 48 — court height
REF_U = 1.5               # ground level
REF_WALL_WIDTH = 1.0      # net width
REF_WALL_HEIGHT = 3.5     # net height
PLAYER_SPEED_X = 10 * 1.75   # 17.5
PLAYER_SPEED_Y = 10 * 1.35   # 13.5
MAX_BALL_SPEED = 15 * 1.5    # 22.5
TIMESTEP = 1 / 30.0          # 0.0333...
NUDGE = 0.1
FRICTION = 1.0            # 1 = no friction
GRAVITY = -9.8 * 2 * 1.5  # -29.4
MAXLIVES = 5
INIT_DELAY_FRAMES = 30


# ---------------------------------------------------------------------------
# Particle — used for ball and fence stub
# ---------------------------------------------------------------------------

class Particle:
    def __init__(self, x, y, vx, vy, r):
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y
        self.vx = vx
        self.vy = vy
        self.r = r

    def move(self):
        self.prev_x = self.x
        self.prev_y = self.y
        self.x += self.vx * TIMESTEP
        self.y += self.vy * TIMESTEP

    def applyAcceleration(self, ax, ay):
        self.vx += ax * TIMESTEP
        self.vy += ay * TIMESTEP

    def checkEdges(self):
        # left wall
        if self.x <= (self.r - REF_W / 2):
            self.vx *= -FRICTION
            self.x = self.r - REF_W / 2 + NUDGE * TIMESTEP
        # right wall
        if self.x >= (REF_W / 2 - self.r):
            self.vx *= -FRICTION
            self.x = REF_W / 2 - self.r - NUDGE * TIMESTEP
        # ground — returns -1 if landed on left side, +1 if right side
        if self.y <= (self.r + REF_U):
            self.vy *= -FRICTION
            self.y = self.r + REF_U + NUDGE * TIMESTEP
            if self.x <= 0:
                return -1
            else:
                return 1
        # ceiling
        if self.y >= (REF_H - self.r):
            self.vy *= -FRICTION
            self.y = REF_H - self.r - NUDGE * TIMESTEP
        # fence (right side)
        if ((self.x <= (REF_WALL_WIDTH / 2 + self.r)) and
                (self.prev_x > (REF_WALL_WIDTH / 2 + self.r)) and
                (self.y <= REF_WALL_HEIGHT)):
            self.vx *= -FRICTION
            self.x = REF_WALL_WIDTH / 2 + self.r + NUDGE * TIMESTEP
        # fence (left side)
        if ((self.x >= (-REF_WALL_WIDTH / 2 - self.r)) and
                (self.prev_x < (-REF_WALL_WIDTH / 2 - self.r)) and
                (self.y <= REF_WALL_HEIGHT)):
            self.vx *= -FRICTION
            self.x = -REF_WALL_WIDTH / 2 - self.r - NUDGE * TIMESTEP
        return 0

    def getDist2(self, p):
        dx = p.x - self.x
        dy = p.y - self.y
        return dx * dx + dy * dy

    def isColliding(self, p):
        r = self.r + p.r
        return r * r > self.getDist2(p)

    def bounce(self, p):
        abx = self.x - p.x
        aby = self.y - p.y
        abd = math.sqrt(abx * abx + aby * aby)
        abx /= abd
        aby /= abd
        nx = abx
        ny = aby
        abx *= NUDGE
        aby *= NUDGE
        while self.isColliding(p):
            self.x += abx
            self.y += aby
        ux = self.vx - p.vx
        uy = self.vy - p.vy
        un = ux * nx + uy * ny
        unx = nx * (un * 2.0)
        uny = ny * (un * 2.0)
        ux -= unx
        uy -= uny
        self.vx = ux + p.vx
        self.vy = uy + p.vy

    def limitSpeed(self, minSpeed, maxSpeed):
        mag2 = self.vx * self.vx + self.vy * self.vy
        if mag2 > (maxSpeed * maxSpeed):
            mag = math.sqrt(mag2)
            self.vx /= mag
            self.vy /= mag
            self.vx *= maxSpeed
            self.vy *= maxSpeed
        if mag2 < (minSpeed * minSpeed):
            mag = math.sqrt(mag2)
            if mag > 0:
                self.vx /= mag
                self.vy /= mag
                self.vx *= minSpeed
                self.vy *= minSpeed


# ---------------------------------------------------------------------------
# RelativeState — observation computation (exact original)
# ---------------------------------------------------------------------------

class RelativeState:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.bx = 0
        self.by = 0
        self.bvx = 0
        self.bvy = 0
        self.ox = 0
        self.oy = 0
        self.ovx = 0
        self.ovy = 0

    def getObservation(self):
        result = [self.x, self.y, self.vx, self.vy,
                  self.bx, self.by, self.bvx, self.bvy,
                  self.ox, self.oy, self.ovx, self.ovy]
        scaleFactor = 10.0
        return np.array(result) / scaleFactor


# ---------------------------------------------------------------------------
# Agent — player entity (exact original)
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, dir, x, y, r=1.5):
        self.dir = dir   # -1 = left, 1 = right
        self.x = x
        self.y = y
        self.r = r
        self.vx = 0
        self.vy = 0
        self.desired_vx = 0
        self.desired_vy = 0
        self.state = RelativeState()
        self.life = MAXLIVES

    def lives(self):
        return self.life

    def setAction(self, action):
        forward = action[0] > 0
        backward = action[1] > 0
        jump = action[2] > 0
        self.desired_vx = 0
        self.desired_vy = 0
        if forward and not backward:
            self.desired_vx = -PLAYER_SPEED_X
        if backward and not forward:
            self.desired_vx = PLAYER_SPEED_X
        if jump:
            self.desired_vy = PLAYER_SPEED_Y

    def update(self):
        self.vy += GRAVITY * TIMESTEP
        if self.y <= REF_U + NUDGE * TIMESTEP:
            self.vy = self.desired_vy
        self.vx = self.desired_vx * self.dir
        self.x += self.vx * TIMESTEP
        self.y += self.vy * TIMESTEP
        if self.y <= REF_U:
            self.y = REF_U
            self.vy = 0
        # stay in own half
        if self.x * self.dir <= (REF_WALL_WIDTH / 2 + self.r):
            self.vx = 0
            self.x = self.dir * (REF_WALL_WIDTH / 2 + self.r)
        if self.x * self.dir >= (REF_W / 2 - self.r):
            self.vx = 0
            self.x = self.dir * (REF_W / 2 - self.r)

    def updateState(self, ball, opponent):
        self.state.x = self.x * self.dir
        self.state.y = self.y
        self.state.vx = self.vx * self.dir
        self.state.vy = self.vy
        self.state.bx = ball.x * self.dir
        self.state.by = ball.y
        self.state.bvx = ball.vx * self.dir
        self.state.bvy = ball.vy
        self.state.ox = opponent.x * (-self.dir)
        self.state.oy = opponent.y
        self.state.ovx = opponent.vx * (-self.dir)
        self.state.ovy = opponent.vy

    def getObservation(self):
        return self.state.getObservation()


# ---------------------------------------------------------------------------
# Game — the core game loop (exact original)
# ---------------------------------------------------------------------------

class Game:
    def __init__(self, np_random=np.random):
        self.np_random = np_random
        self.ball = None
        self.fenceStub = None
        self.agent_left = None
        self.agent_right = None
        self.delay = 0
        self.reset()

    def reset(self):
        self.fenceStub = Particle(0, REF_WALL_HEIGHT, 0, 0, REF_WALL_WIDTH / 2)
        ball_vx = self.np_random.uniform(low=-20, high=20)
        ball_vy = self.np_random.uniform(low=10, high=25)
        self.ball = Particle(0, REF_W / 4, ball_vx, ball_vy, 0.5)
        self.agent_left = Agent(-1, -REF_W / 4, REF_U)
        self.agent_right = Agent(1, REF_W / 4, REF_U)
        self.agent_left.updateState(self.ball, self.agent_right)
        self.agent_right.updateState(self.ball, self.agent_left)
        self.delay = INIT_DELAY_FRAMES

    def newMatch(self):
        ball_vx = self.np_random.uniform(low=-20, high=20)
        ball_vy = self.np_random.uniform(low=10, high=25)
        self.ball = Particle(0, REF_W / 4, ball_vx, ball_vy, 0.5)
        self.delay = INIT_DELAY_FRAMES

    def step(self):
        # delay countdown
        if self.delay > 0:
            self.delay -= 1

        self.agent_left.update()
        self.agent_right.update()

        if self.delay <= 0:
            self.ball.applyAcceleration(0, GRAVITY)
            self.ball.limitSpeed(0, MAX_BALL_SPEED)
            self.ball.move()

        if self.ball.isColliding(self.agent_left):
            self.ball.bounce(self.agent_left)
        if self.ball.isColliding(self.agent_right):
            self.ball.bounce(self.agent_right)
        if self.ball.isColliding(self.fenceStub):
            self.ball.bounce(self.fenceStub)

        # negated: reward from perspective of right agent
        result = -self.ball.checkEdges()

        if result != 0:
            self.newMatch()
            if result < 0:
                self.agent_right.life -= 1
            else:
                self.agent_left.life -= 1
            return result

        self.agent_left.updateState(self.ball, self.agent_right)
        self.agent_right.updateState(self.ball, self.agent_left)

        return result


# ---------------------------------------------------------------------------
# BaselinePolicy — exact 120-param RNN from David Ha (2015)
# ---------------------------------------------------------------------------

class BaselinePolicy:
    """
    The exact baseline opponent from hardmaru/slimevolleygym.
    Tiny RNN: 7x15 weights + 7 bias = 112 params, 4 recurrent states.
    Trained via neuroevolution: https://blog.otoro.net/2015/03/28/neural-slime-volleyball/
    """

    def __init__(self):
        self.nGameInput = 8
        self.nGameOutput = 3
        self.nRecurrentState = 4
        self.nOutput = self.nGameOutput + self.nRecurrentState  # 7
        self.nInput = self.nGameInput + self.nOutput  # 15

        self.inputState = np.zeros(self.nInput)
        self.outputState = np.zeros(self.nOutput)
        self.prevOutputState = np.zeros(self.nOutput)

        self.weight = np.array([
            7.5719, 4.4285, 2.2716, -0.3598, -7.8189, -2.5422, -3.2034,
            0.3935, 1.2202, -0.49, -0.0316, 0.5221, 0.7026, 0.4179, -2.1689,
            1.646, -13.3639, 1.5151, 1.1175, -5.3561, 5.0442, 0.8451,
            0.3987, -2.9501, -3.7811, -5.8994, 6.4167, 2.5014, 7.338, -2.9887,
            2.4586, 13.4191, 2.7395, -3.9708, 1.6548, -2.7554, -1.5345,
            -6.4708, 9.2426, -0.7392, 0.4452, 1.8828, -2.6277, -10.851, -3.2353,
            -4.4653, -3.1153, -1.3707, 7.318, 16.0902, 1.4686, 7.0391,
            1.7765, -1.155, 2.6697, -8.8877, 1.1958, -3.2839, -5.4425, 1.6809,
            7.6812, -2.4732, 1.738, 0.3781, 0.8718, 2.5886, 1.6911,
            1.2953, -9.0052, -4.6038, -6.7447, -2.5528, 0.4391, -4.9278, -3.6695,
            -4.8673, -1.6035, 1.5011, -5.6124, 4.9747, 1.8998, 3.0359,
            6.2983, -4.8568, -2.1888, -4.1143, -3.9874, -0.0459, 4.7134, 2.8952,
            -9.3627, -4.685, 0.3601, -1.3699, 9.7294, 11.5596, 0.1918,
            3.0783, 0.0329, -0.1362, -0.1188, -0.7579, 0.3278, -0.977, -0.9377,
        ]).reshape(self.nOutput, self.nInput)

        self.bias = np.array([2.2935, -2.0353, -1.7786, 5.4567, -3.6368, 3.4996, -0.0685])

    def reset(self):
        self.inputState = np.zeros(self.nInput)
        self.outputState = np.zeros(self.nOutput)
        self.prevOutputState = np.zeros(self.nOutput)

    def predict(self, obs):
        # obs: 12-element observation (already scaled by /10 from RelativeState)
        # Input: first 8 obs values + 7 previous output state
        self.inputState[:self.nGameInput] = obs[:self.nGameInput]
        self.inputState[self.nGameInput:] = self.outputState
        self.prevOutputState = self.outputState
        self.outputState = np.tanh(self.weight @ self.inputState + self.bias)
        forward = 1 if self.outputState[0] > 0.75 else 0
        backward = 1 if self.outputState[1] > 0.75 else 0
        jump = 1 if self.outputState[2] > 0.75 else 0
        return [forward, backward, jump]


# ---------------------------------------------------------------------------
# SlimeVolleyEnv — gymnasium-compatible wrapper (exact original API)
# ---------------------------------------------------------------------------

class SlimeVolleyEnv:
    """
    Slime Volleyball environment — direct port of hardmaru/slimevolleygym.

    By default, trains the RIGHT agent against the baseline RNN on the LEFT.
    Reward is from the right agent's perspective (+1 win point, -1 lose point).

    Args:
        opponent: Custom opponent policy (must have .predict(obs) or .act(obs))
                  Default: BaselinePolicy (original 120-param RNN)
        self_play: If True, step() accepts action2 for left agent

    Observation: 12-dim float vector (agent x,y,vx,vy, ball x,y,vx,vy, opp x,y,vx,vy)
                 divided by 10.0 (same scaling as original)
    Action: [forward, backward, jump] — binary or float > 0
    """

    def __init__(self, opponent=None, self_play=False):
        self.t = 0
        self.t_limit = 3000
        self.self_play = self_play

        self.observation_size = 12
        self.action_size = 3

        self.game = Game()
        self.policy = opponent or BaselinePolicy()

    def reset(self, seed=None) -> np.ndarray:
        if seed is not None:
            self.game = Game(np_random=np.random.RandomState(seed))
        else:
            self.game.reset()
        self.t = 0
        if hasattr(self.policy, 'reset'):
            self.policy.reset()
        return self.game.agent_right.getObservation()

    def step(self, action, action2=None) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Args:
            action: Right agent's action [forward, backward, jump]
            action2: Left agent's action (self-play). If None, uses self.policy.

        Returns:
            obs, reward, done, info
        """
        self.t += 1

        if action2 is not None:
            otherAction = action2
        else:
            obs_left = self.game.agent_left.getObservation()
            if hasattr(self.policy, 'predict'):
                otherAction = self.policy.predict(obs_left)
            elif hasattr(self.policy, 'act'):
                otherAction = self.policy.act(obs_left)
            else:
                otherAction = [0, 0, 0]

        self.game.agent_left.setAction(otherAction)
        self.game.agent_right.setAction(action)

        reward = self.game.step()

        obs = self.game.agent_right.getObservation()

        done = False
        if self.t >= self.t_limit:
            done = True
        if self.game.agent_left.life <= 0 or self.game.agent_right.life <= 0:
            done = True

        info = {
            'ale.lives': self.game.agent_right.lives(),
            'ale.otherLives': self.game.agent_left.lives(),
            'otherObs': self.game.agent_left.getObservation(),
            'score_right': MAXLIVES - self.game.agent_right.life,
            'score_left': MAXLIVES - self.game.agent_left.life,
        }

        return obs, reward, done, info

    def set_opponent(self, opponent):
        self.opponent = opponent
        self.policy = opponent

    def get_state(self) -> Dict[str, Any]:
        """Get raw state for rendering."""
        return {
            "player1": {"x": self.game.agent_left.x, "y": self.game.agent_left.y},
            "player2": {"x": self.game.agent_right.x, "y": self.game.agent_right.y},
            "ball": {"x": self.game.ball.x, "y": self.game.ball.y},
            "score": [
                MAXLIVES - self.game.agent_left.life,
                MAXLIVES - self.game.agent_right.life,
            ],
            "court": {"width": REF_W, "height": REF_H},
            "net_height": REF_WALL_HEIGHT,
            "ground": REF_U,
        }

    def close(self):
        pass
