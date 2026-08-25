import math
import time
from collections import Counter
from enum import Enum

import numpy as np


class ParticleState(Enum):
    DEAD = 0
    ALIVE = 1


class Emitter:
    def __init__(
        self,
        num_particles: int = 2000,
        pos: np.ndarray = np.zeros(4),
        max_alive: int = 100,
        num_per_frame: int = 10,
    ) -> None:
        self.emitter_pos = pos
        self.num_particles = num_particles
        self.max_alive = max_alive
        self.min_life = 100
        self.max_life = 500
        self.num_per_frame = num_per_frame
        self.pos = np.zeros((num_particles, 4))
        self.dir = np.zeros((num_particles, 4))
        self.colour = np.zeros((num_particles, 4))
        self.life = np.full(num_particles, 20)
        self.state = np.full(num_particles, ParticleState.DEAD)
        # Frame timing for delta time calculation
        self._last_frame_time = time.time()
        for i in range(num_particles):
            self._reset_particle(i)
        self._birth_particles()

    def _reset_particle(self, index) -> None:
        # w is used to store the size of the particle
        emit_dir = np.array([0.0, 1.0, 0.0, 0.1])
        spread = 5.5
        self.pos[index] = self.emitter_pos
        self.dir[index] = (
            emit_dir * np.random.rand() + self._random_vector_on_sphere(1.0) * spread
        )
        self.dir[index][1] = abs(self.dir[index][1])
        self.colour[index] = np.random.rand(4)
        self.life[index] = self.min_life + np.random.randint(self.max_life)
        self.state[index] = ParticleState.DEAD

    def _reset_particles_vectorized(self, indices) -> None:
        """Vectorized reset for multiple particles at once."""
        if len(indices) == 0:
            return

        # w is used to store the size of the particle
        emit_dir = np.array([0.0, 1.0, 0.0, 0.1])
        spread = 5.5
        num_particles = len(indices)

        # Vectorized position reset
        self.pos[indices] = self.emitter_pos

        # Vectorized direction calculation
        # Generate random scalars for each particle
        random_scalars = np.random.rand(num_particles, 1)
        # Generate random vectors for each particle (fully vectorized)
        random_vectors = self._random_vectors_on_sphere(1.0, num_particles)
        # Combine and apply spread
        self.dir[indices] = emit_dir * random_scalars + random_vectors * spread
        # Ensure y component is positive (vectorized)
        self.dir[indices][:, 1] = np.abs(self.dir[indices][:, 1])

        # Vectorized colour generation
        self.colour[indices] = np.random.rand(num_particles, 4)

        # Vectorized life generation
        self.life[indices] = self.min_life + np.random.randint(
            self.max_life, size=num_particles
        )

        # Vectorized state reset
        self.state[indices] = ParticleState.DEAD

    def _random_vector_on_sphere(self, radius: float) -> np.ndarray:
        theta = np.random.uniform(0, 2 * math.pi)
        phi = np.random.uniform(0, math.pi)
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        return np.array([x, y, z, 0.0])

    def _random_vectors_on_sphere(self, radius: float, count: int) -> np.ndarray:
        """Vectorized version of _random_vector_on_sphere for multiple vectors."""
        # Generate arrays of random angles for all particles
        theta = np.random.uniform(0, 2 * math.pi, count)
        phi = np.random.uniform(0, math.pi, count)

        # Calculate spherical coordinates vectorized
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        # Stack into (count, 4) array with w=0
        return np.column_stack([x, y, z, np.zeros(count)])

    def get_numpy(self) -> np.ndarray:
        array = []
        for i in range(self.num_particles):
            if self.state[i] == ParticleState.ALIVE:
                array.extend(self.pos[i])
                array.extend(self.colour[i])
        return np.array(array, dtype=np.float32)

    def update(self, dt: float | None = None) -> None:
        gravity = np.array([0.0, -9.81, 0.0, 0.0])

        # Calculate delta time based on frame time if not provided
        if dt is None:
            current_time = time.time()
            dt = current_time - self._last_frame_time
            self._last_frame_time = current_time

            # Clamp dt to prevent simulation explosions on frame drops
            # This limits the maximum step to ~50ms (20 FPS)
            dt = min(dt, 0.05)

        # count number of particles alive in the system
        num_alive = Counter(self.state)[ParticleState.ALIVE]
        if num_alive < self.max_alive:
            self._birth_particles()

        alive = self.state == ParticleState.ALIVE
        self.dir[alive] += gravity * dt * 0.5
        self.pos[alive] += self.dir[alive] * dt * 0.5
        self.life[alive] -= 1

        # Find alive particles that should die and reset them
        should_die = (self.state == ParticleState.ALIVE) & (
            (self.life <= 0) | (self.pos[:, 1] < 0)
        )

        # Reset particles using vectorized operation
        self._reset_particles_vectorized(np.where(should_die)[0])

    def debug(self) -> None:
        print("Particles:")
        for i in range(self.num_particles):
            print(
                f"Particle {i}: pos={self.pos[i]}\ndir={self.dir[i]}\ncolour={self.colour[i]}\nlife={self.life[i]}\nstate={self.state[i]}"
            )

    def _birth_particles(self) -> None:
        births = np.random.randint(0, self.num_per_frame)
        for i in range(births):
            for p in range(self.num_particles):
                if self.state[p] == ParticleState.DEAD:
                    self._reset_particle(p)
                    self.state[p] = ParticleState.ALIVE
                    break
