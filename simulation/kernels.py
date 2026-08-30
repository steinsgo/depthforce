"""NVIDIA Warp kernels used by the particle system."""

import warp as wp


@wp.kernel
def integrate_particles(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    rest_positions: wp.array(dtype=wp.vec3),
    source_positions: wp.array(dtype=wp.vec3),
    source_directions: wp.array(dtype=wp.vec3),
    source_strengths: wp.array(dtype=float),
    source_radii: wp.array(dtype=float),
    source_count: int,
    dt: float,
    damping: float,
    spring_strength: float,
    max_velocity: float,
    idle_drift: float,
    simulation_time: float,
):
    particle_index = wp.tid()
    position = positions[particle_index]
    velocity = velocities[particle_index]
    rest = rest_positions[particle_index]
    acceleration = (rest - position) * spring_strength

    for source_index in range(source_count):
        difference = position - source_positions[source_index]
        radius = source_radii[source_index]
        distance_squared = wp.dot(difference, difference)
        if distance_squared < radius * radius:
            distance = wp.sqrt(distance_squared + 1.0e-8)
            radial_direction = difference / distance
            source_direction = source_directions[source_index]
            mixed_direction = wp.normalize(radial_direction * 0.84 + source_direction * 0.16)
            remaining = 1.0 - distance / radius
            # Cubic-ish falloff gives a soft boundary and a forceful core.
            falloff = remaining * remaining * (3.0 - 2.0 * remaining)
            acceleration += mixed_direction * source_strengths[source_index] * falloff

    if idle_drift > 0.0:
        drift = wp.vec3(
            wp.sin(rest[1] * 3.7 + rest[2] * 2.1 + simulation_time * 0.71),
            wp.sin(rest[2] * 4.1 + rest[0] * 1.8 + simulation_time * 0.57),
            wp.sin(rest[0] * 2.9 + rest[1] * 2.4 + simulation_time * 0.49),
        )
        acceleration += drift * idle_drift

    velocity += acceleration * dt
    velocity *= wp.exp(-damping * dt)
    speed = wp.length(velocity)
    if speed > max_velocity:
        velocity *= max_velocity / speed
    position += velocity * dt

    velocities[particle_index] = velocity
    positions[particle_index] = position


@wp.kernel
def reset_particles(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    rest_positions: wp.array(dtype=wp.vec3),
):
    particle_index = wp.tid()
    positions[particle_index] = rest_positions[particle_index]
    velocities[particle_index] = wp.vec3(0.0, 0.0, 0.0)
