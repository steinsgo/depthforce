"""Small compatibility helpers around Warp's current OpenGL renderer."""


def configure_particle_geometry(renderer, latitudes: int = 4, longitudes: int = 6) -> None:
    """Keep tiny particles cheap on Warp 1.16, where ``as_spheres`` is ignored.

    Warp 1.16 creates a 32x32 sphere (2,048 triangles) for every instanced point.
    At a 2-5 pixel on-screen radius that detail is invisible and makes 150k points
    unnecessarily expensive. This changes only the per-renderer mesh factory; it
    does not patch the installed Warp package.
    """
    original_factory = renderer._create_sphere_mesh

    def create_small_sphere(radius=1.0, *args, **kwargs):
        reverse_winding = bool(kwargs.get("reverse_winding", False))
        return original_factory(radius, latitudes, longitudes, reverse_winding)

    renderer._create_sphere_mesh = create_small_sphere
