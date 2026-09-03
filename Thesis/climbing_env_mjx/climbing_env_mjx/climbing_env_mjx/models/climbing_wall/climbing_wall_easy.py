from dm_control import composer, mjcf
from pathlib import Path
import numpy as np
import struct

YELLOW = [0.999, 0.99, 0.2, 1]

POS_WALL_AND_JUG=[0.5, -0.0, 3]
POS_GROUND=[0, 0, 0]

POS_EASY=[0.47, 0.0, 1.2]
POS_X=[0.47,0.3, 1.6]
class ClimbingWall(composer.Entity):
    def _build(self, name="climbing_name"):
        self._mjcf_root = mjcf.RootElement(model=name)

        self._mjcf_root.option.iterations = 100
        self._mjcf_root.option.ls_iterations = 100
        mesh_file = Path(__file__).parent / "meshes" / "appiglio_binary.stl"
        mesh_asset_name = "jug_mesh_asset"
        self.make_jug("appiglio_binary.stl", YELLOW)
        self._mjcf_root.asset.add(
            "mesh",
            name=mesh_asset_name,
            file=str(mesh_file.resolve()),
            scale=[0.001, 0.001, 0.001]
        )

        self._vertices_local = self._load_stl_vertices(mesh_file) * 0.001
        self._cached_step = -1
        self._cached_world_vertices = None

        self._body = self._mjcf_root.worldbody.add(
            "body",
            name="jug_body",
            pos=POS_EASY,
            euler=[0, -np.pi/2, -np.pi],
        )

        self._geom = self._body.add(
            "geom",
            name="jug_mesh_asset",
            type="mesh",
            mesh=mesh_asset_name,
            friction=[5.0, 3.0, 3.0],
            condim=6,
            rgba=[0.999, 0.5, 0.5, 1],
            contype=1,
            conaffinity=1,
            margin=0.002,
            solimp=[0.99999, 0.99999, 0.000001],
            solref=[0.0001, 2.0],
        )
        self._wall = self._mjcf_root.worldbody.add(
            "geom",
            name="wall",
            type="box",
            size=[0.001, 1, 3],
            pos=POS_WALL_AND_JUG,
            rgba=[0.3, 0.35, 0.4, 0.7],
            contype=1,
            conaffinity=1,
            solimp=[0.999, 0.999, 0.0001],
            solref=[0.001, 1.0],
            friction=[1.0, 0.1, 0.01],
        )
        self._ground = self._mjcf_root.worldbody.add(
            "geom",
            name="ground",
            type="plane",
            size=[0, 0, 0.05],
            pos=POS_GROUND,
            rgba=[0.15, 0.15, 0.15, 0.6],
            contype=1,
            conaffinity=1,
            solimp=[0.999, 0.999, 0.0001],
            solref=[0.001, 1.0],
            friction=[1.0, 0.1, 0.01],
        )
    @property
    def mjcf_model(self):
        return self._mjcf_root

    def make_jug(self, jugtype,color):
        mesh_file = Path(__file__).parent / "meshes" / jugtype

        mesh_asset_name = "jug_mesh_asset_2"
        self._mjcf_root.asset.add(
            "mesh",
            name=mesh_asset_name,
            file=str(mesh_file.resolve()),
            scale=[0.001, 0.001, 0.001]
        )

        self._vertices_local = self._load_stl_vertices(mesh_file) * 0.001
        self._cached_step = -1
        self._cached_world_vertices = None

        self._body = self._mjcf_root.worldbody.add(
            "body",
            name="jug_body_2",
            pos=POS_X,
            euler=[0, -np.pi/2, -np.pi],
        )

        self._geom = self._body.add(
            "geom",
            name="jug_mesh_asset_2",
            type="mesh",
            mesh=mesh_asset_name,
            friction=[5.0, 3.0, 3.0],
            condim=6,
            rgba=color,
            contype=1,
            conaffinity=1,
            margin=0.002,
            solimp=[0.99999, 0.99999, 0.000001],
            solref=[0.0001, 2.0],
        )

    def _load_stl_vertices(self, path: Path) -> np.ndarray:
        if not path.exists():
            return np.zeros((0,3), dtype=np.float32)
        data = path.read_bytes()
        if len(data) > 84:
            tri_count = struct.unpack('<I', data[80:84])[0]
            expected_len = 84 + tri_count * 50
            if len(data) == expected_len:
                vertices = []
                offset = 84
                for _ in range(tri_count):
                    offset += 12
                    for _v in range(3):
                        x,y,z = struct.unpack('<fff', data[offset:offset+12])
                        vertices.append((x,y,z))
                        offset += 12
                    offset += 2
                arr = np.array(vertices, dtype=np.float32)
                _, idx = np.unique(arr, axis=0, return_index=True)
                return arr[idx]
        verts = []
        for line in path.read_text(errors='ignore').splitlines():
            line = line.strip()
            if line.startswith('vertex'):
                parts = line.split()
                if len(parts) == 4:
                    verts.append(tuple(float(p) for p in parts[1:]))
        arr = np.array(verts, dtype=np.float32)
        if arr.size == 0:
            return arr
        _, idx = np.unique(arr, axis=0, return_index=True)
        return arr[idx]

    def closest_surface_distance(self, physics, point_world: np.ndarray) -> float:
        if self._vertices_local.size == 0:
            return 0.0
        step = physics.data.time
        if step != self._cached_step:
            body_xpos = physics.bind(self._body).xpos
            body_xmat = physics.bind(self._body).xmat.reshape(3,3)
            self._cached_world_vertices = body_xpos + self._vertices_local @ body_xmat.T
            self._cached_step = step
        diffs = self._cached_world_vertices - point_world
        return float(np.min(np.linalg.norm(diffs, axis=1)))
