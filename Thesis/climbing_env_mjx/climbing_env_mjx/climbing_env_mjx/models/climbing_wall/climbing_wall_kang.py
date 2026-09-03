from dm_control import composer, mjcf
import numpy as np


class KangClimbingWall(composer.Entity):

    def _build(
        self,
        name="climbing_name",
        incline_deg=0.0,
        wall_x=0.5,
        wall_width=2.0,
        wall_length=5.2,
        wall_thickness=0.2,
        hold_rows=12,
        hold_cols=4,
        hold_v0=0.3,
        hold_row_dv=0.4,
        hold_col_dy=0.4,
        hold_radius=0.05,
        hold_protrude=0.03,
        hold_jitter=0.0,
        hold_seed=0,
        hold_y_offset=0.0,
        mirror_y=False,
        hold_dropout=0.0,
        hold_dropout_seed=0,
        holds=None,
        hold_contype=2,
        hold_conaffinity=2,
        hold_mesh=None,
        hold_mesh_scale=0.0015,
        hold_mesh_protrude=0.033,
        hold_mesh_rgba=(0.58, 0.32, 0.76, 1.0),
        hold_mesh_euler=None,
        hold_mesh_collider=False,
        hold_mesh_maxhullvert=24,
        mesh_collider_holds=None,
        hold_mesh_friction=(5.0, 3.0, 3.0),
        hold_mesh_condim=6,
        hold_mesh_solimp=(0.999, 0.999, 0.0001),
        hold_mesh_solref=(0.006, 1.0),
    ):
        self._mjcf_root = mjcf.RootElement(model=name)

        t = np.radians(incline_deg)
        u = np.array([-np.sin(t), 0.0, np.cos(t)])
        n = np.array([-np.cos(t), 0.0, -np.sin(t)])
        base = np.array([wall_x, 0.0, 0.0])

        half = np.array([wall_thickness / 2.0, wall_width / 2.0, wall_length / 2.0])
        quat = [np.cos(-t / 2.0), 0.0, np.sin(-t / 2.0), 0.0]
        center = base + u * (wall_length / 2.0) - n * (wall_thickness / 2.0)
        self._wall = self._mjcf_root.worldbody.add(
            "geom",
            name="wall",
            type="box",
            size=half.tolist(),
            pos=center.tolist(),
            quat=quat,
            rgba=[0.45, 0.62, 0.81, 1.0],
            contype=1,
            conaffinity=1,
            solimp=[0.999, 0.999, 0.0001],
            solref=[0.001, 1.0],
            friction=[1.0, 0.1, 0.01],
        )

        mesh_asset = None
        vis_quat = None
        if hold_mesh is not None:
            mesh_asset = self._mjcf_root.asset.add(
                "mesh", name="hold_mesh", file=hold_mesh,
                scale=[hold_mesh_scale] * 3,
                maxhullvert=hold_mesh_maxhullvert,
            )
            def _qmul(a, b):
                w1, x1, y1, z1 = a
                w2, x2, y2, z2 = b
                return [
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
                    w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
                ]
            qy = [np.cos(-t / 2.0), 0.0, np.sin(-t / 2.0), 0.0]
            if hold_mesh_euler is not None:
                ex, ey, ez = (float(a) / 2.0 for a in hold_mesh_euler)
                q0 = _qmul(_qmul(
                    [np.cos(ex), np.sin(ex), 0.0, 0.0],
                    [np.cos(ey), 0.0, np.sin(ey), 0.0]),
                    [np.cos(ez), 0.0, 0.0, np.sin(ez)])
            else:
                q0 = [0.707061, 0.008016, -0.008016, 0.707061]
            vis_quat = _qmul(qy, q0)

        def _mesh_collides(r, c, idx):
            if not (hold_mesh_collider and mesh_asset is not None):
                return False
            if mesh_collider_holds is None:
                return True
            if isinstance(mesh_collider_holds, int):
                return idx < mesh_collider_holds
            return (r, c) in mesh_collider_holds

        rng = np.random.default_rng(hold_seed)
        y0 = -hold_col_dy * (hold_cols - 1) / 2.0

        if holds is not None:
            placements = [(0, i, float(v), float(y)) for i, (v, y) in enumerate(holds)]
        else:
            placements = []
            for r in range(hold_rows):
                for c in range(hold_cols):
                    v = hold_v0 + r * hold_row_dv
                    y = y0 + c * hold_col_dy
                    if hold_jitter > 0.0:
                        v += rng.uniform(-hold_jitter, hold_jitter)
                        y += rng.uniform(-hold_jitter, hold_jitter)
                    placements.append((r, c, v, y))

        if mirror_y:
            placements = [(r, c, v, -y) for r, c, v, y in placements]
        if hold_y_offset != 0.0:
            placements = [(r, c, v, y + hold_y_offset) for r, c, v, y in placements]
        if hold_dropout > 0.0:
            keep = (np.random.default_rng(hold_dropout_seed).random(len(placements))
                    >= hold_dropout)
            if not keep.any():
                raise ValueError(
                    "hold_dropout removed every hold; lower it or change the seed")
            placements = [pl for pl, k in zip(placements, keep) if k]

        self.hold_names = []
        self.mesh_collider_names = []
        for idx, (r, c, v, y) in enumerate(placements):
            pos = base + v * u + np.array([0.0, y, 0.0]) + hold_protrude * n
            nm = f"hold_{r}_{c}"
            self.hold_names.append(nm)
            mesh_col = _mesh_collides(r, c, idx)
            self._mjcf_root.worldbody.add(
                "geom",
                name=nm,
                type="sphere",
                size=[hold_radius],
                pos=pos.tolist(),
                contype=0 if mesh_col else hold_contype,
                conaffinity=0 if mesh_col else hold_conaffinity,
                condim=3,
                friction=[2.0, 0.5, 0.5],
                rgba=[0.95, 0.45, 0.1, 0.0 if mesh_asset is not None else 1.0],
                solimp=[0.99, 0.99, 0.001],
                solref=[0.005, 1.0],
            )
            if mesh_asset is not None:
                vis_pos = (base + v * u + np.array([0.0, y, 0.0])
                           + hold_mesh_protrude * n)
                mesh_kwargs = dict(
                    name=f"vis_{r}_{c}",
                    type="mesh",
                    mesh=mesh_asset,
                    pos=vis_pos.tolist(),
                    quat=vis_quat,
                    rgba=list(hold_mesh_rgba),
                )
                if mesh_col:
                    self.mesh_collider_names.append(f"vis_{r}_{c}")
                    mesh_kwargs.update(
                        contype=hold_contype,
                        conaffinity=hold_conaffinity,
                        condim=hold_mesh_condim,
                        friction=list(hold_mesh_friction),
                        solimp=list(hold_mesh_solimp),
                        solref=list(hold_mesh_solref),
                    )
                else:
                    mesh_kwargs.update(contype=0, conaffinity=0)
                self._mjcf_root.worldbody.add("geom", **mesh_kwargs)

        self._ground = self._mjcf_root.worldbody.add(
            "geom",
            name="ground",
            type="plane",
            size=[0, 0, 0.05],
            pos=[0, 0, 0],
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
