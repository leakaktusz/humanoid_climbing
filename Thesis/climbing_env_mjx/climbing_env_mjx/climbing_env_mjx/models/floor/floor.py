from dm_control import composer, mjcf

POS_GROUND=[0, 0, 0]

class Floor(composer.Entity):
    def _build(self, name="floor"):
        self._mjcf_root = mjcf.RootElement(model=name)
        self._mjcf_root.option.iterations = 100
        self._mjcf_root.option.ls_iterations = 100
        self._mjcf_root.asset.add(
            "texture", name="checker", type="2d", builtin="checker", mark="edge",
            rgb1=[1.0, 1.0, 1.0], rgb2=[0.75, 0.75, 0.75], markrgb=[0.6, 0.6, 0.6],
            width=300, height=300,
        )
        self._mjcf_root.asset.add(
            "material", name="checker", texture="checker",
            texrepeat=[5, 5], texuniform=True, reflectance=0.2,
        )
        self._ground = self._mjcf_root.worldbody.add(
            "geom",
            name="ground",
            type="plane",
            size=[0, 0, 0.05],
            pos=POS_GROUND,
            material="checker",
            contype=1,
            conaffinity=1,
            solimp=[0.999, 0.999, 0.0001],
            solref=[0.001, 1.0],
            friction=[1.0, 0.1, 0.01],
        )
    @property
    def mjcf_model(self):
        return self._mjcf_root
